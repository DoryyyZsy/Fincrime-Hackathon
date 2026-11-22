"""Train the hit network on silver labels, validate on grouped, non-practice cases.

    python -m screening_ml.train [--config screening_ml/config.yaml] [--device cuda|mps|cpu] [--rebuild]
"""
import argparse
import sys
import time

import numpy as np
import torch

from .common import apply_drops, load_config, report, save_checkpoint, silver_targets
from .features import build_cases
from .holdout import is_practice_hit
from .model import HitNet, decide, loss_fn, pick_device, set_seed


def leakage_groups(cases):
    """Group hits that would leak into each other: same alert, same customer person, or same watchlist name.

    Hits of one alert share the customer's features; one watchlist profile is often screened against several
    customers. Splitting such hits across train and validation/gold inflates the scores.
    """
    parent = list(range(len(cases)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first = {}
    for i, c in enumerate(cases):
        keys = [("alert", c.alert_id)]
        if c.customer_record_id not in (None, "AMBIGUOUS"):
            person = c.comparisons[0].person
            keys.append(("person", person.records[0] if person.records else c.customer_record_id))
        hit = c.comparisons[0].hit if c.comparisons else None
        if hit and hit.names:
            keys.append(("hit", " ".join(sorted(hit.names[0][1]))))
        for k in keys:
            if k in first:
                a, b = find(i), find(first[k])
                parent[max(a, b)] = min(a, b)
            else:
                first[k] = i
    return np.array([find(i) for i in range(len(cases))])


def split_indices(n, exclude, val_fraction, y, seed, groups=None):
    """Train/val split over silver labels that leaves out the gold hits.

    With `groups`, whole groups go to one side (no leakage) and every group touching a gold hit is removed
    from train and val too. Without it, a per-hit stratified split (leaky; kept for comparison).
    """
    rng = np.random.default_rng(seed)
    exclude = set(int(i) for i in exclude)
    if groups is None:
        pool = np.array([i for i in range(n) if i not in exclude])
        train, val = [], []
        for cls in np.unique(y[pool]):
            members = pool[y[pool] == cls]
            rng.shuffle(members)
            k = int(round(len(members) * val_fraction))
            val += list(members[:k])
            train += list(members[k:])
        return np.array(sorted(train)), np.array(sorted(val))
    banned = {groups[i] for i in exclude}
    pool_groups = sorted({g for i, g in enumerate(groups) if g not in banned})
    rng.shuffle(pool_groups)
    target, val_groups, size = val_fraction * sum(groups[i] not in banned for i in range(n)), set(), 0
    for g in pool_groups:  # fill validation with whole groups until it reaches the target size
        if size >= target:
            break
        val_groups.add(g)
        size += int((groups == g).sum())
    train = [i for i in range(n) if groups[i] not in banned and groups[i] not in val_groups]
    val = [i for i in range(n) if groups[i] in val_groups]
    return np.array(train), np.array(val)


def duplicate_rows(X, a, b):
    """How many rows in index set a have an identical feature vector somewhere in index set b."""
    seen = {X[i].tobytes() for i in b}
    return sum(X[i].tobytes() in seen for i in a)


def run_epochs(model, X, y_dec, y_code, train_idx, val_idx, tcfg, device, class_weights=None, seed=0):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg["epochs"])
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad, best_epoch = float("inf"), None, 0, 0
    for epoch in range(tcfg["epochs"]):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), generator=gen).numpy()]
        for start in range(0, len(order), tcfg["batch_size"]):
            b = order[start:start + tcfg["batch_size"]]
            dl, cl = model(X[b])
            loss = loss_fn(dl, cl, y_dec[b], y_code[b], tcfg, class_weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if val_idx is None:  # fixed-length run (final fit): no validation, keep the last weights
            continue
        model.eval()
        with torch.no_grad():
            dl, cl = model(X[val_idx])
            val_loss = loss_fn(dl, cl, y_dec[val_idx], y_code[val_idx], tcfg, class_weights).item()
        if val_loss < best - 1e-5:
            best, bad, best_epoch = val_loss, 0, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch % 25 == 0 or bad == 0 and epoch < 5:
            print(f"  epoch {epoch:3d} val_loss {val_loss:.4f}")
        if bad >= tcfg["patience"]:
            print(f"  early stop at epoch {epoch}, best val_loss {best:.4f}")
            break
    if best_state:
        model.load_state_dict(best_state)
    return best, best_epoch


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config")
    ap.add_argument("--device")
    ap.add_argument("--rebuild", action="store_true", help="rebuild features instead of using the cache")
    ap.add_argument("--out", help="checkpoint path (default artifacts/model.pt)")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config(args.config)
    set_seed(cfg["seed"], cfg.get("deterministic", True))
    device = pick_device(args.device or cfg.get("device", "auto"))
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    t0 = time.time()
    data = build_cases(cfg["data_dir"], cache_dir=f"{cfg['artifacts_dir']}/cache", use_cache=not args.rebuild)
    cases, Xnp = data["cases"], data["X"]
    groups = leakage_groups(cases)
    excluded_groups = {groups[i] for i, case in enumerate(cases) if is_practice_hit(case.hit_id)}
    eligible = np.flatnonzero(~np.isin(groups, list(excluded_groups)))
    groups = groups[eligible]
    Xnp = Xnp[eligible]
    cases = [cases[i] for i in eligible]
    y_dec_np, y_code_np = silver_targets(cases)
    train_idx, val_idx = split_indices(len(cases), [], cfg["labels"]["val_fraction"], y_dec_np, cfg["seed"], groups)
    print(f"split: {len(train_idx)} train, {len(val_idx)} validation; "
          f"{len(data['cases']) - len(eligible)} practice or linked hits excluded")
    Xnp, dropped = apply_drops(Xnp, data["feature_names"], cfg)  # ablation, see config.yaml
    if dropped:
        print(f"dropped features: {dropped}")

    mean = Xnp[train_idx].mean(0)
    std = Xnp[train_idx].std(0) + 1e-6
    X = torch.tensor((Xnp - mean) / std, dtype=torch.float32, device=device)
    y_dec = torch.tensor(y_dec_np, device=device)
    y_code = torch.tensor(y_code_np, device=device)

    weights = None
    if cfg["train"].get("balance_classes", True):
        counts = np.bincount(y_dec_np[train_idx], minlength=3).astype(np.float32)
        weights = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), device=device)

    model = HitNet(X.shape[1], tuple(cfg["model"]["hidden"]), cfg["model"]["dropout"]).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters())} parameters")
    _, best_epoch = run_epochs(model, X, y_dec, y_code, train_idx, val_idx, cfg["train"], device, weights, cfg["seed"])

    model.eval()
    with torch.no_grad():
        dl, cl = model(X)
        dec, code, _ = decide(dl, cl, cfg["inference"])
    dec, code = dec.cpu().numpy(), code.cpu().numpy()
    print("\n== results (network alone, before rule_guard)")
    metrics = {
        "silver_train": report("silver train", y_dec_np[train_idx], dec[train_idx], y_code_np[train_idx], code[train_idx]),
        "silver_val": report("silver val  ", y_dec_np[val_idx], dec[val_idx], y_code_np[val_idx], code[val_idx]),
    }
    out = args.out or f"{cfg['artifacts_dir']}/model.pt"

    if cfg["train"].get("final_fit", True) and best_epoch:
        # The numbers above are the honest, leak-free estimate. For the model we ship, refit from scratch on
        # every non-gold hit for the epoch count early stopping chose; there is no held-out set left after this.
        save_checkpoint(out.replace(".pt", "_eval.pt"), model, mean, std, cfg, data["feature_names"], metrics)
        set_seed(cfg["seed"], cfg.get("deterministic", True))
        fit_idx = np.concatenate((train_idx, val_idx))
        mean = Xnp[fit_idx].mean(0)
        std = Xnp[fit_idx].std(0) + 1e-6
        X = torch.tensor((Xnp - mean) / std, dtype=torch.float32, device=device)
        if weights is not None:
            counts = np.bincount(y_dec_np[fit_idx], minlength=3).astype(np.float32)
            weights = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), device=device)
        model = HitNet(X.shape[1], tuple(cfg["model"]["hidden"]), cfg["model"]["dropout"]).to(device)
        run_epochs(model, X, y_dec, y_code, fit_idx, None, dict(cfg["train"], epochs=best_epoch), device, weights, cfg["seed"])
        model.eval()
        with torch.no_grad():
            dl, cl = model(X)
            dec, code, _ = decide(dl, cl, cfg["inference"])
        dec, code = dec.cpu().numpy(), code.cpu().numpy()
        metrics["final_fit"] = {"epochs": best_epoch, "n": len(fit_idx)}
    save_checkpoint(out, model, mean, std, cfg, data["feature_names"], metrics)
    print(f"\nsaved {out} ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
