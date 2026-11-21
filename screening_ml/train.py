"""Train the hit network on silver labels, validate on the gold practice labels.

    python -m screening_ml.train [--config screening_ml/config.yaml] [--device cuda|mps|cpu] [--rebuild]
"""
import argparse
import sys
import time

import numpy as np
import torch

from .common import gold_targets, load_config, report, save_checkpoint, silver_targets
from .features import build_cases
from .model import HitNet, decide, loss_fn, pick_device, set_seed


def split_indices(n, exclude, val_fraction, y, seed):
    """Stratified train/val split over silver labels, leaving out the gold hits."""
    rng = np.random.default_rng(seed)
    pool = np.array([i for i in range(n) if i not in set(exclude)])
    train, val = [], []
    for cls in np.unique(y[pool]):
        members = pool[y[pool] == cls]
        rng.shuffle(members)
        k = int(round(len(members) * val_fraction))
        val += list(members[:k])
        train += list(members[k:])
    return np.array(sorted(train)), np.array(sorted(val))


def run_epochs(model, X, y_dec, y_code, train_idx, val_idx, tcfg, device, class_weights=None, seed=0):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg["epochs"])
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad = float("inf"), None, 0
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
        model.eval()
        with torch.no_grad():
            dl, cl = model(X[val_idx])
            val_loss = loss_fn(dl, cl, y_dec[val_idx], y_code[val_idx], tcfg, class_weights).item()
        if val_loss < best - 1e-5:
            best, bad = val_loss, 0
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
    return best


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
    y_dec_np, y_code_np = silver_targets(cases)
    g_idx, g_dec, g_code, g_rows = gold_targets(cases, data["practice_labels"])
    print(f"features: {Xnp.shape[0]} hits x {Xnp.shape[1]} features; gold labels: {len(g_idx)}")

    exclude = g_idx if cfg["labels"].get("exclude_gold_from_train", True) else []
    train_idx, val_idx = split_indices(len(cases), exclude, cfg["labels"]["val_fraction"], y_dec_np, cfg["seed"])
    print(f"split: train {len(train_idx)} silver, val {len(val_idx)} silver, gold held out {len(exclude)}")

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
    run_epochs(model, X, y_dec, y_code, train_idx, val_idx, cfg["train"], device, weights, cfg["seed"])

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
    if len(g_idx):
        metrics["gold"] = report("gold (practice labels)", g_dec, dec[g_idx], g_code, code[g_idx], g_rows)
        metrics["teacher_on_gold"] = report("rule teacher on gold  ", g_dec, y_dec_np[g_idx], g_code, y_code_np[g_idx], g_rows)
    out = args.out or f"{cfg['artifacts_dir']}/model.pt"
    save_checkpoint(out, model, mean, std, cfg, data["feature_names"], metrics)
    print(f"\nsaved {out} ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
