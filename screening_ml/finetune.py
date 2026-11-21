"""Fine-tune a trained checkpoint on gold labels (practice_labels.jsonl format).

Use this when you have more human-checked labels than the rule teacher can explain. With freeze_backbone
only the two heads move, which is the safe choice for a few dozen labels.

    python -m screening_ml.finetune [--labels my_labels.jsonl] [--checkpoint artifacts/model.pt]
"""
import argparse
import sys

import numpy as np
import torch

from .common import gold_targets, load_checkpoint, load_config, report, save_checkpoint
from .data import read_jsonl
from .features import build_cases
from .model import decide, pick_device, set_seed
from .train import run_epochs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config")
    ap.add_argument("--checkpoint", default="artifacts/model.pt")
    ap.add_argument("--labels")
    ap.add_argument("--out", default="artifacts/model_finetuned.pt")
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config(args.config)
    fcfg = cfg["finetune"]
    set_seed(cfg["seed"], cfg.get("deterministic", True))
    device = pick_device(args.device or cfg.get("device", "auto"))
    model, ckpt = load_checkpoint(args.checkpoint, device)

    data = build_cases(cfg["data_dir"], cache_dir=f"{cfg['artifacts_dir']}/cache")
    labels = read_jsonl(args.labels or fcfg["labels"])
    idx, y_dec_g, y_code_g, rows = gold_targets(data["cases"], labels)
    if len(idx) < 10:
        raise SystemExit(f"only {len(idx)} usable labels; need at least 10")
    rng = np.random.default_rng(cfg["seed"])
    perm = rng.permutation(len(idx))
    k = max(1, int(round(len(idx) * fcfg["holdout_fraction"])))
    hold, fit = perm[:k], perm[k:]
    print(f"device: {device}; labels: {len(idx)} (fine-tune on {len(fit)}, hold out {len(hold)})")

    X = torch.tensor((data["X"] - ckpt["mean"]) / ckpt["std"], dtype=torch.float32, device=device)
    y_dec = torch.zeros(len(X), dtype=torch.long, device=device)
    y_code = torch.zeros(len(X), dtype=torch.long, device=device)
    y_dec[idx] = torch.tensor(y_dec_g, device=device)
    y_code[idx] = torch.tensor(y_code_g, device=device)

    def evaluate(tag):
        model.eval()
        with torch.no_grad():
            dl, cl = model(X[idx[hold]])
            dec, code, _ = decide(dl, cl, cfg["inference"])
        return report(tag, y_dec_g[hold], dec.cpu().numpy(), y_code_g[hold], code.cpu().numpy(), [rows[i] for i in hold])

    before = evaluate("held-out labels BEFORE fine-tune")
    if fcfg.get("freeze_backbone", True):
        for p in model.backbone.parameters():
            p.requires_grad = False
    tcfg = dict(cfg["train"], epochs=fcfg["epochs"], lr=fcfg["lr"], patience=fcfg["epochs"])
    # validation for early stopping = the fit set itself (too few labels to split again)
    run_epochs(model, X, y_dec, y_code, idx[fit], idx[fit], tcfg, device, None, cfg["seed"])
    after = evaluate("held-out labels AFTER fine-tune ")
    ckpt["config"]["finetune_used"] = fcfg
    save_checkpoint(args.out, model, ckpt["mean"], ckpt["std"], ckpt["config"], ckpt["feature_names"],
                    {"before": before, "after": after, "base_metrics": ckpt["metrics"]})
    print(f"saved {args.out}. Keep it only if the held-out score did not drop.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
