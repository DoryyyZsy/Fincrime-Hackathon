"""Shared helpers: config, label arrays, metrics in the organisers' terms, checkpoints."""
import json
import os

import numpy as np
import torch
import yaml

from .compare import DECISIONS, REASON_CODES
from .model import COST, HitNet

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config(path=None):
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_drops(X, feature_names, cfg):
    """Zero the features listed in cfg['drop_features'] (same at train, predict and fine-tune time)."""
    cols = [feature_names.index(f) for f in cfg.get("drop_features") or [] if f in feature_names]
    if not cols:
        return X, []
    X = X.copy()
    X[:, cols] = 0.0
    return X, [feature_names[c] for c in cols]


def silver_targets(cases):
    y_dec = np.array([DECISIONS.index(c.teacher_decision) for c in cases])
    y_code = np.array([REASON_CODES.index(c.teacher_code) for c in cases])
    return y_dec, y_code


def gold_targets(cases, labels):
    """Map gold labels onto case indices. Returns (indices, decision ids, code ids, label rows)."""
    pos = {c.hit_id: i for i, c in enumerate(cases)}
    rows = [l for l in labels if l["hit_id"] in pos]
    idx = np.array([pos[l["hit_id"]] for l in rows], dtype=int)
    y_dec = np.array([DECISIONS.index(l["decision"]) for l in rows], dtype=int)
    y_code = np.array([REASON_CODES.index(l["reason_codes"][0]) for l in rows], dtype=int)
    return idx, y_dec, y_code, rows


def decision_score(y_true, y_pred):
    """The organisers' decision score: share of cost saved versus sending everything to REVIEW (0..100)."""
    cost = COST.numpy()
    total = sum(cost[t, p] for t, p in zip(y_true, y_pred))
    baseline = sum(cost[t, DECISIONS.index("REVIEW")] for t in y_true)
    return float(100 * max(0.0, 1 - total / baseline)) if baseline else 0.0


def report(name, y_true, y_pred, code_true=None, code_pred=None, gold_rows=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    acc = (y_true == y_pred).mean() if len(y_true) else 0.0
    diff = DECISIONS.index("DIFFERENT_ENTITY")
    unsafe = int(((y_pred == diff) & (y_true != diff)).sum())
    line = f"{name}: n={len(y_true)} acc={acc:.1%} decision_score={decision_score(y_true, y_pred):.1f} unsafe_clears={unsafe}"
    right = y_true == y_pred
    if gold_rows is not None and code_pred is not None and right.any():  # gold: any listed code is accepted
        ok = [REASON_CODES[c] in row["reason_codes"] for c, row, r in zip(code_pred, gold_rows, right) if r]
        line += f" reason_code_ok={np.mean(ok):.0%}"
    elif code_true is not None and code_pred is not None and right.any():
        line += f" reason_code_ok={(np.asarray(code_true)[right] == np.asarray(code_pred)[right]).mean():.0%}"
    print(line)
    confusion = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1
    print("    confusion (rows=true, cols=pred: DIFF SAME REVIEW): " + " | ".join(" ".join(f"{v:3d}" for v in r) for r in confusion))
    return {"n": int(len(y_true)), "acc": float(acc), "decision_score": decision_score(y_true, y_pred), "unsafe": unsafe}


def save_checkpoint(path, model, mean, std, cfg, feature_names, metrics):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "config": cfg,
                "feature_names": feature_names, "metrics": metrics}, path)
    with open(os.path.splitext(path)[0] + "_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    mcfg = ckpt["config"]["model"]
    model = HitNet(len(ckpt["feature_names"]), tuple(mcfg["hidden"]), mcfg["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
