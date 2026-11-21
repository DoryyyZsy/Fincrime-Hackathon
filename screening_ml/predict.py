"""Predict every hit in a data folder and write the submission file (one JSON line per hit).

    python -m screening_ml.predict --data data/screening/dev --out output/dev.jsonl [--checkpoint artifacts/model.pt]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

from .common import load_checkpoint, load_config
from .compare import CLEAR_ORDER, DECISIONS, REASON_CODES, evidence_for
from .features import build_cases
from .model import decide, pick_device


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/screening/dev")
    ap.add_argument("--out", default="output/dev.jsonl")
    ap.add_argument("--checkpoint", default="artifacts/model.pt")
    ap.add_argument("--config", help="override inference settings from this config")
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    device = pick_device(args.device or "auto")
    model, ckpt = load_checkpoint(args.checkpoint, device)
    icfg = (load_config(args.config) if args.config else ckpt["config"])["inference"]
    data = build_cases(args.data, cache_dir=os.path.join(os.path.dirname(args.checkpoint) or ".", "cache"))
    if data["feature_names"] != ckpt["feature_names"]:
        raise SystemExit("feature list changed since this checkpoint was trained; retrain first")
    cases = data["cases"]
    X = torch.tensor((data["X"] - ckpt["mean"]) / ckpt["std"], dtype=torch.float32, device=device)
    with torch.no_grad():
        dl, cl = model(X)
        dec, code, probs = decide(dl, cl, icfg)
    dec, code, probs = dec.cpu().numpy(), code.cpu().numpy(), probs.cpu().numpy()

    guarded = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for i, case in enumerate(cases):
            decision, reason_code = DECISIONS[dec[i]], REASON_CODES[code[i]]
            if decision == "DIFFERENT_ENTITY" and icfg.get("rule_guard", True):
                fired = [c for c in CLEAR_ORDER if case.comparisons and case.comparisons[0].checks[c].fired]
                if case.teacher_decision != "DIFFERENT_ENTITY" or not fired:
                    decision, reason_code, guarded = "REVIEW", case.teacher_code if case.teacher_decision == "REVIEW" else "INSUFFICIENT_DATA", guarded + 1
                elif reason_code not in fired:
                    reason_code = fired[0]
            evidence, reason = evidence_for(case, reason_code)
            check = case.comparisons[0].checks.get(reason_code) if case.comparisons else None
            if decision != case.teacher_decision and not (check and check.fired):
                # the network went beyond the rules: say so honestly instead of borrowing the rule's wording
                reason = (f"The model chose {decision} with {probs[i].max():.0%} confidence although no SOP rule "
                          f"settles it. Rule view: " + reason)
            row = {
                "hit_id": case.hit_id, "alert_id": case.alert_id, "decision": decision, "reason_code": reason_code,
                "customer_record_id": case.customer_record_id, "reason": reason, "evidence": evidence,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {d: 0 for d in DECISIONS}
    with open(args.out, encoding="utf-8") as handle:
        for line in handle:
            counts[json.loads(line)["decision"]] += 1
    agree = sum(1 for i, c in enumerate(cases) if DECISIONS[dec[i]] == c.teacher_decision)
    print(f"wrote {len(cases)} lines to {args.out}: {counts}")
    print(f"network agrees with the rule teacher on {agree}/{len(cases)} hits; rule_guard blocked {guarded} clearances")
    return 0


if __name__ == "__main__":
    sys.exit(main())
