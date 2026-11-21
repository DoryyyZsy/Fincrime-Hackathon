"""Audit a data folder before training: files, sizes, joins, label coverage, and whether labels suffice.

    python -m screening_ml.check_data --data data/screening/dev
"""
import argparse
import collections
import os
import sys

from .data import TABLES, read_jsonl

REQUIRED = TABLES + ["name_variants.json"]
# rough rule of thumb for a small tabular classifier: examples per class and per input feature
MIN_PER_CLASS = 100
MIN_PER_FEATURE = 10


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/screening/dev")
    ap.add_argument("--n-features", type=int, default=None, help="feature count (default: read from features.py)")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    ok = True

    print(f"== 1. files in {args.data}")
    for name in REQUIRED:
        path = os.path.join(args.data, name if name.endswith(".json") else f"{name}.jsonl")
        exists = os.path.exists(path)
        ok &= exists
        print(f"  {'ok ' if exists else 'MISSING'} {os.path.basename(path)}"
              + (f"  ({os.path.getsize(path) / 1e6:.1f} MB)" if exists else ""))
    if not ok:
        print("required files are missing; stopping")
        return 1
    t = {name: read_jsonl(os.path.join(args.data, f"{name}.jsonl")) for name in TABLES}
    labels_path = os.path.join(args.data, "practice_labels.jsonl")
    labels = read_jsonl(labels_path) if os.path.exists(labels_path) else []

    print("\n== 2. row counts")
    for name, rows in t.items():
        print(f"  {name:16} {len(rows):>7}")

    print("\n== 3. integrity")
    hit_ids = {h["hit_id"] for h in t["hits"]}
    alert_ids = {a["alert_id"] for a in t["alerts"]}
    listed = [hid for a in t["alerts"] for hid in a["hit_ids"]]
    issues = {
        "duplicate hit_id": len(t["hits"]) - len(hit_ids),
        "duplicate alert_id": len(t["alerts"]) - len(alert_ids),
        "alert lists unknown hit": sum(h not in hit_ids for h in listed),
        "hit points to unknown alert": sum(h["alert_id"] not in alert_ids for h in t["hits"]),
        "duplicate customer record_id": len(t["customers"]) - len({c["record_id"] for c in t["customers"]}),
    }
    accounts = collections.Counter(a["account_id"] for a in t["accounts"])
    for k, v in issues.items():
        print(f"  {'ok ' if v == 0 else 'WARN'} {k}: {v}")
    print(f"  info joint accounts (several holders): {sum(v > 1 for v in accounts.values())}")
    links = collections.Counter("account" if a.get("account_ref") else "login" if a.get("login_handle")
                                else "email" if a.get("email") else "none" for a in t["alerts"])
    print(f"  info alert link types: {dict(links)}")

    print("\n== 4. labels")
    print(f"  hits to decide: {len(hit_ids)}")
    print(f"  labelled hits (practice_labels.jsonl): {len(labels)}  ({len(labels) / max(len(hit_ids), 1):.1%} of hits)")
    unknown = [l["hit_id"] for l in labels if l["hit_id"] not in hit_ids]
    if unknown:
        print(f"  WARN {len(unknown)} labels point at hits not in this folder")
    dec = collections.Counter(l["decision"] for l in labels)
    codes = collections.Counter(c for l in labels for c in l["reason_codes"][:1])
    print(f"  decisions: {dict(dec)}")
    print(f"  main reason codes: {dict(codes)}")

    if args.n_features is None:
        from .features import FEATURE_NAMES
        n_features = len(FEATURE_NAMES)
    else:
        n_features = args.n_features
    print("\n== 5. is that enough to train a neural network?")
    smallest = min(dec.values()) if dec else 0
    checks = [
        (f"examples per decision class >= {MIN_PER_CLASS}", smallest, smallest >= MIN_PER_CLASS),
        (f"examples per feature >= {MIN_PER_FEATURE} ({n_features} features)", len(labels) / max(n_features, 1),
         len(labels) >= MIN_PER_FEATURE * n_features),
        ("all 11 reason codes seen", len(codes), len(codes) >= 11),
    ]
    enough = True
    for text, value, passed in checks:
        enough &= passed
        print(f"  {'ok  ' if passed else 'FAIL'} {text}: {value:.0f}" if isinstance(value, float) else
              f"  {'ok  ' if passed else 'FAIL'} {text}: {value}")
    if enough:
        print("\n  VERDICT: enough gold labels for supervised training.")
    else:
        print("\n  VERDICT: NOT enough gold labels to train a network on them alone.\n"
              "  Plan used by train.py: label every hit with the SOP rule engine (silver labels, compare.py),\n"
              "  train the network on silver labels, and keep the gold labels only for validation.\n"
              "  Add more gold labels (same format as practice_labels.jsonl) and use finetune.py to go beyond the rules.")
        try:
            from .features import build_cases
            data = build_cases(args.data, verbose=False)
            by_id = {c.hit_id: c for c in data["cases"]}
            agree = sum(by_id[l["hit_id"]].teacher_decision == l["decision"] for l in labels if l["hit_id"] in by_id)
            silver = collections.Counter(c.teacher_decision for c in data["cases"])
            print(f"\n  silver labels: {len(data['cases'])} hits, decisions {dict(silver)}")
            print(f"  silver vs gold agreement on the practice hits: {agree}/{len(labels)}")
        except Exception as error:  # the audit should still finish if feature building breaks
            print(f"  (could not build silver labels: {error})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
