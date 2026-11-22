"""How the Fellegi–Sunter score behaves on a data folder. It does not change decisions.

    python -m screening_ml.eval_linkage --data data/screening/dev

The development folder is git-ignored. If it is not in this checkout the script
stops and says so. It does not invent a score for data it cannot read.

With practice_labels.jsonl it also reports whether the probability ranks
same-person labels above different-person labels, and what would happen if
review hits below a cost threshold were cleared. That last table is
counterfactual: the engine does not apply it.
"""
import argparse
import os
import sys

from .linkage import CLEAR_PROBABILITY_CEILING, PROMOTE
from .pipeline import adjudicate


def _mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def _median(values):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


def _fmt(value):
    if value is None:
        return "—"
    return f"{value:.4f}"


def _print_group(title, rows):
    print(f"\n{title}")
    print(f"  {'group':<20} {'n':>5} {'mean':>8} {'median':>8} {'min':>8} {'max':>8}")
    for name in sorted({row[0] for row in rows}):
        values = [value for label, value in rows if label == name]
        print(f"  {name:<20} {len(values):>5} {_fmt(_mean(values)):>8} {_fmt(_median(values)):>8} "
              f"{_fmt(min(values)):>8} {_fmt(max(values)):>8}")


def report(folder):
    from .data import DataRoom

    room = DataRoom(folder)
    # HitNet decides. The score is computed beside that decision and is not written back onto it.
    rows, errors, _stats, scores, fit = adjudicate(room, None)
    if not scores or fit is None:
        print("The match score did not fit. Decisions were still produced.")
        return 1
    print(f"hits: {len(rows)}  errors sent to review: {errors}")
    print(f"EM iterations: {fit.iterations}  converged: {fit.converged}  "
          f"P(match) prior after fit: {fit.lambda_:.4f}  pairs: {fit.n_pairs}")
    if getattr(fit, "log_likelihood", None) is not None:
        print(f"observed-data log-likelihood: {fit.log_likelihood:.2f}")
    print(f"promotion: {'on' if PROMOTE else 'off'}  "
          f"cost ceiling for a clearance P(unsafe) < {CLEAR_PROBABILITY_CEILING:.2f} (not applied)")
    changed = [
        (row["hit_id"], row["decision"], score["decision"])
        for row, score in zip(rows, scores) if row["decision"] != score["decision"]
    ]
    print(f"decisions changed by the score: {len(changed)}")
    _print_group("By the HitNet decision", [(row["decision"], score["match_probability"]) for row, score in zip(rows, scores)])

    ranked = [score for score in scores if score["decision"] == "REVIEW"]
    print("\nReview queue, highest P(same) first (top 10):")
    if not ranked:
        print("  no REVIEW hits")
    for score in sorted(ranked, key=lambda item: (item["review_rank"] is None, item["review_rank"] or 0))[:10]:
        levels = ", ".join(f"{key}={value}" for key, value in score["levels"].items())
        print(f"  #{score['review_rank']:<3} {score['hit_id']}  p={score['match_probability']:.4f}  "
              f"weight={score['match_weight']:.2f}  {levels}")

    labels = room.practice_labels
    if not labels:
        print("\nNo practice_labels.jsonl in this folder, so the score was not compared with gold labels.")
        return 0

    gold = {row["hit_id"]: row["decision"] for row in labels}
    paired = [(gold[score["hit_id"]], score["match_probability"]) for score in scores if score["hit_id"] in gold]
    print(f"\nPractice labels present: {len(labels)}  scored: {len(paired)}")
    _print_group("By the gold decision", paired)
    same = [value for label, value in paired if label == "SAME_ENTITY"]
    different = [value for label, value in paired if label == "DIFFERENT_ENTITY"]
    if same and different:
        total = len(same) * len(different)
        wins = sum(1 for left in same for right in different if left > right)
        ties = sum(1 for left in same for right in different if left == right)
        print(f"pairwise: a gold SAME_ENTITY outranks a gold DIFFERENT_ENTITY on {wins}/{total} pairs ({ties} ties)")

    print("\nCounterfactual: clearing a HitNet REVIEW hit when P(same) is below a threshold.")
    print("The engine does not do this. The score is not allowed to change HitNet's decision.")
    unsafe_anywhere = False
    for threshold in (CLEAR_PROBABILITY_CEILING, 0.02, 0.01):
        picked = [score for score in scores if score["decision"] == "REVIEW" and score["match_probability"] < threshold]
        unsafe = sum(1 for score in picked if gold.get(score["hit_id"]) in ("SAME_ENTITY", "REVIEW"))
        true_clear = sum(1 for score in picked if gold.get(score["hit_id"]) == "DIFFERENT_ENTITY")
        unlabelled = sum(1 for score in picked if score["hit_id"] not in gold)
        if unsafe:
            unsafe_anywhere = True
        print(f"  p < {threshold:.2f}: {len(picked)} review hits  gold unsafe {unsafe}  "
              f"gold different {true_clear}  unlabelled {unlabelled}")
    if unsafe_anywhere:
        print("Promotion stays off: at least one threshold would clear a practice hit that is not a different person.")
    else:
        print("No labelled unsafe clearance in this sweep. Promotion stays off anyway: most hits are unlabelled, "
              "the unseen set is not here, and the score does not clear a HitNet review.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=os.path.join("data", "screening", "dev"))
    args = parser.parse_args(argv)
    hits = os.path.join(args.data, "hits.jsonl")
    if not os.path.isdir(args.data) or not os.path.exists(hits):
        print(f"No screening data at {args.data} (hits.jsonl is missing).")
        print("The dev folder is git-ignored and is not in every checkout, so this run has no score to report.")
        print("When the folder is available:")
        print(f"  python -m screening_ml.eval_linkage --data {args.data}")
        print("  python inspector.py")
        print("  python -m unittest tests/test_linkage.py")
        return 2
    return report(args.data)


if __name__ == "__main__":
    sys.exit(main())
