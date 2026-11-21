#!/usr/bin/env python3
"""Perturb a data folder and measure whether clearances stay safe.

The development set is participant-only and is not in the repository. Point this
script at a local copy:

    python3 tests/perturb_screening.py <data folder> [--system system/run]

Each perturbation is a copy of that folder. The script runs system/run, scores
practice labels with validate.py, and counts decisions that flip to
DIFFERENT_ENTITY against an unperturbed run of the same system. Exit status is
1 when any perturbation produces an unsafe clearance (a DIFFERENT_ENTITY whose
practice label is SAME_ENTITY or REVIEW).

Perturbations
  shift_kyc_dates     KYC-note created_at += 2 days; the date written in the note stays
  residence_wording   lived/work/resid/... in non-KYC notes become stayed/job/...
  age_wording         age phrases in hit profile notes are paraphrased
  rejected_kyc        KYC notes say the check did not pass (wording the old pattern
                      missed) and the stated birth year moves to 2019
"""
import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COST = {
    "DIFFERENT_ENTITY": {"DIFFERENT_ENTITY": 0, "SAME_ENTITY": 3, "REVIEW": 3},
    "SAME_ENTITY": {"DIFFERENT_ENTITY": 60, "SAME_ENTITY": 0, "REVIEW": 2},
    "REVIEW": {"DIFFERENT_ENTITY": 60, "SAME_ENTITY": 2, "REVIEW": 0},
}

_RESIDENCE = [
    (re.compile(r"\bresidences?\b", re.I), "stays"),
    (re.compile(r"\bresidency\b", re.I), "a stay"),
    (re.compile(r"\bresidents?\b", re.I), "a visitor"),
    (re.compile(r"\bresided\b", re.I), "stayed"),
    (re.compile(r"\bresiding\b", re.I), "staying"),
    (re.compile(r"\breside\b", re.I), "stay"),
    (re.compile(r"\bliving\b", re.I), "staying"),
    (re.compile(r"\blived\b", re.I), "stayed"),
    (re.compile(r"\blives\b", re.I), "stays"),
    (re.compile(r"\bemployed\b", re.I), "in a job"),
    (re.compile(r"\bemployment\b", re.I), "a job"),
    (re.compile(r"\bemployer\b", re.I), "the job"),
    (re.compile(r"\bworking\b", re.I), "doing a job"),
    (re.compile(r"\bworked\b", re.I), "did a job"),
    (re.compile(r"\bworks\b", re.I), "does a job"),
    (re.compile(r"\bwork\b", re.I), "job"),
    (re.compile(r"\bhouseholds?\b", re.I), "place"),
    (re.compile(r"\bhomes?\b", re.I), "place"),
    (re.compile(r"\bbased\b", re.I), "present"),
    (re.compile(r"\blocated\b", re.I), "situated"),
    (re.compile(r"\brelocated\b", re.I), "shifted"),
    (re.compile(r"\brelocat(?:e|ing|ion)\b", re.I), "shift"),
    (re.compile(r"\bmoved\b", re.I), "went"),
    (re.compile(r"\bdwelling\b", re.I), "staying"),
]


def _shift_iso(value, days):
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})(.*)", value or "")
    if not match:
        return value
    shifted = datetime.date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    shifted += datetime.timedelta(days=days)
    return shifted.isoformat() + match.group(4)


def shift_kyc_dates(row):
    if row.get("channel") == "kyc" and row.get("created_at"):
        row["created_at"] = _shift_iso(row["created_at"], 2)
    return row


def paraphrase_residence(row):
    if row.get("channel") == "kyc":
        return row
    text = row.get("text") or ""
    for pattern, replacement in _RESIDENCE:
        text = pattern.sub(replacement, text)
    row["text"] = text
    return row


def paraphrase_age(text):
    """Same facts, different age wording. Each form is one the extractor still reads."""
    text = re.sub(r"\b(\d{1,3})\s+years?\s+old\b", r"\1 years of age", text, flags=re.I)
    text = re.sub(r"\b(\d{1,3})-year-old\b", r"\1 year old", text, flags=re.I)
    text = re.sub(r"\bage\s+of\s+(\d{1,3})\b", r"age \1", text, flags=re.I)
    text = re.sub(r"\baged\s+(\d{1,3})\b", r"age \1", text, flags=re.I)
    text = re.sub(
        r"\b(he|she|they)\s+(was|were|is)\s+(\d{1,3})\b",
        r"\1 \2 age \3",
        text,
        flags=re.I,
    )
    text = re.sub(r"\bwas\s+(\d{1,3})\s+when\b", r"at age \1 when", text, flags=re.I)
    text = re.sub(r"([^\W\d_]),\s*(\d{1,3}),", r"\1, age \2,", text)
    return text


def paraphrase_hit_ages(row):
    if row.get("profile_notes"):
        row["profile_notes"] = paraphrase_age(row["profile_notes"])
    return row


def reject_kyc(row):
    """Rejection wording the previous pattern did not match, plus a wrong birth year.

    'not verified' / 'did not pass' / 'resubmit' are not in the old
    reject|fail|expired|declin|... expression, so that engine still reads the note.
    """
    if row.get("channel") != "kyc":
        return row
    text = row.get("text") or ""

    def _year(match):
        year = int(match.group(0))
        return "2019" if year <= 2012 else match.group(0)

    text = re.sub(r"\b(?:19|20)\d{2}\b", _year, text)
    row["text"] = (
        "Identity was not verified; the applicant did not pass and must resubmit. " + text
    )
    return row


PERTURBATIONS = [
    ("shift_kyc_dates", "notes.jsonl", shift_kyc_dates),
    ("residence_wording", "notes.jsonl", paraphrase_residence),
    ("age_wording", "hits.jsonl", paraphrase_hit_ages),
    ("rejected_kyc", "notes.jsonl", reject_kyc),
]


def _copy_room(src, dst, filename, mutate):
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        source = os.path.join(src, name)
        target = os.path.join(dst, name)
        if name == filename:
            with open(source, encoding="utf-8") as handle, open(target, "w", encoding="utf-8") as out:
                for line in handle:
                    if not line.strip():
                        continue
                    row = mutate(json.loads(line))
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
        elif os.path.isfile(source):
            os.symlink(source, target)


def _read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _run(system, folder, output, extra_env=None):
    env = os.environ.copy()
    env["SCREENING_LLM"] = "0"
    if extra_env:
        env.update(extra_env)
    subprocess.run([sys.executable, system, folder, output], check=True, stdout=subprocess.DEVNULL, env=env)


def _score(validate, folder, output, labels):
    completed = subprocess.run(
        [sys.executable, validate, "--data", folder, "--output", output, "--score", labels],
        check=False,
        capture_output=True,
        text=True,
    )
    text = completed.stdout + completed.stderr
    return completed.returncode, text


def _metrics(output, labels, baseline):
    answers = {row["hit_id"]: row for row in _read_jsonl(output)}
    practice = _read_jsonl(labels)
    cost = sum(COST[row["decision"]][answers.get(row["hit_id"], {}).get("decision", "REVIEW")] for row in practice)
    base_cost = sum(COST[row["decision"]]["REVIEW"] for row in practice)
    right = [row for row in practice if answers.get(row["hit_id"], {}).get("decision") == row["decision"]]
    unsafe = [
        row["hit_id"] for row in practice
        if row["decision"] != "DIFFERENT_ENTITY"
        and answers.get(row["hit_id"], {}).get("decision") == "DIFFERENT_ENTITY"
    ]
    codes = sum(1 for row in right if answers[row["hit_id"]].get("reason_code") in row["reason_codes"])
    records = sum(
        1 for row in right
        if answers[row["hit_id"]].get("customer_record_id") in row["customer_record_ids"]
        or (row["customer_record_ids"] == ["AMBIGUOUS"]
            and answers[row["hit_id"]].get("customer_record_id") in (None, "", "AMBIGUOUS"))
    )
    base = {row["hit_id"]: row for row in _read_jsonl(baseline)}
    flips = [
        hit_id for hit_id, row in answers.items()
        if row["decision"] == "DIFFERENT_ENTITY" and base.get(hit_id, {}).get("decision") != "DIFFERENT_ENTITY"
    ]
    decisions = {}
    for row in answers.values():
        decisions[row["decision"]] = decisions.get(row["decision"], 0) + 1
    return {
        "right": len(right),
        "n": len(practice),
        "unsafe": len(unsafe),
        "unsafe_ids": unsafe,
        "score": 100 * max(0.0, 1 - cost / base_cost) if base_cost else 0.0,
        "reason": (codes / len(right)) if right else 0.0,
        "record": (records / len(right)) if right else 0.0,
        "flips": len(flips),
        "decisions": decisions,
        "problems": None,
    }


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", help="data folder (dev/)")
    parser.add_argument("--system", default=os.path.join(ROOT, "system", "run"))
    parser.add_argument("--validate", default=os.path.join(ROOT, "data", "screening", "validate.py"))
    parser.add_argument("--labels", help="practice_labels.jsonl (default: <data>/practice_labels.jsonl)")
    args = parser.parse_args(argv)
    labels = args.labels or os.path.join(args.data, "practice_labels.jsonl")
    if not os.path.isdir(args.data) or not os.path.exists(labels):
        sys.exit(f"need a data folder with practice_labels.jsonl, got {args.data}")

    work = tempfile.mkdtemp(prefix="perturb-screening-")
    try:
        baseline = os.path.join(work, "baseline.jsonl")
        print(f"unperturbed run via {args.system}", file=sys.stderr)
        _run(args.system, args.data, baseline)
        code, text = _score(args.validate, args.data, baseline, labels)
        base = _metrics(baseline, labels, baseline)
        base["problems"] = code
        rules_outputs = {"unperturbed": baseline}
        rows = [("unperturbed", base, text)]
        for name, filename, mutate in PERTURBATIONS:
            folder = os.path.join(work, name)
            output = os.path.join(work, name + ".jsonl")
            print(f"perturbation {name}", file=sys.stderr)
            _copy_room(args.data, folder, filename, mutate)
            _run(args.system, folder, output)
            code, text = _score(args.validate, folder, output, labels)
            metrics = _metrics(output, labels, baseline)
            metrics["problems"] = code
            rows.append((name, metrics, text))
            rules_outputs[name] = output

        print(f"{'perturbation':<20} {'right':>8} {'score':>7} {'unsafe':>7} {'flips':>7} {'problems':>9}  decisions")
        failed = False
        for name, metrics, _text in rows:
            decisions = " ".join(f"{key[0]}={value}" for key, value in sorted(metrics["decisions"].items()))
            print(
                f"{name:<20} {metrics['right']:>3}/{metrics['n']:<4} {metrics['score']:>6.1f} "
                f"{metrics['unsafe']:>7} {metrics['flips']:>7} {metrics['problems']:>9}  {decisions}"
            )
            if metrics["unsafe"]:
                failed = True
                print(f"  unsafe: {', '.join(metrics['unsafe_ids'])}")
            if metrics["problems"]:
                failed = True

        # Rules + an offline model stand-in. No OpenRouter key is required. The stand-in
        # agrees with the regex (or says unknown), so it must not open a new clearance.
        print("\nrules+mock-llm (SCREENING_LLM_MOCK=1, no network)", file=sys.stderr)
        mock_env = {"SCREENING_LLM": "1", "SCREENING_LLM_MOCK": "1"}
        llm_rows = []
        for name, filename, mutate in [("unperturbed", None, None)] + list(PERTURBATIONS):
            if name == "unperturbed":
                folder = args.data
            else:
                folder = os.path.join(work, "llm-" + name)
                _copy_room(args.data, folder, filename, mutate)
            output = os.path.join(work, "llm-" + name + ".jsonl")
            _run(args.system, folder, output, mock_env)
            code, _text = _score(args.validate, folder, output, labels)
            metrics = _metrics(output, labels, rules_outputs[name])
            metrics["problems"] = code
            llm_rows.append((name, metrics))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"{'rules+mock':<20} {'right':>8} {'score':>7} {'unsafe':>7} {'newclear':>9} {'problems':>9}")
    for name, metrics in llm_rows:
        print(
            f"{name:<20} {metrics['right']:>3}/{metrics['n']:<4} {metrics['score']:>6.1f} "
            f"{metrics['unsafe']:>7} {metrics['flips']:>9} {metrics['problems']:>9}"
        )
        if metrics["unsafe"] or metrics["flips"] or metrics["problems"]:
            failed = True
            if metrics["unsafe_ids"]:
                print(f"  unsafe: {', '.join(metrics['unsafe_ids'])}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
