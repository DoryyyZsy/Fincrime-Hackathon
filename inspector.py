#!/usr/bin/env python3
"""One command for inspectors to test this project. Python 3 standard library only.

    python inspector.py                      verify everything on data/screening/dev
    python inspector.py verify --data DIR    same, on another data folder (dev or unseen format)
    python inspector.py run DIR OUT          run the submitted system (system/run)
    python inspector.py score OUT --data DIR validate an output file and score the practice labels
    python inspector.py explain H-XXXXXX     show one hit: decision, reason and the records it cites
    python inspector.py check "Full Name" [--dob 1968-03-05 --country Spain ...]
                                             screen one customer in the terminal, evidence by evidence
    python inspector.py tests                run the screening test suite
    python inspector.py demo                 start the local demo pages (staff / management / review)

`verify` runs the system with SCREENING_LLM=0 (rules only, deterministic) unless --with-llm is given.
"""
import argparse
import filecmp
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join("data", "screening", "dev")
VALIDATE = os.path.join(ROOT, "data", "screening", "validate.py")
COMMITTED = os.path.join(ROOT, "output", "dev.jsonl")
WORK = os.path.join(ROOT, "artifacts", "inspector")
THIRD_PARTY = ("numpy", "torch", "pandas", "yaml", "requests", "sklearn")
SCREENING_TESTS = ["tests/test_screening_names.py", "tests/test_screening_rewording.py", "tests/test_screening_output.py",
                   "tests/test_screening_extract.py", "tests/test_llm_extract.py", "tests/test_extract_ages.py"]

USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
if USE_COLOR and os.name == "nt":
    os.system("")  # enable ANSI colours in the Windows console
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def paint(text, code):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


OK, BAD, WARN, INFO = paint("PASS", "32;1"), paint("FAIL", "31;1"), paint("WARN", "33;1"), paint("INFO", "36;1")


def section(title):
    print("\n" + paint(f"== {title}", "1"))


def run_system(data, out, with_llm=False, hashseed=None):
    env = dict(os.environ)
    if not with_llm:
        env["SCREENING_LLM"] = "0"
    if hashseed is not None:
        env["PYTHONHASHSEED"] = str(hashseed)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "system", "run"), data, out],
                          capture_output=True, text=True, env=env, cwd=ROOT)
    return proc, time.time() - t0


def same_rows(a, b):
    """Same JSON lines in the same order. Line endings are ignored (Git on Windows may check out CRLF)."""
    with open(a, encoding="utf-8") as fa, open(b, encoding="utf-8") as fb:
        return [l.rstrip("\r\n") for l in fa if l.strip()] == [l.rstrip("\r\n") for l in fb if l.strip()]


def find_validator(data):
    for path in (VALIDATE, os.path.join(data, "..", "validate.py"), os.path.join(data, "validate.py")):
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


def score(out, data):
    """Run the organisers' validate.py (with --score when the folder has practice labels)."""
    validator = find_validator(data)
    if not validator:
        return None, "validate.py not found (expected data/screening/validate.py)"
    cmd = [sys.executable, validator, "--data", data, "--output", out]
    labels = os.path.join(data, "practice_labels.jsonl")
    if os.path.exists(labels):
        cmd += ["--score", labels]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return proc, (proc.stdout + proc.stderr).strip()


def stdlib_only():
    """Import the submitted engine in a clean interpreter and list any third-party modules it pulls in."""
    code = ("import sys; sys.path.insert(0, %r); "
            "import screening_ml.data, screening_ml.pipeline, screening_ml.llm_extract; "
            "print(','.join(sorted(m for m in sys.modules if m.split('.')[0] in %r)))") % (ROOT, THIRD_PARTY)
    proc = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    return proc.returncode == 0, proc.stdout.strip() or proc.stderr.strip()


def run_tests():
    have_pytest = subprocess.run([sys.executable, "-c", "import pytest"], capture_output=True).returncode == 0
    files = [f for f in SCREENING_TESTS if os.path.exists(os.path.join(ROOT, f))]
    if have_pytest:
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *files]
    else:  # the stdlib unittest files still run without pytest
        mods = [f[:-3].replace("/", ".") for f in files if f.endswith(("extract.py", "llm_extract.py"))]
        cmd = [sys.executable, "-m", "unittest", *mods]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    lines = (proc.stdout + proc.stderr).strip().splitlines()
    tally = [l.strip(" =") for l in lines if any(w in l for w in (" passed", " failed", "Ran ", "OK", "FAILED"))
             and "warn" not in l.lower()]
    last = tally[-1] if tally else (lines[-1] if lines else "")
    return proc.returncode == 0, last, ("pytest" if have_pytest else "unittest (pytest not installed)")


# ------------------------------------------------------------------ commands
def cmd_verify(args):
    data = args.data
    results = []

    def record(name, ok, detail, level=None):
        tag = level or (OK if ok else BAD)
        results.append((name, ok if level is None else True, tag))
        print(f"  {tag}  {name}: {detail}")

    section("1. Environment")
    v = sys.version_info
    record("Python 3.8+", v >= (3, 8), f"{sys.executable} ({v.major}.{v.minor}.{v.micro})")
    need = ["alerts.jsonl", "hits.jsonl", "customers.jsonl", "accounts.jsonl", "onboarding.jsonl",
            "kyc_checks.jsonl", "notes.jsonl", "search_results.jsonl"]
    missing = [f for f in need if not os.path.exists(os.path.join(data, f))]
    record("data folder", not missing, data if not missing else f"{data} is missing {', '.join(missing)}")
    if missing:
        return summary(results)
    record("system/run present", os.path.exists(os.path.join(ROOT, "system", "run")), "system/run")
    key = bool(os.environ.get("OPENROUTER_API_KEY"))
    record("model key", True, ("OPENROUTER_API_KEY set; " if key else "no OPENROUTER_API_KEY; ")
           + ("using it (--with-llm)" if args.with_llm and key else "rules only (SCREENING_LLM=0)"), INFO)

    section("2. Run the submitted system")
    out1 = os.path.join(WORK, "run1.jsonl")
    proc, secs = run_system(data, out1, args.with_llm, hashseed=0)
    ok = proc.returncode == 0 and os.path.exists(out1)
    lines = sum(1 for _ in open(out1, encoding="utf-8")) if ok else 0
    record("system/run", ok, f"{lines} lines in {secs:.1f}s (limit 30 min) → {os.path.relpath(out1, ROOT)}"
           + ("" if ok else f"\n{proc.stderr.strip()[-600:]}"))
    if not ok:
        return summary(results)
    print("        " + proc.stderr.strip().splitlines()[-1])

    section("3. Validate and score (organisers' validate.py)")
    vproc, text = score(out1, data)
    if vproc is None:
        record("validate.py", False, text)
    else:
        for line in text.splitlines():
            print("        " + line)
        problems_ok = "0 problems" in text
        record("output format", problems_ok, "0 problems" if problems_ok else "see problems above")
        if "practice labels:" in text:
            unsafe = [l for l in text.splitlines() if "unsafe clearances" in l]
            record("no unsafe clearances", bool(unsafe) and " 0 unsafe" in unsafe[0], unsafe[0].strip() if unsafe else "")

    section("4. Reproducibility")
    out2 = os.path.join(WORK, "run2.jsonl")
    proc2, _ = run_system(data, out2, args.with_llm, hashseed=12345)
    same = proc2.returncode == 0 and filecmp.cmp(out1, out2, shallow=False)
    record("two runs identical", same, "byte-identical with PYTHONHASHSEED=0 and 12345" if same else "outputs differ")
    if os.path.abspath(data) == os.path.abspath(os.path.join(ROOT, DEFAULT_DATA)) and os.path.exists(COMMITTED):
        match = same_rows(out1, COMMITTED)
        record("matches output/dev.jsonl", match, "the committed predictions came from this system"
               if match else "output/dev.jsonl differs from a fresh run (regenerate it)")

    section("5. Rules of the brief")
    ok, mods = stdlib_only()
    record("standard library only", ok and not mods, "no third-party modules imported" if ok and not mods else mods)

    if not args.skip_tests:
        section("6. Tests")
        ok, last, runner = run_tests()
        record("test suite", ok, f"{last} ({runner})")
    return summary(results)


def summary(results):
    failed = [n for n, ok, _ in results if not ok]
    section("Summary")
    print(f"  {len(results) - len(failed)}/{len(results)} checks passed" + (f"; failed: {', '.join(failed)}" if failed else ""))
    print("  " + (paint("ALL CHECKS PASSED", "32;1") if not failed else paint("SOME CHECKS FAILED", "31;1")))
    return 1 if failed else 0


def cmd_run(args):
    proc, secs = run_system(args.data, args.out, args.with_llm)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode


def cmd_score(args):
    proc, text = score(args.out, args.data)
    print(text)
    return 1 if proc is None else proc.returncode


def cmd_explain(args):
    out = args.out
    if not os.path.exists(out):
        print(f"{out} not found; run: python inspector.py run {args.data} {out}")
        return 1
    rows = {r["hit_id"]: r for r in map(json.loads, open(out, encoding="utf-8"))}
    row = rows.get(args.hit_id)
    if not row:
        print(f"{args.hit_id} is not in {out}")
        return 1
    sys.path.insert(0, ROOT)
    from screening_ml.data import read_jsonl
    hit = next((h for h in read_jsonl(os.path.join(args.data, "hits.jsonl")) if h["hit_id"] == args.hit_id), {})
    alert = next((a for a in read_jsonl(os.path.join(args.data, "alerts.jsonl")) if a["alert_id"] == row["alert_id"]), {})
    colour = {"DIFFERENT_ENTITY": "32;1", "SAME_ENTITY": "31;1", "REVIEW": "33;1"}[row["decision"]]
    print(paint(f"{row['hit_id']}  {row['decision']}  ({row['reason_code']})", colour))
    c = alert.get("customer", {})
    print(f"Screened customer: {c.get('name')} · {c.get('entity_type')} · DOB {c.get('dob') or '-'} · "
          f"{c.get('country') or '-'} · link {alert.get('account_ref') or alert.get('login_handle') or alert.get('email') or 'none'}")
    print(f"Watchlist hit:     {hit.get('name')} · {hit.get('primary_category') or ', '.join(hit.get('categories') or [])} · "
          f"{', '.join(hit.get('countries') or []) or '-'} · DOB {', '.join(hit.get('dobs') or []) or '-'}")
    print(f"Customer record:   {row['customer_record_id']}")
    print(f"\nReason:\n  {row['reason']}\n\nEvidence:")
    wanted = set(row["evidence"])
    found = {}
    for name, key in (("customers", "record_id"), ("kyc_checks", "check_id"), ("notes", "note_id"),
                      ("search_results", "result_id"), ("accounts", "account_id")):
        if not wanted - set(found):
            break
        for rec in read_jsonl(os.path.join(args.data, f"{name}.jsonl")):
            if rec[key] in wanted and rec[key] not in found:
                found[rec[key]] = rec
    for rid in row["evidence"]:
        rec = found.get(rid)
        text = rec.get("text") or rec.get("snippet") if rec else None
        if rec and text:
            print(f"  {rid}: {text}")
        elif rec:
            print(f"  {rid}: " + ", ".join(f"{k}={v}" for k, v in rec.items() if v not in (None, "") and k != "record_id")[:300])
        else:
            print(f"  {rid}")
    labels = os.path.join(args.data, "practice_labels.jsonl")
    if os.path.exists(labels):
        gold = next((l for l in read_jsonl(labels) if l["hit_id"] == args.hit_id), None)
        if gold:
            verdict = "matches" if gold["decision"] == row["decision"] else "DOES NOT match"
            print(f"\nPractice label: {gold['decision']} {gold['reason_codes']} ({verdict})")
    return 0


def cmd_check(args):
    sys.path.insert(0, ROOT)
    from screening_ml.data import DataRoom
    from screening_ml.demo_server import Engine

    class TerminalEngine(Engine):  # the demo engine without building the web pages
        def __init__(self, folder):
            self.room = DataRoom(folder)
            self.profiles = self._unique_profiles()

    t0 = time.time()
    print(paint(f"Loading {args.data} …", "2"))
    engine = TerminalEngine(args.data)
    form = {"name": args.name, "entity_type": args.type, "dob": args.dob, "country": args.country, "city": args.city,
            "account_ref": args.account, "login_handle": args.login, "email": args.email}
    res = engine.check(form)
    if res.get("error"):
        print(res["error"])
        return 1
    icon = {"match": paint("✓", "31;1"), "mismatch": paint("✗", "32;1"), "warn": paint("!", "33;1"),
            "unknown": paint("?", "2"), "info": paint("i", "36;1")}
    stamp = {"SAME_ENTITY": paint("ALERT ", "31;1"), "REVIEW": paint("REVIEW", "33;1"), "DIFFERENT_ENTITY": paint("CLEAR ", "32;1")}
    cust = res["customer"]
    print(f"\nCustomer: {cust.get('record') or cust.get('text')}"
          + (f" · legal name {cust['name']}" if cust.get("name") else "")
          + (f" · born {cust['dob']}" if cust.get("dob") else "")
          + (f" · {', '.join(cust['countries'])}" if cust.get("countries") else ""))
    print(f"{len(res['hits'])} watchlist profile(s) match the name, out of {res['profiles_screened']} screened\n")
    for h in res["hits"]:
        w = h["watchlist"]
        print(f"{stamp[h['decision']]}  {w['name']}  ({w['category']}{', ' + ', '.join(w['countries']) if w['countries'] else ''})")
        for ev in h["evidence"]:
            print(f"    {icon[ev['status']]} {ev['label']}" + (f"  {paint(ev['detail'], '2')}" if ev["detail"] else ""))
        if args.why:
            print(f"    {paint(h['reason'], '2')}")
        print()
    overall = {"ALERT": paint("ALERT: possible match, escalate to compliance", "31;1"),
               "REVIEW": paint("REVIEW: an analyst should look at it", "33;1"),
               "CLEAR": paint("CLEAR: the hits are different people", "32;1")}[res["overall"]]
    print(f"Overall: {overall}   ({time.time() - t0:.1f}s)")
    print(paint("  ✓ points to the same person  ✗ points to a different person  ! caution  ? not enough information", "2"))
    return 0


def cmd_tests(args):
    ok, last, runner = run_tests()
    print(f"{OK if ok else BAD}  {last} ({runner})")
    return 0 if ok else 1


def cmd_demo(args):
    print(f"Starting the demo on http://127.0.0.1:{args.port}/ (screening review; also /staff.html and "
          f"/management.html). Ctrl+C to stop; loading takes ~15s")
    return subprocess.call([sys.executable, "-m", "screening_ml.demo_server", "--data", args.data, "--port", str(args.port)], cwd=ROOT)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="inspector.py", description="Test the Track 2 screening project.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 1)[1])
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("verify", help="run, validate, score, reproduce, stdlib check, tests")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--with-llm", action="store_true", help="let system/run use OPENROUTER_API_KEY if set")
    p.add_argument("--skip-tests", action="store_true")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("run", help="run system/run")
    p.add_argument("data")
    p.add_argument("out")
    p.add_argument("--with-llm", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("score", help="validate an output file and score the practice labels")
    p.add_argument("out")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("explain", help="show one hit's decision, reason and cited records")
    p.add_argument("hit_id")
    p.add_argument("--out", default=os.path.join("output", "dev.jsonl"))
    p.add_argument("--data", default=DEFAULT_DATA)
    p.set_defaults(fn=cmd_explain)

    p = sub.add_parser("check", help="screen one customer in the terminal")
    p.add_argument("name")
    p.add_argument("--dob")
    p.add_argument("--country")
    p.add_argument("--city")
    p.add_argument("--type", choices=["Individual", "Company"], default="Individual")
    p.add_argument("--account")
    p.add_argument("--login")
    p.add_argument("--email")
    p.add_argument("--why", action="store_true", help="also print the full reason for each hit")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("tests", help="run the screening test suite")
    p.set_defaults(fn=cmd_tests)

    p = sub.add_parser("demo", help="start the local demo pages")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(fn=cmd_demo)

    args = ap.parse_args(argv)
    if not args.cmd:
        args = ap.parse_args(["verify"] + (argv or sys.argv[1:]))
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
