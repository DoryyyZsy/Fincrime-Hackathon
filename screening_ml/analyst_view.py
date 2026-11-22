"""Build the analyst screen: one self-contained HTML page with every hit, its decision and the evidence chain.

    python -m screening_ml.analyst_view [--data data/screening/dev] [--out artifacts/analyst_view.html]

Standard library only. The page embeds customer data, so it is written to artifacts/ (git-ignored):
never commit or publish it. Analyst overrides are kept in the browser and exported as JSON Lines.
"""
import argparse
import json
import os
import sys
import time

from . import normalize as N
from .compare import CLEAR_ORDER, CONFIRM_ORDER, decide_case
from .customer import resolve
from .data import DataRoom
from .hit import build_hit
from .linkage import fit_for_pages, score_cases

TEMPLATE = os.path.join(os.path.dirname(__file__), "analyst_view.html")
CHECK_ORDER = CLEAR_ORDER + CONFIRM_ORDER + ["AMBIGUOUS_CUSTOMER_RECORD", "INSUFFICIENT_DATA"]


def _path(room, alert, res):
    """Human-readable trail from the alert to the customer, like the practice labels' `path`."""
    steps = []
    if res.link_type == "account":
        holders = room.holders_by_account.get(alert["account_ref"], [])
        steps.append(f"account_ref {alert['account_ref']} → accounts → "
                     + (f"joint holders {', '.join(holders)}" if len(holders) > 1 else f"customer {', '.join(holders) or 'none'}"))
    elif res.link_type == "login":
        steps.append(f"login_handle {alert['login_handle']} → accounts → customer {', '.join(room.customer_by_login.get(alert['login_handle'].lower(), [])) or 'none'}")
    elif res.link_type == "email":
        steps.append(f"email {alert['email']} → customers")
    else:
        steps.append("no link → search customers by name, date of birth, country and city")
    if res.candidates:
        steps.append(f"screened details fit {', '.join(res.candidates[:6])}" + (" …" if len(res.candidates) > 6 else ""))
    else:
        steps.append("no customer record fits → compare with the screened details")
    for p in res.persons:
        if len(p.records) > 1:
            steps.append(f"same phone and date of birth, or same document number → one person: {', '.join(p.records)}")
    if len(res.persons) > 1:
        steps.append(f"{len(res.persons)} different people fit the alert")
    return steps


def _person(p):
    return {
        "records": p.records, "primary": p.primary, "type": p.entity_type,
        "names": [{"name": " ".join(f.value).title(), "src": f.sources} for f in p.names],
        "verified_name": {"name": " ".join(p.verified_name.value).title(), "src": p.verified_name.sources} if p.verified_name else None,
        "dob": {"value": N.fmt_date(p.dob.value), "src": p.dob.sources, "verified": p.dob_verified} if p.dob else None,
        "countries": [{"country": c, "src": sorted(set(s))} for c, s in sorted(p.countries.items())],
        "unread": p.unread_notes, "approved": p.n_approved_checks, "rejected": p.n_rejected_checks,
    }


def _hit(h, raw):
    return {
        "name": raw.get("name"), "type": h.entity_type, "category": h.category, "phrase": h.phrase,
        "score": raw.get("match_score"), "names": [n for n, _ in h.names],
        "dobs": [N.fmt_date(d) for d in h.dobs], "dob_source": h.dob_source, "dob_text": h.dob_text,
        "dob_evidence": [e for e in h.dob_evidence if "#" not in e],
        "countries": sorted(h.countries), "earliest_report": N.fmt_date(h.earliest_report) if h.earliest_report else None,
        "deceased": N.fmt_date(h.deceased) if h.deceased else None,
        "position": raw.get("position_or_allegation"), "notes": (raw.get("profile_notes") or "")[:1500],
        "urls": (raw.get("source_urls") or [])[:5],
    }


def build(folder):
    room = DataRoom(folder)
    kyc = {k["check_id"]: k for k in room.kyc_checks}
    notes = {n["note_id"]: n for n in room.notes}
    search = {s["result_id"]: s for s in room.search_results}
    gold = {l["hit_id"]: l for l in room.practice_labels}
    records, hits, res_cache = {}, [], {}
    cases, raws = [], []

    def keep(rid):
        if rid in records or not rid:
            return
        if rid in room.customer_by_id:
            records[rid] = room.customer_by_id[rid]
        elif rid in kyc:
            records[rid] = kyc[rid]
        elif rid in notes:
            records[rid] = notes[rid]
        elif rid in search:
            records[rid] = search[rid]

    for raw in room.hits:
        alert = room.alert_by_id[raw["alert_id"]]
        if alert["alert_id"] not in res_cache:
            res_cache[alert["alert_id"]] = resolve(room, alert)
        res = res_cache[alert["alert_id"]]
        hf = build_hit(room, raw)
        case = decide_case(res, hf, alert)
        comps = []
        for comp in case.comparisons:
            # INSUFFICIENT_DATA is the fallback that always exists: show it as fired only when it is the outcome
            checks = [{"code": c, "fired": comp.checks[c].fired and (c != "INSUFFICIENT_DATA" or case.teacher_code == c),
                       "text": comp.checks[c].text,
                       "evidence": [e for e in comp.checks[c].evidence if "#" not in e]}
                      for c in CHECK_ORDER if c in comp.checks]
            comps.append({"person": _person(comp.person), "checks": checks})
            for p_rid in comp.person.records:
                keep(p_rid)
                c = room.customer_by_id[p_rid]
                for app in room.applicants_by_customer.get(p_rid, []):
                    for k in room.checks_by_applicant.get(app, []):
                        keep(k["check_id"])
                for n in room.notes_by_email.get((c.get("email") or "").lower(), []):
                    if n["channel"] == "kyc" or "resid" in n["text"].lower() or n["note_id"] in comp.person.unread_notes:
                        keep(n["note_id"])
        for e in case.teacher_evidence + hf.dob_evidence:
            keep(e)
        g = gold.get(raw["hit_id"])
        cases.append(case)
        raws.append(raw)
        hits.append({
            "hit_id": raw["hit_id"], "alert_id": alert["alert_id"], "screened_at": alert.get("screened_at"),
            "screened": alert["customer"], "link": res.link_type, "path": _path(room, alert, res),
            "decision": case.teacher_decision, "code": case.teacher_code, "reason": case.teacher_reason,
            "evidence": case.teacher_evidence, "record": case.customer_record_id,
            "hit": _hit(hf, raw), "comparisons": comps, "n_hits_in_alert": len(alert["hit_ids"]),
            "gold": {"decision": g["decision"], "codes": g["reason_codes"], "reason": g.get("reference_reason")} if g else None,
        })
    score_meta = None
    try:
        scores, fit = score_cases(room, cases, raws)
    except Exception:
        scores, fit = [], None
    if scores and fit is not None and len(scores) == len(hits):
        score_meta = fit_for_pages(fit)
        for hit, score in zip(hits, scores):
            hit["match_probability"] = score["match_probability"]
            hit["match_weight"] = score["match_weight"]
            hit["review_rank"] = score["review_rank"]
            hit["match_levels"] = score["levels"]
            hit["field_weights"] = score["field_weights"]
    return {"hits": hits, "records": records, "score": score_meta,
            "folder": os.path.abspath(folder), "built": time.strftime("%Y-%m-%d %H:%M")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/screening/dev")
    ap.add_argument("--out", default="artifacts/analyst_view.html")
    args = ap.parse_args(argv)
    t0 = time.time()
    payload = build(args.data)
    with open(TEMPLATE, encoding="utf-8") as handle:
        html = handle.read()
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(html.replace("/*__DATA__*/null", data))
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB, {len(payload['hits'])} hits) in {time.time() - t0:.1f}s")
    print("contains customer data: keep it out of git and do not publish it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
