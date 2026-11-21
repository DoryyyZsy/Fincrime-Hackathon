"""Run the rule engine, optionally after the OpenRouter extractors.

With no client the decisions are exactly the regex engine. A client that fails
open is treated the same way.
"""
import json

from .compare import decide_case
from .customer import resolve
from .hit import build_hit
from .linkage import score_cases
from .llm_extract import reconcile


def adjudicate(room, client=None):
    """One decision row per hit, in hits.jsonl order. Returns (rows, errors, stats_line)."""
    resolutions = {}
    for alert in room.alerts:
        try:
            resolutions[alert["alert_id"]] = resolve(room, alert)
        except Exception:
            resolutions[alert["alert_id"]] = None
    book = None
    stats = None
    if client is not None:
        try:
            book = client.extract(room, resolutions)
            stats = client.stats.line(client.mock)
        except Exception:
            book = None
            stats = "llm extraction failed; using the regex engine"
    rows, cases, raws, errors = [], [], [], 0
    for hit in room.hits:
        alert = room.alert_by_id.get(hit.get("alert_id"))
        raws.append(hit)
        if alert is None:
            rows.append(_fallback(hit, hit.get("alert_id"), "its alert is missing"))
            cases.append(None)
            errors += 1
            continue
        try:
            resolution = resolutions.get(alert["alert_id"])
            if resolution is None:
                resolution = resolve(room, alert)
            regex_hit = build_hit(room, hit)
            regex_case = decide_case(resolution, regex_hit, alert)
            if book is None:
                case = regex_case
            else:
                llm_resolution = resolve(room, alert, book)
                llm_hit = build_hit(room, hit, book)
                llm_case = decide_case(llm_resolution, llm_hit, alert)
                case = reconcile(regex_case, llm_case, llm_hit, llm_resolution)
            rows.append(_row(case))
            cases.append(case)
        except Exception as error:
            rows.append(_fallback(hit, alert["alert_id"], type(error).__name__, resolutions.get(alert["alert_id"])))
            cases.append(None)
            errors += 1
    # The score annotates. It is applied through guarded_decision, which keeps
    # the rule decision, so a failure here must not change what was decided.
    try:
        scores, fit = score_cases(room, cases, raws)
    except Exception:
        scores, fit = [], None
    return rows, errors, stats, scores, fit


def _row(case):
    return {
        "hit_id": case.hit_id, "alert_id": case.alert_id, "decision": case.teacher_decision,
        "reason_code": case.teacher_code, "customer_record_id": case.customer_record_id,
        "reason": case.teacher_reason, "evidence": case.teacher_evidence,
    }


def _fallback(hit, alert_id, why, resolution=None):
    record = None
    if resolution is not None and resolution.persons:
        record = "AMBIGUOUS" if len(resolution.persons) > 1 else resolution.persons[0].primary
    return {
        "hit_id": hit["hit_id"], "alert_id": alert_id, "decision": "REVIEW", "reason_code": "INSUFFICIENT_DATA",
        "customer_record_id": record,
        "reason": f"The system could not evaluate this hit automatically ({why}), so a person must review it.",
        "evidence": [],
    }


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
