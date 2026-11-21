"""Decision review data: why each hit was decided, who reviews it, and what they should check.

Routing:
  SAME_ENTITY       -> MLRO queue (a confirmed match is escalated straight away)
  REVIEW            -> analyst queue (agree, override, request information or escalate to the MLRO)
  DIFFERENT_ENTITY  -> auto-cleared; a fixed 5% sample (by hit-ID hash, the same hits every run) goes to QA

"What to check" and the reason recorded for every reviewer action are written by fixed templates from the facts the engine used (no model), so it is the same
every run and never states a fact the data does not hold. Standard library only. Used by demo_server.py.
"""
import hashlib
import json
import os
from datetime import datetime, timezone

from . import normalize as N
from .compare import CLEAR_ORDER, CODE_DECISION, decide_case
from .customer import resolve
from .hit import build_hit

QA_SAMPLE_PERCENT = 5
ROUTE = {"SAME_ENTITY": "mlro", "REVIEW": "analyst", "DIFFERENT_ENTITY": "cleared"}
CONFIRM_ORDER = ["NAME_DOB_LOCATION_MATCH", "NAME_DOB_MATCH", "NAME_LOCATION_MATCH"]
ACTIONS = {
    "analyst": {"clear": "Clear: different person", "confirm_escalate": "Confirm match and send to MLRO",
                "escalate": "Escalate to MLRO (unsure)", "request_info": "Request more information"},
    "mlro": {"mlro_confirm": "Confirm match: act on it", "mlro_clear": "Not a match: clear",
             "return": "Return to analyst"},
    "qa": {"qa_ok": "Clearance correct", "qa_reopen": "Clearance wrong: reopen to analyst"},
}
STATUS = {None: "New", "clear": "Cleared by analyst", "confirm_escalate": "With MLRO (confirmed by analyst)",
          "escalate": "With MLRO (escalated)", "request_info": "Waiting for information",
          "mlro_confirm": "Match confirmed by MLRO", "mlro_clear": "Cleared by MLRO",
          "return": "Returned to analyst", "qa_ok": "QA passed", "qa_reopen": "Reopened by QA"}
CHECK_LABEL = {
    "ENTITY_TYPE_MISMATCH": "Person vs company", "DECEASED": "Died before screening",
    "TEMPORAL_IMPOSSIBILITY": "Too young at earliest report", "DOB_MISMATCH": "Date of birth contradicts",
    "LOCATION_MISMATCH": "No country in common", "NAME_MISMATCH_VERIFIED": "Verified name differs",
    "NAME_DOB_LOCATION_MATCH": "Name + DOB + country agree", "NAME_DOB_MATCH": "Name + DOB agree",
    "NAME_LOCATION_MATCH": "Company name + country agree",
}


def in_qa_sample(hit_id, percent=QA_SAMPLE_PERCENT):
    return int(hashlib.sha1(hit_id.encode()).hexdigest()[:8], 16) % 100 < percent


def current_queue(route, qa_sample, status):
    """Where a hit sits now, given the system's route and the last human action."""
    if status in ("escalate", "confirm_escalate"):
        return "mlro"
    if status in ("return", "qa_reopen", "request_info"):
        return "analyst"
    if status in ("clear", "mlro_confirm", "mlro_clear", "qa_ok"):
        return "done"
    if route == "cleared":
        return "qa" if qa_sample else "done"
    return route


def _fmt(d):
    return N.fmt_date(d) if d else None


def _code_of(comp):
    return next((k for k in CLEAR_ORDER + CONFIRM_ORDER if comp.checks[k].fired), "INSUFFICIENT_DATA")


def guidance(case, raw_hit):
    """What is missing and what the reviewer should check, from the facts only."""
    out = []
    if case.teacher_code == "AMBIGUOUS_CUSTOMER_RECORD":
        for c in case.comparisons:
            code = _code_of(c)
            out.append(f"If the customer is {c.person.primary}, this hit would be {CODE_DECISION[code]} ({code}).")
        out.append("Establish which of these people was screened (account, login or email on the alert, onboarding "
                   "records) and decide the hit for that person only.")
        return out
    if not case.comparisons:
        return ["The alert could not be resolved to a customer: find the customer before deciding."]
    comp = case.comparisons[0]
    p, h = comp.person, comp.hit
    sources = (raw_hit.get("source_urls") or [])[:2]
    if case.teacher_decision == "REVIEW":
        if getattr(h, "dob_withheld", False) and not h.dobs:
            out.append("The hit's profile states an age or birth date that could not be read reliably, so it was "
                       "treated as unknown and no clearance was allowed on other grounds: read the profile notes.")
        elif not h.dobs:
            out.append("The hit gives no date of birth (not in its record, the first paragraph of its profile, or a "
                       "search result that matches its listing)" + (f": check the list sources ({'; '.join(sources)})."
                                                                    if sources else "."))
        if not p.dob:
            out.append("No date of birth is known for the customer: request ID or look for an approved identity check.")
        elif not p.dob_verified:
            out.append(f"The customer's date of birth {_fmt(p.dob.value)} is only declared ({', '.join(p.dob.sources)}); "
                       "no approved identity check confirms it.")
        if p.dob and h.dobs and not comp.dob_agree and not comp.dob_contra:
            out.append("The dates of birth neither agree nor contradict (for example the year matches but a month is "
                       "unknown on one side): compare the full dates in the sources.")
        if not comp.name_agree:
            out.append(f"The names only partly agree (closest hit name: {comp.matched_hit_name or h.names[0][0]}): "
                       "compare full names and aliases.")
        if not comp.name_usable:
            out.append("The customer's name has fewer than two real parts, so policy allows neither clearing nor "
                       "confirming on the name.")
        cust_c, hit_c = set(p.countries), set(h.countries)
        if cust_c and hit_c and not (cust_c & hit_c):
            if comp.name_agree and comp.dob_agree:
                out.append(f"Name and date of birth agree but the countries differ ({', '.join(sorted(cust_c))} vs "
                           f"{', '.join(sorted(hit_c))}): policy forbids clearing on location here. Look for any link "
                           "between the customer and the hit's countries.")
            elif getattr(p, "unread_notes", None):
                out.append(f"The countries differ, but notes {', '.join(p.unread_notes[:3])} mention where the customer "
                           "lived and could not be read reliably: read them before clearing on location.")
        if not cust_c:
            out.append("No country is known for the customer.")
        if not hit_c:
            out.append("The hit is tied to no country.")
        if getattr(p, "identity_unreadable", False):
            out.append("An identity note could not be read reliably, so its name, birth date and document country "
                       "were not used: read it.")
        if p.n_rejected_checks:
            out.append(f"{p.n_rejected_checks} rejected or expired identity check(s) were ignored as evidence.")
    elif case.teacher_decision == "SAME_ENTITY":
        if h.dob_source in ("search", "notes"):
            where = "a web search result" if h.dob_source == "search" else "the hit's profile notes"
            out.append(f"The hit's date of birth comes from {where} ({h.dob_text or ', '.join(h.dob_evidence)}), not "
                       "from the list record: confirm it against the list source before acting.")
        if p.dob and not p.dob_verified:
            out.append(f"The customer's date of birth is declared, not verified ({', '.join(p.dob.sources)}).")
        position = (raw_hit.get("position_or_allegation") or "").strip().rstrip(".")
        out.append(f"Listed as: {h.category or 'watchlist entry'}" + (f" - {position}" if position else "")
                   + ". Decide the action (exit, report, enhanced due diligence) under the bank's policy.")
    else:
        fired = [k for k in CLEAR_ORDER if comp.checks[k].fired]
        out.append("Cleared on: " + ", ".join(CHECK_LABEL[k] for k in fired) + ". QA: confirm the cited records ("
                   + (", ".join(case.teacher_evidence) or "none") + ") belong to this customer and are current.")
    return out


def _lower_first(text):
    return text[:1].lower() + text[1:] if text else text


def action_reason(d, role, action):
    """The audit reason for a reviewer action, written from the hit's facts and POLICY.md (never typed).

    When the action goes against what the data supports, the reason says so and names the policy condition the
    data does not meet, so an override is visible in the audit log."""
    hid, dec, code, why = d["hit_id"], d["decision"], d["code"], d["reason"]
    held = {c["label"] for c in d["checks"] if c["holds"]}
    contradictions = [CHECK_LABEL[k] for k in CLEAR_ORDER if CHECK_LABEL[k] in held]
    agreements = [CHECK_LABEL[k] for k in CONFIRM_ORDER if CHECK_LABEL[k] in held]
    if code == "AMBIGUOUS_CUSTOMER_RECORD":
        # the checks shown are for one of several people; they settle nothing until the customer is known
        contradictions, agreements = [], []
    todo = d["guidance"]
    missing = " ".join(todo[:2]) if todo else "No further gap is recorded in the files."
    listing = (d.get("hit") or {}).get("category") or "watchlist entry"

    def cleared():
        if dec == "DIFFERENT_ENTITY" or contradictions:
            return (f"{hid} cleared as a different {'company' if (d.get('hit') or {}).get('entity_type') == 'Company' else 'person'} "
                    f"under POLICY principle 6 (clear only on a contradiction): "
                    f"{', '.join(contradictions) or code}. {why}")
        return (f"{hid} cleared by reviewer OVERRIDE. The files show no contradiction (POLICY principle 6 requires one: "
                f"person vs company, death before screening, too young, date of birth, location or verified name), "
                f"so this clearance rests on information outside the files. System decision was {dec} ({code}).")

    def confirmed():
        if dec == "SAME_ENTITY" or agreements:
            return (f"{hid} confirmed as the customer under POLICY principle 7 (confirm only on agreement): "
                    f"{', '.join(agreements) or code}. {why}")
        if contradictions:
            return (f"{hid} confirmed by reviewer OVERRIDE although the files contradict it under POLICY principle 6 "
                    f"({', '.join(contradictions)}): {why} System decision was {dec} ({code}).")
        return (f"{hid} confirmed by reviewer OVERRIDE. POLICY principle 7 is not met in the files (name and date of "
                f"birth, or for a company name and country, do not both agree): {_lower_first(missing)} "
                f"System decision was {dec} ({code}).")

    def second_view(verb, who):
        if dec == "REVIEW":
            return (f"{hid} {verb} {who}: the evidence cannot settle it (POLICY: REVIEW when the evidence is "
                    f"insufficient or contradictory; {code}). {missing}")
        return (f"{hid} {verb} {who} for a second view of the system decision {dec} ({code}), which the files support: "
                f"{why}")

    if role == "analyst":
        if action == "clear":
            return cleared()
        if action == "confirm_escalate":
            return confirmed() + " Sent to the MLRO for action."
        if action == "escalate":
            return second_view("escalated", "to the MLRO")
        if action == "request_info":
            return f"More information requested for {hid} before a decision ({code}). Needed: {' '.join(todo) or missing}"
    if role == "mlro":
        if action == "mlro_confirm":
            return confirmed() + f" Listed as {listing}: the MLRO takes the action the bank's policy requires."
        if action == "mlro_clear":
            return cleared() + " Decided by the MLRO."
        if action == "return":
            return second_view("returned", "to the analyst by the MLRO")
    if role == "qa":
        if action == "qa_ok":
            if dec == "DIFFERENT_ENTITY":
                return (f"QA: the clearance of {hid} meets POLICY principle 6 ({', '.join(contradictions) or code}); "
                        f"the cited records ({', '.join(d['evidence_ids']) or 'none'}) belong to the resolved customer.")
            return f"QA: {hid} is not a clearance ({dec}); nothing to pass."
        if action == "qa_reopen":
            if dec != "DIFFERENT_ENTITY":
                return f"QA: {hid} is not a clearance ({dec}); nothing to reopen."
            return (f"QA reopened the clearance of {hid} for an analyst: the clearance rests on "
                    f"{', '.join(contradictions) or code} from {', '.join(d['evidence_ids']) or 'no cited record'}, "
                    f"which QA could not confirm against the files.")
    return f"{hid}: {action} by {role}."


def _person(p):
    if p is None:
        return None
    vn = p.verified_name
    return {
        "records": p.records, "primary": p.primary, "entity_type": p.entity_type,
        "names": sorted({" ".join(f.value).title() for f in p.names}),
        "verified_name": ({"value": getattr(vn, "raw", None) or " ".join(vn.value).title(), "source": vn.sources}
                          if vn else None),
        "dob": ({"value": _fmt(p.dob.value), "source": p.dob.sources, "verified": p.dob_verified} if p.dob else None),
        "countries": {c: sorted(set(s)) for c, s in sorted(p.countries.items())},
        "approved_checks": p.n_approved_checks, "rejected_checks": p.n_rejected_checks,
    }


def _hit(h, raw):
    return {
        "name": raw.get("name"), "names": [n for n, _ in h.names][:6], "entity_type": h.entity_type,
        "category": h.category or "unknown", "dobs": [_fmt(d) for d in h.dobs], "dob_source": h.dob_source,
        "dob_text": h.dob_text, "countries": sorted(h.countries), "earliest_report": _fmt(h.earliest_report),
        "deceased": _fmt(h.deceased), "position": (raw.get("position_or_allegation") or "")[:200],
        "source_urls": (raw.get("source_urls") or [])[:5], "match_score": raw.get("match_score"),
    }


class ReviewBook:
    """Every hit of a data folder decided once at start-up, plus the append-only audit log."""

    def __init__(self, room, present, audit_path):
        self.room, self.audit_path = room, audit_path
        self._records = {}
        for k in (x for xs in room.checks_by_applicant.values() for x in xs):
            self._records[k["check_id"]] = k
        for n in (x for xs in room.notes_by_email.values() for x in xs):
            self._records[n["note_id"]] = n
        for s in (x for xs in room.search_by_query.values() for x in xs):
            self._records[s["result_id"]] = s
        self.items, self.detail = [], {}
        by_alert = {}
        for alert in room.alerts:
            res = resolve(room, alert)
            for hid in alert["hit_ids"]:
                raw = room.hit_by_id[hid]
                case = decide_case(res, build_hit(room, raw), alert)
                comp = case.comparisons[0] if case.comparisons else None
                route = ROUTE[case.teacher_decision]
                item = {"hit_id": hid, "alert_id": alert["alert_id"], "decision": case.teacher_decision,
                        "code": case.teacher_code, "route": route, "qa_sample": route == "cleared" and in_qa_sample(hid),
                        "screened": (alert.get("customer") or {}).get("name"), "listed": raw.get("name"),
                        "category": comp.hit.category if comp else ""}
                self.items.append(item)
                by_alert.setdefault(alert["alert_id"], []).append(case.teacher_decision)
                self.detail[hid] = {
                    **item, "reason": case.teacher_reason, "evidence_ids": case.teacher_evidence,
                    "customer_record_id": case.customer_record_id, "link_type": res.link_type,
                    "screened_details": alert.get("customer") or {}, "screened_at": alert.get("screened_at"),
                    "chips": present(case, raw)["evidence"] if comp else [],
                    "guidance": guidance(case, raw),
                    "customer": _person(comp.person) if comp and res.persons else None,
                    "other_people": [_person(c.person) for c in case.comparisons[1:]],
                    "hit": _hit(comp.hit, raw) if comp else None,
                    "checks": [{"label": CHECK_LABEL[k], "holds": comp.checks[k].fired} for k in CHECK_LABEL] if comp else [],
                    "records": [{"id": i, "record": self._record(i)} for i in case.teacher_evidence],
                }
        for item in self.items:   # an alert closes only when every hit in it is cleared
            item["alert_open"] = any(d != "DIFFERENT_ENTITY" for d in by_alert[item["alert_id"]])
            self.detail[item["hit_id"]]["alert_open"] = item["alert_open"]

    def _record(self, rid):
        if rid.startswith("CUS-"):
            return self.room.customer_by_id.get(rid)
        return self._records.get(rid)

    def audit(self):
        if not os.path.exists(self.audit_path):
            return []
        with open(self.audit_path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.endswith("\n")]

    def queue(self):
        last = {}
        for row in self.audit():
            last[row["hit_id"]] = row["action"]
        out = []
        for item in self.items:
            status = last.get(item["hit_id"])
            out.append({**item, "status": STATUS.get(status, status), "queue": current_queue(item["route"], item["qa_sample"], status)})
        return out

    def hit(self, hit_id):
        d = self.detail.get(hit_id)
        if d is None:
            return None
        trail = [r for r in self.audit() if r["hit_id"] == hit_id]
        status = trail[-1]["action"] if trail else None
        reasons = {role: {a: action_reason(d, role, a) for a in acts} for role, acts in ACTIONS.items()}
        return {**d, "trail": trail, "status": STATUS.get(status, status),
                "queue": current_queue(d["route"], d["qa_sample"], status), "actions": ACTIONS, "action_reasons": reasons}

    def act(self, form):
        hit_id, role, action = form.get("hit_id"), form.get("role"), form.get("action")
        reviewer, note = (form.get("reviewer") or "").strip()[:80], (form.get("note") or "").strip()[:1000]
        if hit_id not in self.detail or role not in ACTIONS or action not in ACTIONS[role]:
            return {"error": "Unknown hit, role or action."}
        if not reviewer:
            return {"error": "Enter your name: the audit log records who acted."}
        d = self.detail[hit_id]
        # the reason is generated here from the facts and POLICY.md; any reason text sent by the page is ignored
        reason = action_reason(d, role, action)
        entry = {"hit_id": hit_id, "alert_id": d["alert_id"], "role": role, "reviewer": reviewer, "action": action,
                 "system_decision": d["decision"], "system_code": d["code"], "reason": reason,
                 "override": "OVERRIDE" in reason, "note": note or None,
                 "timestamp": datetime.now(timezone.utc).isoformat()}
        os.makedirs(os.path.dirname(self.audit_path) or ".", exist_ok=True)
        fd = os.open(self.audit_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8"))
        finally:
            os.close(fd)
        return {"ok": True, "status": STATUS[action], "reason": reason}
