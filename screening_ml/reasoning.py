"""Write the `reason` field: two to four sentences naming the decisive facts, honest about what is missing.

Every reason has the same shape:
  1. who the customer is and how the alert led to them (or that no record fits, or that several people do);
  2. the decisive facts, with the records they came from (the check's own sentence from compare.py);
  3. what that means: what else agreed, or what could not be compared.
Deterministic templates only, so the same input always gives the same words.
"""
import re

from . import normalize as N

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")
LABEL = {"DIFFERENT_ENTITY": "a different person", "SAME_ENTITY": "the customer", "REVIEW": "unresolved"}


def sentences(text):
    return [s.strip() for s in _SENT.split((text or "").strip()) if s.strip()]


def _who(resolution, alert):
    """Sentence 1: how the alert led to the customer."""
    persons = resolution.persons
    if not persons:
        bits = [b for b in (resolution.screened.get("dob") and f"born {resolution.screened['dob']}",
                            resolution.screened.get("country")) if b]
        return (f"No customer record fits the screened details ({resolution.screened.get('name')}"
                + (", " + ", ".join(bits) if bits else "") + "), so the hit is compared with what was screened.")
    p = persons[0]
    if len(persons) > 1:
        ids = [q.primary for q in persons]
        return (f"{len(persons)} different people in the system of record fit the alert ({', '.join(ids[:4])}"
                f"{' and others' if len(ids) > 4 else ''}), and each of them leads to the same decision.")
    if resolution.link_type == "account":
        how = f"account {alert.get('account_ref')}"
        if len(resolution.candidates) > 1 or "joint" in how:
            how += ", a joint account,"
    elif resolution.link_type == "login":
        how = f"login {alert.get('login_handle')}"
    elif resolution.link_type == "email":
        how = "the alert's email address"
    else:
        how = "a search on the screened name, date of birth and location"
    sentence = f"The alert resolves to customer {p.primary} through {how}".replace(",,", ",").rstrip(",")
    if p.verified_name or p.dob_verified:
        src = (p.verified_name.sources if p.verified_name else p.dob.sources)[0]
        sentence += f" (identity verified in {src})"
    if len(p.records) > 1:
        others = [r for r in p.records if r != p.primary]
        sentence += (f", and {', '.join(others)} {'is' if len(others) == 1 else 'are'} the same person "
                     f"(shared phone and date of birth, or document number)")
    return sentence + "."


def _meaning(comp, code, decision):
    """Sentence 3: what the decisive facts mean, and what else was or was not known."""
    name = (comp.matched_hit_name or "").rstrip(". ")
    if decision == "DIFFERENT_ENTITY":
        if code == "NAME_MISMATCH_VERIFIED":
            return ("The name on the customer's record or alert may resemble the hit, but the verified legal name "
                    "outranks it under the policy, so the hit is a different person.")
        if comp.name_agree:
            return (f"The names agree ({name}), but this contradiction is enough under the policy "
                    f"to show the hit is a different person who shares the name.")
        return "This contradiction alone shows the hit is a different entity."
    if decision == "SAME_ENTITY":
        if code == "NAME_DOB_MATCH" and not comp.country_overlap:
            return "No country links them, but nothing in the customer's records contradicts the match, so the alert goes to compliance."
        return "Nothing in the customer's records contradicts the match, so the alert goes to compliance."
    return ""


def compose(resolution, alert, comp, code, decision, check_text, per_person=None):
    """Assemble the final reason (2-4 sentences)."""
    if code == "AMBIGUOUS_CUSTOMER_RECORD":
        ids = [p.primary for p in resolution.persons if p.primary]
        first = (f"{len(ids)} different people in the system of record fit the alert "
                 f"({', '.join(ids[:5])}{' and others' if len(ids) > 5 else ''}).")
        detail = ""
        if per_person:
            parts = [f"{pid} would be {LABEL[d]} ({c})" for pid, d, c in per_person[:3]]
            detail = "Taken one at a time, " + "; ".join(parts) + "."
        return " ".join(s for s in (first, detail, "Because they lead to different decisions, a person must decide which customer was screened.") if s)

    who = _who(resolution, alert)
    body = sentences(check_text)
    meaning = _meaning(comp, code, decision) if comp is not None else ""
    parts = [who] + body + ([meaning] if meaning else [])
    # keep 2-4 sentences: drop the meaning sentence first, then the who sentence, never the decisive facts
    if len(parts) > 4 and meaning:
        parts.remove(meaning)
    if len(parts) > 4:
        parts.remove(who)
    if len(parts) > 4:
        parts = parts[:4]
    if len(parts) < 2:
        parts.append("No other fact in the files changes this.")
    return " ".join(parts)


def fmt_dates(dates):
    return " or ".join(N.fmt_date(d) for d in dates)
