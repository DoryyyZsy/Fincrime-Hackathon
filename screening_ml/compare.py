"""Compare one person with one hit: every SOP check, its evidence, its reason sentence, and NN features.

`teacher_decision` applies POLICY.md principles 6-7 exactly as written. It produces the silver labels the
network is trained on, because only 50 gold labels exist (see check_data.py).
"""
from dataclasses import dataclass, field

from . import normalize as N
from . import reasoning
from .customer import screened_person

DECISIONS = ["DIFFERENT_ENTITY", "SAME_ENTITY", "REVIEW"]
REASON_CODES = [
    "TEMPORAL_IMPOSSIBILITY", "DOB_MISMATCH", "LOCATION_MISMATCH", "NAME_MISMATCH_VERIFIED",
    "ENTITY_TYPE_MISMATCH", "DECEASED", "NAME_DOB_LOCATION_MATCH", "NAME_DOB_MATCH", "NAME_LOCATION_MATCH",
    "INSUFFICIENT_DATA", "AMBIGUOUS_CUSTOMER_RECORD",
]
CODE_DECISION = {
    "TEMPORAL_IMPOSSIBILITY": "DIFFERENT_ENTITY", "DOB_MISMATCH": "DIFFERENT_ENTITY",
    "LOCATION_MISMATCH": "DIFFERENT_ENTITY", "NAME_MISMATCH_VERIFIED": "DIFFERENT_ENTITY",
    "ENTITY_TYPE_MISMATCH": "DIFFERENT_ENTITY", "DECEASED": "DIFFERENT_ENTITY",
    "NAME_DOB_LOCATION_MATCH": "SAME_ENTITY", "NAME_DOB_MATCH": "SAME_ENTITY", "NAME_LOCATION_MATCH": "SAME_ENTITY",
    "INSUFFICIENT_DATA": "REVIEW", "AMBIGUOUS_CUSTOMER_RECORD": "REVIEW",
}
MIN_AGE_AT_REPORT = 16


@dataclass
class Check:
    fired: bool
    evidence: list = field(default_factory=list)
    text: str = ""


@dataclass
class Comparison:
    person: object
    hit: object
    checks: dict                       # reason code -> Check
    name_agree: bool
    name_frac: float
    name_matched: int
    name_sim: float
    name_usable: bool
    matched_hit_name: str
    dob_agree: bool
    dob_contra: bool
    year_gap: float
    country_overlap: set
    age_at_report: float


# ------------------------------------------------------------------ date logic (principles 6 and 7)
def dob_contradicted(cust, hit_date):
    cy, cm, cd = cust
    hy, hm, hd = hit_date
    if abs(cy - hy) > 1:
        return True
    if cy == hy and cm and hm and cm != hm:
        return True
    if cy == hy and cm and hm and cm == hm and cd and hd and cd != hd:
        return True
    return False


def dob_agrees(cust, hit_date):
    return all(a is None or b is None or a == b for a, b in zip(cust, hit_date))


def _age_at(birth, when):
    """Largest age the customer could have had at `when` (unknown parts give the benefit of the doubt)."""
    by, bm, bd = birth
    wy, wm, wd = when
    age = wy - by
    if bm and wm and (wm, wd or 31) < (bm, bd or 1):
        age -= 1
    return age


def _before(a, b):
    """True only if date a is certainly before date b."""
    a_late = (a[0], a[1] or 12, a[2] or 31)
    b_early = (b[0], b[1] or 1, b[2] or 1)
    return a_late < b_early


# ------------------------------------------------------------------ comparison
def compare(person, hit, screened_at):
    person_company = person.entity_type == "Company"
    hit_company = hit.entity_type == "Company"
    checks = {code: Check(False) for code in REASON_CODES}
    cid = [person.primary] if person.primary else []

    # names (principle 8)
    best = (False, 0, 0.0, "", 0.0)
    for _, cparts in [(None, f.value) for f in person.names] + (
            [(None, person.verified_name.value)] if person.verified_name else []):
        for display, hparts in hit.names:
            agree, matched, frac = N.name_agreement(cparts, hparts)
            sim = N.best_similarity(cparts, hparts)
            if (agree, matched, frac, sim) > (best[0], best[1], best[2], best[4]):
                best = (agree, matched, frac, display, sim)
    name_agree, name_matched, name_frac, matched_name, name_sim = best
    real_parts = max([len(f.value) for f in person.names] + [len(person.verified_name.value) if person.verified_name else 0] + [0])
    name_usable = person_company or real_parts >= 2

    # entity type
    if person.entity_type and hit.entity_type and person_company != hit_company:
        checks["ENTITY_TYPE_MISMATCH"] = Check(True, cid, f"The customer is {'a company' if person_company else 'a person'} but the hit is {'a company' if hit_company else 'a person'}.")

    # deceased before screening
    s_date = N.parse_date(screened_at)
    if hit.deceased and s_date and _before(hit.deceased, s_date):
        checks["DECEASED"] = Check(True, [f"{hit.hit_id}#deceased_date"],
                                   f"The hit's profile gives a date of death of {N.fmt_date(hit.deceased)}, before the customer was screened on {screened_at}.")

    # dates of birth
    cust_dob = person.dob.value if person.dob else None
    dob_src = person.dob.sources if person.dob else []
    dob_agree = dob_contra = False
    year_gap = 0.0
    if cust_dob and hit.dobs and not hit_company:
        dob_agree = any(cust_dob[0] == h[0] and dob_agrees(cust_dob, h) for h in hit.dobs)
        dob_contra = all(dob_contradicted(cust_dob, h) for h in hit.dobs)
        year_gap = min(abs(cust_dob[0] - h[0]) for h in hit.dobs)
        hit_years = " or ".join(N.fmt_date(h) for h in sorted(hit.dobs, key=N.date_key))
        if dob_contra:
            checks["DOB_MISMATCH"] = Check(True, dob_src + [e for e in hit.dob_evidence if not e.endswith("#dobs")],
                                           f"The customer's date of birth is {N.fmt_date(cust_dob)} ({', '.join(dob_src) or 'screened'}) but the hit's {'profile notes put' if hit.dob_source == 'notes' else 'search results put' if hit.dob_source == 'search' else 'record gives'} its date of birth as {hit_years}.")
    elif cust_dob and hit.dobs and hit_company and person_company:  # incorporation dates
        dob_contra = all(dob_contradicted(cust_dob, h) for h in hit.dobs)
        dob_agree = any(dob_agrees(cust_dob, h) and cust_dob[0] == h[0] for h in hit.dobs)
        if dob_contra:
            checks["DOB_MISMATCH"] = Check(True, dob_src, f"The customer's incorporation date {N.fmt_date(cust_dob)} contradicts the hit's {', '.join(N.fmt_date(h) for h in hit.dobs)}.")

    # temporal impossibility
    age_at_report = 99.0
    if cust_dob and hit.earliest_report:
        if person_company:
            if _before(hit.earliest_report, cust_dob):
                checks["TEMPORAL_IMPOSSIBILITY"] = Check(True, dob_src + [f"{hit.hit_id}#profile_notes"],
                                                         f"The company was incorporated on {N.fmt_date(cust_dob)}, after the hit's earliest dated report on {N.fmt_date(hit.earliest_report)}.")
        else:
            age_at_report = float(_age_at(cust_dob, hit.earliest_report))
            if age_at_report < MIN_AGE_AT_REPORT:
                checks["TEMPORAL_IMPOSSIBILITY"] = Check(True, dob_src,
                                                         f"The customer was born in {cust_dob[0]} ({', '.join(dob_src) or 'screened'}), so they were at most {int(age_at_report)} when the hit's profile reports activity on {N.fmt_date(hit.earliest_report)}.")

    # location
    cust_countries = set(person.countries)
    overlap = cust_countries & hit.countries
    if cust_countries and hit.countries and not overlap:
        srcs = sorted({s for c in cust_countries for s in person.countries[c]})
        if not (name_agree and dob_agree):  # name + dob agreeing turns a country difference into REVIEW
            checks["LOCATION_MISMATCH"] = Check(True, srcs,
                                                f"The customer is linked only to {', '.join(sorted(cust_countries))} but the hit is linked only to {', '.join(sorted(hit.countries))}.")

    # verified legal name agrees with none of the hit's names
    if person.verified_name and name_usable and hit.names:
        vparts = person.verified_name.value
        if len(vparts) >= 2 or person_company:
            if not any(N.name_agreement(vparts, h)[0] for _, h in hit.names):
                checks["NAME_MISMATCH_VERIFIED"] = Check(True, person.verified_name.sources,
                                                         f"The customer's verified legal name ({' '.join(vparts).title()}, {', '.join(person.verified_name.sources)}) agrees with none of the hit's names.")

    held, held_ev = _safety_net(person, checks)

    # agreement codes
    loc_src = sorted({s for c in overlap for s in person.countries[c]})
    name_txt = f"The customer's name agrees with the hit's name {(matched_name or '').rstrip('. ')}."
    dob_txt = (f"The customer's date of birth {N.fmt_date(cust_dob)} ({', '.join(dob_src) or 'screened'}) agrees with the hit's "
               f"{' or '.join(N.fmt_date(h) for h in hit.dobs)}.") if cust_dob and hit.dobs else ""
    loc_txt = f"Both the customer and the hit are linked to {', '.join(sorted(overlap))}." if overlap else ""
    if name_agree and name_usable and dob_agree and not person_company:
        ev = cid + dob_src + [e for e in hit.dob_evidence if not e.startswith(hit.hit_id)]
        if overlap:
            checks["NAME_DOB_LOCATION_MATCH"] = Check(True, ev + loc_src, f"{dob_txt} {loc_txt} {name_txt}")
        checks["NAME_DOB_MATCH"] = Check(True, ev, f"{dob_txt} {name_txt}")
    if name_agree and overlap and person_company and hit_company:
        checks["NAME_LOCATION_MATCH"] = Check(True, cid + loc_src, f"{loc_txt} {name_txt}")

    missing = []
    if not name_agree:
        missing.append("the names do not clearly agree")
    if not cust_dob or not hit.dobs:
        missing.append("the date of birth cannot be compared with it")
    elif not dob_agree:
        missing.append("the dates of birth neither agree nor contradict")
    parts = " ".join(t for t in (loc_txt, name_txt if name_agree else "") if t)
    if held:
        text = ((parts + " " if parts else "") + " ".join(held)
                + (" Otherwise " + " and ".join(missing) + "." if missing else "")
                + " The hit is kept for a person to review rather than cleared on uncertain evidence.")
        checks["INSUFFICIENT_DATA"] = Check(True, (loc_src or cid) + held_ev, text)
    else:
        checks["INSUFFICIENT_DATA"] = Check(True, loc_src or cid,
                                            (parts + " " if parts else "") + "Nothing in the customer's records contradicts the hit, and "
                                            + (" and ".join(missing) or "the evidence is not enough to confirm it")
                                            + ", so the hit can be neither cleared nor confirmed.")
    return Comparison(person, hit, checks, name_agree, name_frac, name_matched, name_sim, name_usable,
                      matched_name, dob_agree, dob_contra, year_gap, overlap, age_at_report)


SAFETY_NET = True  # switch off only to measure what it costs


def _safety_net(person, checks):
    """Un-fire clears that rest on evidence we may have misread (a wrong clear costs 60, a review 2-3).

    1. LOCATION_MISMATCH while a note looks like it says where the customer lived but no country could be read.
    2. NAME_MISMATCH_VERIFIED from a free-text note whose extracted name shares no part with the customer's
       record names, or from a note we only partly understood: the name extraction may have grabbed the wrong words.
    3. DOB_MISMATCH on an unverified (declared) date of birth while an identity note could not be fully read:
       the verified date may be the one we missed.
    """
    if not SAFETY_NET:
        return [], []
    held, evidence = [], []
    unread = list(person.unread_notes)
    if checks["LOCATION_MISMATCH"].fired and unread:
        checks["LOCATION_MISMATCH"] = Check(False)
        held.append(f"The countries look different, but the customer's notes ({', '.join(unread[:3])}) mention "
                    f"where they lived or their identity in a way that could not be read reliably.")
        evidence += unread[:3]
    vn = person.verified_name
    if checks["NAME_MISMATCH_VERIFIED"].fired and vn and any(s.startswith("NOTE-") for s in vn.sources):
        shares = any(N.parts_match(p, q) for f in person.names for q in f.value for p in vn.value)
        if not shares or any(s in unread for s in vn.sources):
            checks["NAME_MISMATCH_VERIFIED"] = Check(False)
            held.append(f"The legal name read from {', '.join(vn.sources)} ({' '.join(vn.value).title()}) differs from "
                        f"the hit's names, but it shares nothing with the customer's own records, so it may have been misread.")
            evidence += vn.sources
    if checks["DOB_MISMATCH"].fired and not person.dob_verified and any(n.startswith("NOTE-") for n in unread):
        checks["DOB_MISMATCH"] = Check(False)
        held.append("The declared date of birth contradicts the hit, but an identity note could not be fully read, "
                    "so the verified date of birth may differ.")
        evidence += [n for n in unread if n.startswith("NOTE-")][:2]
    return held, evidence


CLEAR_ORDER = ["ENTITY_TYPE_MISMATCH", "DECEASED", "TEMPORAL_IMPOSSIBILITY", "DOB_MISMATCH",
               "LOCATION_MISMATCH", "NAME_MISMATCH_VERIFIED"]
CONFIRM_ORDER = ["NAME_DOB_LOCATION_MATCH", "NAME_DOB_MATCH", "NAME_LOCATION_MATCH"]


def teacher_code(comp):
    for code in CLEAR_ORDER:
        if comp.checks[code].fired:
            return code
    for code in CONFIRM_ORDER:
        if comp.checks[code].fired:
            return code
    return "INSUFFICIENT_DATA"


@dataclass
class HitCase:
    """Everything needed to decide one hit: the comparisons (one per candidate person) and the teacher's answer."""
    hit_id: str
    alert_id: str
    resolution: object
    comparisons: list
    customer_record_id: object
    teacher_decision: str
    teacher_code: str
    teacher_evidence: list
    teacher_reason: str
    alert: dict = None


def decide_case(resolution, hit_facts, alert):
    persons = resolution.persons or [screened_person(resolution.screened)]
    comps = [compare(p, hit_facts, alert.get("screened_at")) for p in persons]
    codes = [teacher_code(c) for c in comps]
    decisions = {CODE_DECISION[c] for c in codes}
    if len(persons) > 1:
        record = "AMBIGUOUS"
    else:
        record = persons[0].primary
    if len(decisions) > 1:
        code = "AMBIGUOUS_CUSTOMER_RECORD"
        evidence = sorted({p.primary for p in persons if p.primary})
        per_person = [(p.primary, CODE_DECISION[c], c) for p, c in zip(persons, codes)]
        reason = reasoning.compose(resolution, alert, None, code, "REVIEW", "", per_person)
    else:
        code = codes[0]
        evidence = comps[0].checks[code].evidence
        reason = reasoning.compose(resolution, alert, comps[0], code, CODE_DECISION[code], comps[0].checks[code].text)
    return HitCase(hit_facts.hit_id, alert["alert_id"], resolution, comps, record,
                   CODE_DECISION[code], code, clean_evidence(evidence), reason.strip(), alert)


def clean_evidence(ids):
    """Bank and search record IDs only, de-duplicated, in order (the hit itself is implied)."""
    return list(dict.fromkeys(i for i in ids if i and "#" not in i))


def evidence_for(case, code):
    """Evidence and reason text for any code the model picks (falls back to the teacher's when not fired)."""
    if code == "AMBIGUOUS_CUSTOMER_RECORD" or not case.comparisons:
        return case.teacher_evidence, case.teacher_reason
    check = case.comparisons[0].checks.get(code)
    if check and check.fired and check.text:
        text = reasoning.compose(case.resolution, case.alert or {}, case.comparisons[0], code, CODE_DECISION[code], check.text)
        return clean_evidence(check.evidence), text
    return case.teacher_evidence, case.teacher_reason
