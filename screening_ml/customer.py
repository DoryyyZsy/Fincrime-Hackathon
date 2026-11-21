"""Customer side: find who the alert is about (principle 2) and assemble one picture per person (3, 4)."""
import re
from dataclasses import dataclass, field

from . import normalize as N

# A KYC note is evidence only when it says the check passed and does not say it failed.
# "not verified", "did not pass", "resubmit" and the like are rejections even though
# none of the development-set notes use them. Unsure stays unused (principle 4).
_POSITIVE_KYC = re.compile(
    r"(?i)\b(?:verif\w*|approv\w*|confirm\w*|cleared|clears|pass(?:ed|es|ing)?|"
    r"success\w*|validat\w*|complet\w*|match(?:es|ed|ing)?|accept(?:ed|s|ance)?)\b"
)
_NEGATIVE_KYC = re.compile(
    r"(?i)(?:"
    r"\breject|\bfail(?:ed|ure|s)?|\bexpired|\bdeclin|"
    r"\bcould not|\bcouldn'?t|\bunable|"
    r"\bdid not match|\bdidn'?t match|\bmismatch|\bdenied|"
    r"\bunsuccess|\bnot verified|\bnot approved|\bnot confirmed|\bnot validated|\bnot cleared|"
    r"\bnot completed|\bdid not pass|\bdidn'?t pass|\bnot pass|"
    r"\bdid not (?:clear|verify|complete|succeed)|"
    r"\bresubmit|\bincomplete|\bpending|\binconclusive|"
    r"\bwithdrawn|\bcancell?ed"
    r")"
)
# Birth cues. The date beside the cue is the DOB; the first other full date in the
# note is not, because that other date is usually the day the check was run.
_DOB_CUE = re.compile(
    r"(?i)\b(?:date of birth|birth\s*date|birthdate|birthday|d\.?\s*o\.?\s*b\.?|born|birth)\b"
)
_CUE_WINDOW = 40
_MAX_PERSON_AGE = 110
# "verified on 2024-07-12" is the check, not a birth date. "born on 6 May 1972" is.
_CHECK_ON = re.compile(r"(?i)(?:\bon|\bas of)\s*$")


@dataclass
class Fact:
    """A value plus the record IDs it came from (for the evidence list)."""
    value: object
    sources: list


@dataclass
class Person:
    records: list                                   # CUS ids of this one person
    primary: str                                    # record the alert resolved to
    entity_type: str
    names: list = field(default_factory=list)       # Fact(list of parts)
    verified_name: Fact = None                      # from latest approved check or KYC note
    dob: Fact = None                                # verified dob preferred, else record dob
    dob_verified: bool = False
    countries: dict = field(default_factory=dict)   # country -> [source ids]
    n_approved_checks: int = 0
    n_rejected_checks: int = 0
    # A KYC note was present but could not be read as a passed check. Its document
    # country and DOB are unknown, not known to contradict the hit.
    identity_unreadable: bool = False
    # Regex and the optional model disagreed about a birth date, so the date is unused.
    dob_withheld: bool = False
    # The birth date came only from a model quote that was checked against the note.
    llm_dob_only: bool = False
    cities: dict = field(default_factory=dict)


@dataclass
class Resolution:
    link_type: str                                  # account | login | email | none
    candidates: list                                # CUS ids that fit the alert
    persons: list                                   # Person objects (more than one = ambiguous)
    screened: dict


def _screen_fits(room, cid, screened):
    """Does a customer record fit what was screened? Unknown parts never disqualify."""
    c = room.customer_by_id[cid]
    company = c.get("entity_type") == "Company"
    s_parts = N.name_parts(screened.get("name"), company)
    names = [N.name_parts(c.get(f), company) for f in ("full_name", "preferred_name", "login_name")]
    if s_parts and not any(N.name_agreement(s_parts, n)[0] for n in names if n):
        return False
    s_dob, c_dob = N.parse_date(screened.get("dob")), N.parse_date(c.get("dob"))
    if s_dob and c_dob and any(a is not None and b is not None and a != b for a, b in zip(s_dob, c_dob)):
        return False
    s_country = N.canon_country(screened.get("country"))
    if s_country:
        mine = {N.canon_country(c.get("residence_country")), N.canon_country(c.get("nationality"))}
        if s_country not in mine:
            return False
    city = (screened.get("city") or "").strip().lower()
    if city:
        if city not in (c.get("city") or "").lower() and city not in (c.get("address") or "").lower():
            return False
    return True


def _search_candidates(room, screened):
    company = screened.get("entity_type") == "Company"
    parts = N.name_parts(screened.get("name"), company)
    pool = set()
    for p in parts:
        pool |= room.customers_by_part.get(p, set())
    dob = N.parse_date(screened.get("dob"))
    if dob and all(dob):
        pool |= room.customers_by_dob.get(dob, set())
    return sorted(cid for cid in pool if _screen_fits(room, cid, screened))


def resolve(room, alert, extractions=None):
    screened = dict(alert["customer"])
    if alert.get("account_ref"):
        link, linked = "account", list(room.holders_by_account.get(alert["account_ref"], []))
    elif alert.get("login_handle"):
        link, linked = "login", list(room.customer_by_login.get(alert["login_handle"].lower(), []))
    elif alert.get("email"):
        link, linked = "email", list(room.customer_by_email.get(alert["email"].lower(), []))
    else:
        link, linked = "none", None

    if linked is None or not linked:
        candidates = _search_candidates(room, screened)
    else:
        linked = list(dict.fromkeys(linked))
        fitting = [cid for cid in linked if _screen_fits(room, cid, screened)]
        candidates = fitting or (linked if len(linked) == 1 else fitting)  # a link can lead to several people

    persons, seen = [], set()
    for cid in candidates:
        root = room.person_of.get(cid, cid)
        if root in seen:
            continue
        seen.add(root)
        persons.append(build_person(room, cid, alert.get("screened_at"), extractions))
    return Resolution(link, candidates, persons, screened)


def kyc_note_failed(text):
    """True when a KYC note says the check did not pass, or never says that it did."""
    if not text:
        return True
    if _NEGATIVE_KYC.search(text):
        return True
    return _POSITIVE_KYC.search(text) is None


def _plausible_person_dob(dob, *refs):
    """A person's birth date: a real day, not after a known reference, age 0-110 there."""
    if not dob or dob[0] is None:
        return False
    year, month, day = dob
    if month and day and not N.is_real_date(year, month, day):
        return False
    for ref in refs:
        if not ref or ref[0] is None:
            continue
        earliest = (year, month or 1, day or 1)
        latest_ref = (ref[0], ref[1] or 12, ref[2] or 31)
        if earliest > latest_ref:
            return False
        if ref[0] - year > _MAX_PERSON_AGE or ref[0] - year < 0:
            return False
    return True


def _merge_dates(dates):
    """Combine dates that do not disagree. Conflicting parts → None (unknown)."""
    merged = dates[0]
    for other in dates[1:]:
        if any(left is not None and right is not None and left != right for left, right in zip(merged, other)):
            return None
        merged = tuple(left if left is not None else right for left, right in zip(merged, other))
    return merged


def _is_check_run_date(text, start):
    """A date introduced by 'on' or 'as of' is when the check ran, unless a birth word leads into it.

    'verified on 2024-07-12' is the check. 'born in France on 14 February 1953' is the DOB.
    """
    pre = text[max(0, start - 48):start]
    found = _CHECK_ON.search(pre)
    if not found:
        return False
    return re.search(r"(?i)\b(?:born|birth|d\.?\s*o\.?\s*b\.?)\b", pre[:found.start()]) is None


def _dob_from_kyc_text(text, created, screened_at):
    """DOB stated next to a birth cue, or a date that is the whole parenthesis.

    The day the check was written is not a birth date, and shifting `created_at`
    must not make it one. Two birth dates that disagree → unknown.

    Slash dates in these notes stay day-first. Every unambiguous slash date in the
    bank's customer records, KYC checks and KYC notes has the day first (the first
    number is above 12 thousands of times and the second number never is), and the
    screened date agrees with that reading. Hit profiles do not: those use
    `date_readings`.
    """
    text = text or ""
    # Partial dates count: "DOB 1999" is a year, and the full date later in the
    # sentence is the day the check ran. Requiring a full date made that day the DOB.
    spans = [span for span in N.date_spans(text) if not _is_check_run_date(text, span[1])]
    candidates = []
    for cue in _DOB_CUE.finditer(text):
        nearest = None
        for date, start, end, raw in spans:
            if end <= cue.start():
                gap = cue.start() - end
            elif cue.end() <= start:
                gap = start - cue.end()
            else:
                gap = 0
            if gap > _CUE_WINDOW:
                continue
            if nearest is None or gap < nearest[0]:
                nearest = (gap, date, raw)
        if nearest is None:
            continue
        candidates.append(nearest[1])
    for match in re.finditer(r"\(([^()]{4,48})\)", text or ""):
        inner = [span for span in N.date_spans(match.group(1)) if all(span[0])]
        if len(inner) != 1:
            continue
        raw = inner[0][3]
        leftover = match.group(1).replace(raw, "")
        leftover = re.sub(r"[\s,./\-]+", "", leftover)
        if leftover:
            continue
        candidates.append(inner[0][0])
    plausible, seen = [], set()
    for date in candidates:
        if date in seen or not _plausible_person_dob(date, created, screened_at):
            continue
        seen.add(date)
        plausible.append(date)
    if not plausible:
        return None
    return _merge_dates(plausible)


def _kyc_note_identity(note, screened_at=None):
    """Pull verified name / dob / document country out of a free-text onboarding note."""
    text = note.get("text") or ""
    if kyc_note_failed(text):
        return None
    created = N.parse_date(note.get("created_at"))
    screened = N.parse_date(screened_at) if not isinstance(screened_at, tuple) else screened_at
    dob = _dob_from_kyc_text(text, created, screened)
    caps = re.findall(r"\b[A-Z][A-Z'\-]+(?:\s+[A-Z][A-Z'\-]+)+\b", text)
    caps = [c for c in caps if not re.fullmatch(r"(?:DOB|ID|KYC|USA|UK)(?:\s.*)?", c)]
    name = max(caps, key=len) if caps else None
    countries = N.countries_in_text(text.replace(name, " ") if name else text)  # 'MARCO JORDAN' is not Jordan
    return {"name": name, "dob": dob, "countries": countries, "date": created}


def build_person(room, cid, screened_at=None, extractions=None):
    root = room.person_of.get(cid, cid)
    records = sorted(room.records_of_person.get(root, [cid]))
    primary = room.customer_by_id[cid]
    company = primary.get("entity_type") == "Company"
    person = Person(records=records, primary=cid, entity_type=primary.get("entity_type"))

    verified = []   # (date, name, dob, countries, source)
    record_dobs = []
    # A KYC note we could not treat as a passed check. Its facts are not evidence,
    # and the declared record DOB must not clear the hit in its place: that DOB is
    # often a typo the check was meant to replace (principle 4).
    discarded_kyc_note = False
    dob_conflict = False
    for rid in records:
        c = room.customer_by_id[rid]
        for f in ("full_name", "preferred_name", "login_name"):
            parts = N.name_parts(c.get(f), company)
            if parts:
                person.names.append(Fact(parts, [rid]))
        for f in ("residence_country", "nationality"):
            country = N.canon_country(c.get(f))
            if country:
                person.countries.setdefault(country, []).append(rid)
        dob = N.parse_date(c.get("dob"))
        if dob:
            record_dobs.append(Fact(dob, [rid]))

        for app in room.applicants_by_customer.get(rid, []):
            for k in room.checks_by_applicant.get(app, []):
                if k.get("result") != "approved":
                    person.n_rejected_checks += 1
                    continue
                person.n_approved_checks += 1
                country = N.canon_country(k.get("document_country"))
                if country:
                    person.countries.setdefault(country, []).append(k["check_id"])
                verified.append((N.parse_date(k.get("checked_at")), k.get("verified_name"),
                                 N.parse_date(k.get("verified_dob")), k["check_id"]))

        for note in room.notes_by_email.get((c.get("email") or "").lower(), []):
            if extractions is None:
                if note.get("channel") == "kyc":
                    ident = _kyc_note_identity(note, screened_at)
                    if ident:
                        for country in sorted(ident["countries"]):
                            person.countries.setdefault(country, []).append(note["note_id"])
                        verified.append((ident["date"], ident["name"], ident["dob"], note["note_id"]))
                    else:
                        discarded_kyc_note = True
                else:
                    # Any country a support note names is tied to the customer. Gating on
                    # "lived" / "work" drops the country when the sentence is reworded
                    # ("stayed", "job"), and the missing country becomes a LOCATION_MISMATCH
                    # clearance. An extra country can only remove that clearance.
                    for country in sorted(N.countries_in_text(note.get("text") or "")):
                        person.countries.setdefault(country, []).append(note["note_id"])
                continue
            from .llm_extract import apply_customer_note
            discarded, conflict = apply_customer_note(person, note, screened_at, extractions, verified, _kyc_note_identity)
            discarded_kyc_note = discarded_kyc_note or discarded
            dob_conflict = dob_conflict or conflict

    llm_only = False
    if verified:  # principle 4: the most recent approved identity check wins
        verified.sort(key=lambda v: N.date_key(v[0]))
        last = verified[-1]
        date, name, dob, source = last[0], last[1], last[2], last[3]
        llm_only = bool(last[4]) if len(last) > 4 else False
        if name:
            person.verified_name = Fact(N.name_parts(name, company), [source])
        if dob:
            person.dob, person.dob_verified = Fact(dob, [source]), True
            person.llm_dob_only = llm_only
    if person.dob is None and record_dobs and not discarded_kyc_note:
        own = [f for f in record_dobs if f.sources[0] == cid]
        person.dob = (own or record_dobs)[0]
    person.identity_unreadable = discarded_kyc_note
    person.dob_withheld = dob_conflict
    return person


def screened_person(screened):
    """When no record fits, compare the hit with what was screened (customer_record_id stays null)."""
    company = screened.get("entity_type") == "Company"
    p = Person(records=[], primary=None, entity_type=screened.get("entity_type"))
    parts = N.name_parts(screened.get("name"), company)
    if parts:
        p.names.append(Fact(parts, []))
    dob = N.parse_date(screened.get("dob"))
    if dob:
        p.dob = Fact(dob, [])
    country = N.canon_country(screened.get("country"))
    if country:
        p.countries[country] = []
    return p
