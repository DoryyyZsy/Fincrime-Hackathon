"""Customer side: find who the alert is about (principle 2) and assemble one picture per person (3, 4)."""
import re
from dataclasses import dataclass, field

from . import normalize as N

# A support note "about where the customer lived" (principle 6). Deliberately broad: the unseen set rewords these
# notes ("established their life and career inside...", "had professional and personal ties to..."), and missing
# one can turn a REVIEW into an unsafe clear. Adding a country can only make a location clear less likely.
RESIDENCE_WORDS = re.compile(
    r"(?i)resid|\blived?\b|living|\bbased\b|\bbase\b|employ|\bwork|career|profession|personal life|\blife\b|"
    r"\bties\b|moved|relocat|located|\bhome\b|\bstay|abroad|emigrat|immigrat|settled|address")
YEAR = re.compile(r"\b(?:19|20)\d\d\b")
NEGATIVE_KYC = re.compile(r"(?i)reject|fail|expired|declin|could not|unable|did not match|mismatch|denied|unsuccess")


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
    unread_notes: list = field(default_factory=list)  # notes that look like residence/identity info we could not parse


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


def resolve(room, alert):
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
        persons.append(build_person(room, cid))
    return Resolution(link, candidates, persons, screened)


def _kyc_note_identity(note):
    """Pull verified name / dob / document country out of a free-text onboarding note."""
    text = note["text"]
    if NEGATIVE_KYC.search(text):
        return None
    created = N.parse_date(note.get("created_at"))
    dates = [d for d in N.find_dates(text) if d != created]
    full = [d for d in dates if all(d)]
    dob = full[0] if full else (dates[0] if dates else None)   # "DOB 1974" still gives a birth year
    name = _caps_name(text)
    countries = N.countries_in_text(text.replace(name, " ") if name else text)  # 'MARCO JORDAN' is not Jordan
    return {"name": name, "dob": dob, "countries": countries, "date": created}


_NOT_NAME = {"DOB", "ID", "KYC", "USA", "UK", "US", "EU", "UAE", "AML", "PEP", "SSN", "NIN", "BVN", "OK"}


def _caps_name(text):
    """The longest run of ALL-CAPS words (accents allowed: FERNÁNDEZ), i.e. the name as written on the document."""
    runs, cur = [], []
    for m in re.finditer(r"[^\s,;:()]+", text):
        tok = re.sub(r"(?:['’]s|\.)$", "", m.group(0))
        if any(c.isalpha() for c in tok) and tok == tok.upper() and tok.upper() not in _NOT_NAME:
            cur.append(tok)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    if not runs:
        return None
    best = max(runs, key=lambda r: (len(r), len(" ".join(r))))
    return " ".join(best)


def build_person(room, cid):
    root = room.person_of.get(cid, cid)
    records = sorted(room.records_of_person.get(root, [cid]))
    primary = room.customer_by_id[cid]
    company = primary.get("entity_type") == "Company"
    person = Person(records=records, primary=cid, entity_type=primary.get("entity_type"))

    verified = []   # (date, name, dob, countries, source)
    record_dobs = []
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
            if note.get("channel") == "kyc":
                ident = _kyc_note_identity(note)
                if ident:
                    for country in ident["countries"]:
                        person.countries.setdefault(country, []).append(note["note_id"])
                    verified.append((ident["date"], ident["name"], ident["dob"], note["note_id"]))
                    if not ident["dob"] or not ident["name"] or not ident["countries"]:
                        person.unread_notes.append(note["note_id"])   # identity note only partly understood
            elif RESIDENCE_WORDS.search(note["text"]) or YEAR.search(note["text"]):
                found = N.countries_in_text(note["text"])
                for country in found:
                    person.countries.setdefault(country, []).append(note["note_id"])
                if not found and RESIDENCE_WORDS.search(note["text"]) and YEAR.search(note["text"]):
                    person.unread_notes.append(note["note_id"])       # says where they lived, but not readably

    if verified:  # principle 4: the most recent approved identity check wins
        verified.sort(key=lambda v: N.date_key(v[0]))
        date, name, dob, source = verified[-1]
        if name:
            person.verified_name = Fact(N.name_parts(name, company), [source])
        if dob:
            person.dob, person.dob_verified = Fact(dob, [source]), True
    if person.dob is None and record_dobs:
        own = [f for f in record_dobs if f.sources[0] == cid]
        person.dob = (own or record_dobs)[0]
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
