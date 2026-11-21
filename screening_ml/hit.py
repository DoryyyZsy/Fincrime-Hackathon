"""Watchlist side: gather a hit's facts wherever they are (POLICY.md principle 5).

Free-text extraction is regex-based. This is the part most exposed to new wordings in the unseen set;
`extract_ages` is the natural place to plug in an LLM extractor later.
"""
import re
from dataclasses import dataclass, field

from . import normalize as N

LISTING_PHRASES = {
    "sanctions": "a sanctioned individual",
    "pep": "a politically exposed person",
    "enforcement": "the subject of enforcement action",
    "adverse media": "named in adverse media",
    "soe": "a state-owned enterprise official",
}
DEFAULT_PHRASE = "named on a watchlist"
DATED_REPORT = re.compile(r"According to [^;\n]+;\s*([A-Z][a-z]+ \d{1,2}, \d{4}):")

# The first patterns are the development-set wordings. The rest cover the same fact
# said differently ("age 46", "46 years of age", "30 year old", "at age 25 when").
# A number counts only beside an age word or in one of these frames, so "28 years
# in prison" and a fine of 1,500 do not become a birth year.
_AGE_PATTERNS = [
    re.compile(r"\baged?\s+(\d{1,3})\b", re.I),
    re.compile(r"\b(\d{1,3})\s*-?\s*years?\s*-?\s*old\b", re.I),
    re.compile(r"\b(\d{1,3})\s+years?\s+of\s+age\b", re.I),
    re.compile(r"\bage\s+of\s+(\d{1,3})\b", re.I),
    re.compile(r"\b(?:he|she|they)\s+(?:was|were|is)\s+(?:age\s+)?(\d{1,3})\b", re.I),
    re.compile(r"\bat\s+which\s+point\s+(?:he|she|they)\s+(?:was|were)\s+(?:age\s+)?(\d{1,3})\b", re.I),
    re.compile(r"\bwas\s+(?:age\s+)?(\d{1,3})\s+when\b", re.I),
    re.compile(r"\bat\s+age\s+(\d{1,3})\b", re.I),
    re.compile(r"[^\W\d_],\s*(?:age\s+)?(\d{1,3}),"),  # "Elena Johansson, 51, ..." or ", age 51,"
    re.compile(r"\b(\d{1,3})\s*-?\s*yo\b", re.I),
    re.compile(r"[^\W\d_] \((\d{1,3})\)"),            # "John Smith (41) was sentenced ..."
    re.compile(r"(?:^|[.;]\s*)At (\d{1,3}),"),        # "At 52, John Smith was appointed ..."
    re.compile(r"\bwho (?:was|were|is) (\d{1,3})\b(?! ?(?:%|percent|per cent|times|counts?|years? (?:ago|later|before|after)))", re.I),
    re.compile(r"\b(?:turning|turned|turns) (\d{1,3})\b(?!\s*(?:%|percent|per cent))", re.I),
    # only after a person: "Drozd was 31 at the time", "his age was 24." (not "the fine was 50.")
    re.compile(r"(?:\b(?:he|she|they|who|age)|\b[A-Z][^\W\d_]+) (?:was|were|is) (\d{1,3})"
               r"(?= at the time| at that time| then|\s*[,.;)]|\s*$)"),
    re.compile(r"\bthen (?:aged )?(\d{1,3})\b(?=\s*[,.;)]|\s*$| years)", re.I),   # "Pere Punti, then 37,"
    re.compile(r"(?:\b[A-Z][^\W\d_]+|\b(?:he|she|they)),? at (\d{1,3})(?=\s*[,;)])"),  # "Mark Phillips, at 24, was"
    re.compile(r"[^\W\d_], (\d{1,3})(?=\.\s|\.$|\s*$)"),                          # "... included David Corral, 41."
]
_AGE_WORD = re.compile(r"(?i)\b(?:aged?|years?\s+of\s+age|years?\s+old|year[-\s]old)\b")
_BORN = re.compile(r"\bborn (?:on |in )?([^.;\n]{4,40})", re.I)
# Read only the phrase attached to "born": "born in 1973, was designated on July 14, 2026" must give 1973,
# never the designation date (that turned a confirmed OFAC match into a clearance under rewording).
_BORN_STOP = re.compile(r",(?!\s*\d)|\b(?:and|was|were|is|who|which|but|when)\b", re.I)  # keeps "July 17, 1939"
# Wordings that say "this date is a birth date" without the date right after "born":
# "In 1952, X was born.", "X entered the world on 17 July 1939", "date of birth: 23 April 1975". Not "birthday".
_BIRTH_CUE = re.compile(r"\b(?:born|birth ?date|date of birth|birth year|year of birth|d\.o\.b\.?|dob|"
                        r"(?:came|come|comes|entered|enter|enters) (?:into )?(?:the|this) world)\b", re.I)
# Any sign that the text states an age or a birth date. When it is present but nothing could be read,
# the hit's birth date is unknown for a reason we cannot see, so it must not be replaced by a location clear.
_DOB_SIGNAL = re.compile(r"(?i)\b(?:born|birth|aged?|years? old|year-old|years? of age|turn(?:ed|ing|s)\s+\d|"
                         r"(?:came|entered) (?:into )?the world)\b")


@dataclass
class HitFacts:
    hit_id: str
    entity_type: str
    names: list = field(default_factory=list)          # (display name, parts)
    dobs: list = field(default_factory=list)           # possible birth dates (y, m, d)
    dob_source: str = "none"                           # field | notes | search | none
    dob_evidence: list = field(default_factory=list)   # SR ids or "H-xxx#field"
    dob_text: str = ""
    countries: set = field(default_factory=set)
    cities: list = field(default_factory=list)
    # Regex and the optional model disagreed, or a search hit was rejected, so no
    # birth date from that text is used and a location clearance may not replace it.
    dob_withheld: bool = False
    # Birth years came only from a model quote checked against the profile or snippet.
    llm_dob_only: bool = False
    earliest_report: tuple = None
    deceased: tuple = None
    category: str = ""
    phrase: str = DEFAULT_PHRASE
    rich: bool = False


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]


def _age_number(sentence):
    """The age stated in one sentence, or None when no age wording is present."""
    for pattern in _AGE_PATTERNS:
        match = pattern.search(sentence)
        if not match:
            continue
        # A pattern may have the number in a later group ("he was age 24").
        raw = next(group for group in match.groups() if group)
        age = int(raw)
        if 0 < age < 110:
            return age
    for word in _AGE_WORD.finditer(sentence):
        start = max(0, word.start() - 16)
        window = sentence[start: word.end() + 16]
        best = None
        for number in re.finditer(r"\b(\d{1,3})\b", window):
            raw = number.group(1)
            if len(raw) > 1 and raw.startswith("0"):  # 004 from a register number, not an age
                continue
            age = int(raw)
            if not 0 < age < 110:
                continue
            distance = abs((start + number.start()) - word.start())
            if best is None or distance < best[0]:
                best = (distance, age)
        if best:
            return best[1]
    return None


def extract_ages(text):
    """Birth-year candidates from 'aged 45 in March 2019' style sentences: two years per age.

    Slash dates in the sentence are not assumed day-first: 6/20/2019 is 20 June, and
    an ambiguous 05/06/2019 still yields 2019 as the reference year. A 'born on'
    slash date contributes every calendar-valid reading instead of one hard value.
    """
    years, quotes = [], []
    for sentence in _sentences(text):
        ref = N.find_dates(sentence, flexible_slash=True)
        if not ref:
            continue
        age = _age_number(sentence)
        if age is not None:
            ref_year = ref[0][0]
            if 10 <= age <= 105:  # "(2)" or "aged 0" are not ages of someone on a watchlist
                years += [ref_year - age - 1, ref_year - age]
                quotes.append(sentence)
            continue
        born = _BORN.search(sentence)
        window = _BORN_STOP.split(born.group(1))[0] if born else ""
        readings = N.date_readings(window) if window else []
        if not readings and _BIRTH_CUE.search(sentence):
            # the birth word is not directly followed by the date: take it only when the sentence holds
            # exactly one date, so an unrelated date can never become a birth date
            distinct = list(dict.fromkeys(ref))
            readings = distinct if len(distinct) == 1 else []
        if readings:
            years.extend(readings)
            quotes.append(sentence)
    out = []
    for year in years:
        date = year if isinstance(year, tuple) else (year, None, None)
        if date not in out:
            out.append(date)
    return out, " ".join(quotes)


def hit_countries(hit):
    countries = set()
    for f in ("countries", "nationality", "citizenship", "addresses"):
        for value in hit.get(f) or []:
            c = N.canon_country(value)
            if c:
                countries.add(c)
    if hit.get("place_of_birth"):
        c = N.canon_country(hit["place_of_birth"])
        if c:
            countries.add(c)
    return countries


def search_key(name):
    """Search results are keyed by the name's accent-free lowercase words in alphabetical order."""
    folded = N._fold(name).lower()
    return " ".join(sorted(re.sub(r"[^a-z0-9\s]", " ", folded).split()))


def build_hit(room, hit, extractions=None):
    company = hit.get("entity_type") == "Company"
    facts = HitFacts(hit_id=hit["hit_id"], entity_type=hit.get("entity_type"), rich=bool(hit.get("primary_category")))
    names = [hit.get("name"), hit.get("best_matching_name"),
             " ".join(x for x in (hit.get("first_name"), hit.get("middle_name"), hit.get("last_name")) if x)]
    names += list(hit.get("aliases") or [])
    for n in names:
        parts = N.name_parts(n, company)
        if parts and all(parts != p for _, p in facts.names):
            facts.names.append((n, parts))

    cats = hit.get("categories") or []
    facts.category = hit.get("primary_category") or (cats[0] if cats else "")
    facts.phrase = LISTING_PHRASES.get(facts.category.lower(), DEFAULT_PHRASE)
    facts.countries = hit_countries(hit)

    # date of birth: dobs field -> first paragraph of the profile notes -> web search results.
    # An ambiguous slash date is both readings, so a clearance needs every reading
    # to contradict and one agreeing reading is enough to stop it.
    for value in hit.get("dobs") or []:
        for date in N.date_readings(value):
            if date not in facts.dobs:
                facts.dobs.append(date)
    if facts.dobs:
        facts.dob_source, facts.dob_evidence = "field", [f"{hit['hit_id']}#dobs"]
    notes = hit.get("profile_notes") or ""
    if extractions is None:
        if not facts.dobs and not company:
            first_para = re.split(r"\n\s*\n", notes.strip())[0] if notes.strip() else ""
            dobs, quote = extract_ages(first_para)
            if dobs:
                facts.dobs, facts.dob_source, facts.dob_text = dobs, "notes", quote
                facts.dob_evidence = [f"{hit['hit_id']}#profile_notes"]
        if not facts.dobs and not company:
            for s in room.search_by_query.get(search_key(hit.get("name") or ""), []):
                snippet = s.get("snippet") or ""
                if facts.phrase.lower() not in snippet.lower():
                    continue
                if not (N.countries_in_text(snippet) & facts.countries):
                    continue
                dobs, quote = extract_ages(snippet)
                if dobs:
                    facts.dobs, facts.dob_source, facts.dob_text = dobs, "search", quote
                    facts.dob_evidence = [s["result_id"]]
                    break
        if not facts.dobs and not company:
            first_para = re.split(r"\n\s*\n", notes.strip())[0] if notes.strip() else ""
            if _DOB_SIGNAL.search(first_para):
                facts.dob_withheld = True   # an age or birth date is stated but could not be read
    else:
        from .llm_extract import apply_hit_extractions
        apply_hit_extractions(facts, hit, room, extractions, extract_ages, search_key, N.countries_in_text)

    reports = [N.parse_date(m) for m in DATED_REPORT.findall(notes)]
    reports = [r for r in reports if r]
    if reports:
        facts.earliest_report = min(reports, key=N.date_key)
    if hit.get("deceased_date"):
        readings = N.date_readings(hit["deceased_date"])
        if readings:
            # Death is before screening only if every reading is. The latest
            # reading is the one `_before` has to clear.
            facts.deceased = max(readings, key=N.date_key)
    return facts
