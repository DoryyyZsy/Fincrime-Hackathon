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

_AGE_PATTERNS = [
    re.compile(r"\baged? (\d{1,3})\b"),
    re.compile(r"\b(\d{1,3})-year-old\b"),
    re.compile(r"\b(\d{1,3}) years? old\b"),
    re.compile(r"\bage of (\d{1,3})\b"),
    re.compile(r"\b(?:he|she|they) (?:was|were|is) (\d{1,3})\b", re.I),
    re.compile(r"\bat which point (?:he|she|they) (?:was|were) (\d{1,3})\b", re.I),
    re.compile(r"\bwas (\d{1,3}) when\b"),
    re.compile(r"[^\W\d_], (\d{1,3}),"),       # "Elena Johansson, 51, ..." (any letter, incl. accents)
]
_BORN = re.compile(r"\bborn (?:on |in )?([^.;\n]{4,30})", re.I)


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
    earliest_report: tuple = None
    deceased: tuple = None
    category: str = ""
    phrase: str = DEFAULT_PHRASE
    rich: bool = False


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]


def extract_ages(text):
    """Birth-year candidates from 'aged 45 in March 2019' style sentences: two years per age."""
    years, quotes = [], []
    for sentence in _sentences(text):
        ref = [d for d in N.find_dates(sentence)]
        if not ref:
            continue
        for pattern in _AGE_PATTERNS:
            m = pattern.search(sentence)
            if not m:
                continue
            age = int(m.group(1))
            ref_year = ref[0][0]
            if 0 < age < 110:
                years += [ref_year - age - 1, ref_year - age]
                quotes.append(sentence)
                break
        else:
            m = _BORN.search(sentence)
            if m:
                d = N.parse_date(m.group(1))
                if d:
                    years.append(d)
                    quotes.append(sentence)
    out = []
    for y in years:
        d = y if isinstance(y, tuple) else (y, None, None)
        if d not in out:
            out.append(d)
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


def build_hit(room, hit):
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

    # date of birth: dobs field -> first paragraph of the profile notes -> web search results
    for value in hit.get("dobs") or []:
        d = N.parse_date(value)
        if d and d not in facts.dobs:
            facts.dobs.append(d)
    if facts.dobs:
        facts.dob_source, facts.dob_evidence = "field", [f"{hit['hit_id']}#dobs"]
    notes = hit.get("profile_notes") or ""
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

    reports = [N.parse_date(m) for m in DATED_REPORT.findall(notes)]
    reports = [r for r in reports if r]
    if reports:
        facts.earliest_report = min(reports, key=N.date_key)
    if hit.get("deceased_date"):
        facts.deceased = N.parse_date(hit["deceased_date"])
    return facts
