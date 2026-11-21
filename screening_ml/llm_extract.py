"""Optional OpenRouter extractors. The rule engine still decides every hit.

No key, a failed call, a timeout, or JSON that does not match the schema: that
text is unknown and the regex extractor stands. A model fact is kept only when
its quote appears in the source. Where the model and the regex disagree, the
fact becomes unknown, which cannot open a new clearance.
"""
import hashlib
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import normalize as N

# Pinned model. Prices are OpenRouter's posted standard rates for this id
# ($0.30 / $2.50 per 1M tokens). Thinking is switched off so output stays the JSON.
MODEL = "google/gemini-2.5-flash"
PRICE_INPUT_PER_TOKEN = 0.30 / 1_000_000
PRICE_OUTPUT_PER_TOKEN = 2.50 / 1_000_000
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
BATCH = 8
WORKERS = 8
TIMEOUT_SECONDS = 12
MAX_CHARS = 4000
MAX_OUTPUT_TOKENS = 2500

_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)

# Fixed prompts. The cache key is the hash of (model, prompt, item text).
PROMPT_NOTE = """You extract identity facts from one bank KYC or support note. Reply with JSON only, no markdown.
Schema: {"items":[{"id":string,"outcome":"passed"|"rejected"|"unclear","dob":null|{"year":int,"month":int|null,"day":int|null,"quote":string},"countries":[{"name":string,"quote":string}],"cities":[{"name":string,"quote":string}]}]}
Rules:
- outcome is passed only if the text says the identity check succeeded. rejected if it failed, expired, or must be resubmitted. unclear otherwise. A support note that is not about an identity check is unclear.
- dob is the customer's date of birth only. A date that says when the check ran, a document was issued, or anything else is not a dob: return null. year/month/day must be copied from the quote, not inferred. quote is an exact substring of the note.
- countries and cities are only places the note states. name is the place. quote is an exact substring that contains that name. Do not infer a country from a city.
- One item per input id, same ids, same order. Unknown stays null or an empty list. Never guess."""

PROMPT_PROFILE = """You extract birth and location facts from a watchlist profile paragraph. Reply with JSON only, no markdown.
Schema: {"items":[{"id":string,"birth_years":[int],"birth_quote":string,"countries":[{"name":string,"quote":string}],"cities":[{"name":string,"quote":string}]}]}
Rules:
- birth_years is empty unless the paragraph states a birth date or an age at a dated event. Someone aged 45 in March 2019 was born in 1973 or 1974: return both years. A year alone returns that year. birth_quote is an exact substring containing the age or birth date. Do not invent years the quote does not support.
- countries and cities are places the paragraph states. quote is an exact substring containing the name. Do not infer a country from a city.
- One item per input id, same ids, same order. Unknown stays an empty list or empty string. Never guess."""

PROMPT_SEARCH = """You decide whether a web snippet is about one listed person or a namesake, and extract birth and location facts. Reply with JSON only, no markdown.
Schema: {"items":[{"id":string,"about_listed_person":bool,"confidence":"high"|"low","birth_years":[int],"birth_quote":string,"countries":[{"name":string,"quote":string}]}]}
Rules:
- about_listed_person is true only when the snippet describes the listed person in context (name plus the same kind of listing). A different person who shares the name is false. confidence is high only when the text itself settles it; otherwise low.
- birth_years follows the same age rule as a profile: aged 45 in March 2019 means 1973 and 1974. birth_quote is an exact substring. Empty when the snippet gives no birth fact.
- countries are places the snippet states about that person. quote contains the name. Do not infer a country from a city.
- One item per input id, same ids, same order. Never guess. Low confidence is the right answer when unsure."""

PROMPTS = {"note": PROMPT_NOTE, "support": PROMPT_NOTE, "profile": PROMPT_PROFILE, "search": PROMPT_SEARCH}


def verbatim(quote, source):
    """True when quote is a non-empty exact substring, ignoring only extra whitespace."""
    if not isinstance(quote, str) or not isinstance(source, str):
        return False
    quote = quote.strip()
    if len(quote) < 2:
        return False
    if quote in source:
        return True
    collapsed = re.sub(r"\s+", " ", quote)
    return collapsed in re.sub(r"\s+", " ", source)


def cache_key(model, prompt, text):
    digest = hashlib.sha256()
    digest.update(model.encode())
    digest.update(b"\0")
    digest.update(prompt.encode())
    digest.update(b"\0")
    digest.update(text.encode())
    return digest.hexdigest()


def _years_follow_quote(years, quote):
    """Reject a birth year the quote cannot support. Unknown is safer than a guess."""
    if not years or not isinstance(quote, str):
        return False
    cleaned = []
    for year in years:
        if isinstance(year, bool) or not isinstance(year, int):
            return False
        if year < 1850 or year > 2035:
            return False
        cleaned.append(year)
    anchors = [int(value) for value in re.findall(r"\b(1[89]\d{2}|20\d{2})\b", quote)]
    ages = []
    for raw in re.findall(r"\b(\d{1,3})\b", quote):
        if len(raw) > 1 and raw.startswith("0"):
            continue
        age = int(raw)
        if 0 < age < 110:
            ages.append(age)
    if not anchors:
        return False
    allowed = set(anchors)
    for anchor in anchors:
        for age in ages:
            allowed.add(anchor - age)
            allowed.add(anchor - age - 1)
    return all(year in allowed for year in cleaned)


def _compatible(left, right):
    return all(a is None or b is None or a == b for a, b in zip(left, right))


def merge_kyc_dob(regex_dob, llm_dob, quote, source):
    """(dob or None, llm_only, conflict).

    The regex date stands when the model is silent. Both dates must agree when
    both exist. A model-only date is kept only with a quote that supports it.
    """
    llm_date = None
    if isinstance(llm_dob, dict) and verbatim(quote, source):
        year, month, day = llm_dob.get("year"), llm_dob.get("month"), llm_dob.get("day")
        if isinstance(year, int) and (month is None or isinstance(month, int)) and (day is None or isinstance(day, int)):
            if _years_follow_quote([year], quote) and N.is_real_date(year, month, day):
                llm_date = (year, month, day)
    if llm_date is None:
        return regex_dob, False, False
    if regex_dob is None:
        return llm_date, True, False
    if _compatible(regex_dob, llm_date):
        return regex_dob, False, False
    return None, False, True


def merge_year_dobs(regex_dobs, years, quote, source):
    """Same contract as merge_kyc_dob, for a list of birth years from a profile or snippet."""
    usable = []
    if verbatim(quote, source) and _years_follow_quote(years, quote):
        for year in dict.fromkeys(years):
            usable.append((year, None, None))
    if not usable:
        return list(regex_dobs), False, False
    if not regex_dobs:
        return usable, True, False
    agreed = [date for date in regex_dobs if date[0] in {item[0] for item in usable}]
    if agreed:
        return agreed, False, False
    return [], False, True


def note_is_evidence(regex_passed, outcome):
    """A note is evidence only when the regex says it passed and the model does not disagree.

    outcome None means the call failed: the regex stands. A regex failure is never
    upgraded to a pass.
    """
    if not regex_passed:
        return False
    if outcome is None:
        return True
    return outcome == "passed"


def quoted_countries(items, source):
    found = []
    if not isinstance(items, list):
        return found
    for item in items:
        if not isinstance(item, dict):
            continue
        name, quote = item.get("name"), item.get("quote")
        if not isinstance(name, str) or not verbatim(quote, source):
            continue
        country = N.canon_country(name)
        if not country:
            continue
        haystack = quote.lower()
        if country.lower() not in haystack and name.lower() not in haystack:
            continue
        if country not in found:
            found.append(country)
    return found


def quoted_cities(items, source):
    found = []
    if not isinstance(items, list):
        return found
    for item in items:
        if not isinstance(item, dict):
            continue
        name, quote = item.get("name"), item.get("quote")
        if not isinstance(name, str) or not verbatim(quote, source):
            continue
        city = re.sub(r"\s+", " ", name).strip()
        if len(city) < 2 or city.lower() not in quote.lower():
            continue
        if city not in found:
            found.append(city)
    return found


@dataclass
class NoteFacts:
    outcome: object = None
    dob: object = None
    quote: str = ""
    countries: list = field(default_factory=list)
    cities: list = field(default_factory=list)


@dataclass
class ProfileFacts:
    years: list = field(default_factory=list)
    quote: str = ""
    countries: list = field(default_factory=list)
    cities: list = field(default_factory=list)


@dataclass
class SearchFacts:
    about: object = None
    confidence: object = None
    years: list = field(default_factory=list)
    quote: str = ""
    countries: list = field(default_factory=list)


@dataclass
class ExtractionBook:
    notes: dict
    profiles: dict
    searches: dict


def _as_int_or_none(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    return value


def _valid_note_item(item, source):
    if not isinstance(item, dict) or item.get("outcome") not in {"passed", "rejected", "unclear"}:
        return None
    dob = item.get("dob")
    quote = ""
    parsed = None
    if dob is not None:
        if not isinstance(dob, dict) or not isinstance(dob.get("quote"), str):
            return None
        year = _as_int_or_none(dob.get("year"))
        month = _as_int_or_none(dob.get("month"))
        day = _as_int_or_none(dob.get("day"))
        if year is False or month is False or day is False or not isinstance(year, int):
            return None
        parsed = {"year": year, "month": month, "day": day}
        quote = dob["quote"]
    return NoteFacts(
        outcome=item["outcome"], dob=parsed, quote=quote,
        countries=quoted_countries(item.get("countries"), source),
        cities=quoted_cities(item.get("cities"), source),
    )


def _valid_profile_item(item, source):
    if not isinstance(item, dict):
        return None
    years = item.get("birth_years")
    quote = item.get("birth_quote", "")
    if not isinstance(years, list) or not isinstance(quote, str):
        return None
    if any(isinstance(year, bool) or not isinstance(year, int) for year in years):
        return None
    return ProfileFacts(
        years=years, quote=quote,
        countries=quoted_countries(item.get("countries"), source),
        cities=quoted_cities(item.get("cities"), source),
    )


def _valid_search_item(item, source):
    if not isinstance(item, dict):
        return None
    if not isinstance(item.get("about_listed_person"), bool) or item.get("confidence") not in {"high", "low"}:
        return None
    years = item.get("birth_years", [])
    quote = item.get("birth_quote", "")
    if not isinstance(years, list) or not isinstance(quote, str):
        return None
    if any(isinstance(year, bool) or not isinstance(year, int) for year in years):
        return None
    return SearchFacts(
        about=item["about_listed_person"], confidence=item["confidence"],
        years=years, quote=quote, countries=quoted_countries(item.get("countries"), source),
    )


def parse_model_json(content):
    if not isinstance(content, str) or not content.strip():
        return None
    text = _JSON_FENCE.sub("", content.strip())
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        return None
    return payload["items"]


def apply_customer_note(person, note, screened_at, book, verified, kyc_identity):
    """Fold one note into a person. Returns (discarded_kyc, dob_conflict)."""
    text = note.get("text") or ""
    facts = book.notes.get(note["note_id"]) if book is not None else None
    if note.get("channel") == "kyc":
        ident = kyc_identity(note, screened_at)
        outcome = facts.outcome if facts is not None else None
        if not note_is_evidence(ident is not None, outcome):
            return True, False
        dob, only, conflict = merge_kyc_dob(
            ident["dob"], None if facts is None else facts.dob, "" if facts is None else facts.quote, text,
        )
        countries = set(ident["countries"])
        if facts is not None:
            countries.update(facts.countries)
            for city in facts.cities:
                person.cities.setdefault(city, []).append(note["note_id"])
        for country in sorted(countries):
            person.countries.setdefault(country, []).append(note["note_id"])
        verified.append((ident["date"], ident["name"], dob, note["note_id"], bool(only and dob is not None)))
        if not dob or not ident["name"] or not (set(ident["countries"]) | (set(facts.countries) if facts is not None else set())):
            person.unread_notes.append(note["note_id"])
        return False, conflict
    from .customer import RESIDENCE_WORDS, YEAR
    countries = set(N.countries_in_text(text))
    if facts is not None:
        countries.update(facts.countries)
        for city in facts.cities:
            person.cities.setdefault(city, []).append(note["note_id"])
    for country in sorted(countries):
        person.countries.setdefault(country, []).append(note["note_id"])
    if not countries and RESIDENCE_WORDS.search(text) and YEAR.search(text):
        person.unread_notes.append(note["note_id"])
    return False, False


def apply_hit_extractions(facts, hit, room, book, extract_ages, search_key, countries_in_text):
    """Merge model facts into a hit. Structured `dobs` are left untouched."""
    company = facts.entity_type == "Company"
    notes = hit.get("profile_notes") or ""
    first = re.split(r"\n\s*\n", notes.strip())[0] if notes.strip() else ""
    profile = book.profiles.get(hit["hit_id"]) if book is not None else None
    if profile is not None and first:
        for country in profile.countries:
            facts.countries.add(country)
        for city in profile.cities:
            if city not in facts.cities:
                facts.cities.append(city)
    if facts.dob_source == "field" or company:
        return
    regex_dobs, regex_quote = extract_ages(first) if first else ([], "")
    if profile is not None and first:
        merged, only, conflict = merge_year_dobs(regex_dobs, profile.years, profile.quote, first)
    else:
        merged, only, conflict = list(regex_dobs), False, False
    if conflict:
        facts.dobs, facts.dob_source, facts.dob_withheld, facts.llm_dob_only = [], "none", True, False
        return
    if merged:
        facts.dobs = merged
        facts.dob_source = "notes"
        facts.dob_text = regex_quote or (profile.quote if profile is not None else "")
        facts.dob_evidence = [f"{hit['hit_id']}#profile_notes"]
        facts.llm_dob_only = only
        return
    _search_dobs(facts, hit, room, book, extract_ages, search_key, countries_in_text)


def _search_dobs(facts, hit, room, book, extract_ages, search_key, countries_in_text):
    rejected_regex = False
    for result in room.search_by_query.get(search_key(hit.get("name") or ""), []):
        snippet = result.get("snippet") or ""
        judgement = book.searches.get(result["result_id"]) if book is not None else None
        regex_ok = (
            facts.phrase.lower() in snippet.lower()
            and bool(countries_in_text(snippet) & facts.countries)
        )
        high = (
            judgement is not None
            and judgement.about is True
            and judgement.confidence == "high"
        )
        if judgement is not None and not high:
            if regex_ok:
                rejected_regex = True
            continue
        if not regex_ok and not (high and judgement is not None and judgement.years):
            continue
        regex_dobs = extract_ages(snippet)[0] if regex_ok else []
        if high:
            for country in judgement.countries:
                facts.countries.add(country)
            merged, only, conflict = merge_year_dobs(regex_dobs, judgement.years, judgement.quote, snippet)
        else:
            merged, only, conflict = list(regex_dobs), False, False
        if conflict:
            facts.dob_withheld = True
            facts.dobs = []
            return
        if merged:
            facts.dobs = merged
            facts.dob_source = "search"
            facts.dob_text = judgement.quote if only and judgement is not None else extract_ages(snippet)[1]
            facts.dob_evidence = [result["result_id"]]
            facts.llm_dob_only = only
            return
    if rejected_regex:
        facts.dob_withheld = True


def reconcile(regex_case, merged_case, hit, resolution):
    """A model may make a clearance harder, or add one only from a quoted birth date.

    Anything else keeps the regex decision. Dropping a regex fact must not open a
    location clearance: that path is blocked before we get here, and a new
    LOCATION_MISMATCH is refused again here.
    """
    if merged_case.teacher_decision != "DIFFERENT_ENTITY":
        return merged_case
    if regex_case.teacher_decision == "DIFFERENT_ENTITY":
        if merged_case.teacher_code == regex_case.teacher_code:
            return merged_case
        return regex_case
    if merged_case.teacher_code not in {"DOB_MISMATCH", "TEMPORAL_IMPOSSIBILITY"}:
        return regex_case
    persons = list(getattr(resolution, "persons", []) or [])
    if getattr(hit, "dob_withheld", False) or any(
        getattr(person, "dob_withheld", False) or getattr(person, "identity_unreadable", False)
        for person in persons
    ):
        return regex_case
    quoted = getattr(hit, "llm_dob_only", False) or any(getattr(person, "llm_dob_only", False) for person in persons)
    if not quoted:
        return regex_case
    return merged_case


class _Stats:
    def __init__(self):
        self.http_calls = 0
        self.cache_hits = 0
        self.failures = 0
        self.items = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.seconds = 0.0

    def cost(self):
        return (
            self.prompt_tokens * PRICE_INPUT_PER_TOKEN
            + self.completion_tokens * PRICE_OUTPUT_PER_TOKEN
        )

    def line(self, mock=False):
        note = "offline mock, no network; " if mock else ""
        return (
            f"llm model {MODEL}: {note}items {self.items}, http {self.http_calls}, "
            f"cache hits {self.cache_hits}, failures {self.failures}, "
            f"{self.seconds:.1f}s, est ${self.cost():.4f} "
            f"({self.prompt_tokens} in / {self.completion_tokens} out tokens)"
        )


class OpenRouterTransport:
    """stdlib urllib. One POST. None on any failure."""

    def __init__(self, api_key):
        self.api_key = api_key

    def complete(self, prompt, user_payload):
        body = {
            "model": MODEL,
            "temperature": 0,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "reasoning": {"max_tokens": 0},
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        }
        request = urllib.request.Request(
            OPENROUTER_URL,
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode())
        choice = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage") or {}
        return choice, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


class EchoTransport:
    """Offline stand-in. Agrees with the regex or says unknown. No network."""

    def complete(self, prompt, user_payload):
        items = []
        for item in user_payload["items"]:
            text = item.get("text") or ""
            kind = user_payload.get("kind")
            if kind in {"note", "support"}:
                from .customer import kyc_note_failed
                outcome = "rejected" if kyc_note_failed(text) else "passed"
                if kind == "support":
                    outcome = "unclear"
                items.append({
                    "id": item["id"], "outcome": outcome, "dob": None,
                    "countries": [], "cities": [],
                })
            elif kind == "profile":
                items.append({
                    "id": item["id"], "birth_years": [], "birth_quote": "",
                    "countries": [], "cities": [],
                })
            else:
                items.append({
                    "id": item["id"], "about_listed_person": True, "confidence": "high",
                    "birth_years": [], "birth_quote": "", "countries": [],
                })
        return json.dumps({"items": items}), 0, 0


class LlmClient:
    def __init__(self, transport):
        self.transport = transport
        self.cache = {}
        self.stats = _Stats()
        self.mock = isinstance(transport, EchoTransport)
        self._lock_import = __import__("threading").Lock()

    @classmethod
    def from_env(cls):
        """None when the model must not run. SCREENING_LLM=0 forces the regex path."""
        if os.environ.get("SCREENING_LLM") == "0":
            return None
        if os.environ.get("SCREENING_LLM_MOCK") == "1":
            return cls(EchoTransport())
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            return None
        return cls(OpenRouterTransport(key))

    def extract(self, room, resolutions):
        jobs = _jobs(room, resolutions)
        self.stats.items = len(jobs)
        started = time.time()
        by_kind = {}
        for kind, item_id, text, context in jobs:
            by_kind.setdefault(kind, []).append((item_id, text, context))
        batches = []
        for kind, rows in by_kind.items():
            rows.sort(key=lambda row: row[0])
            for start in range(0, len(rows), BATCH):
                batches.append((kind, rows[start:start + BATCH]))
        if batches:
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                list(pool.map(lambda batch: self._batch(*batch), batches))
        self.stats.seconds += time.time() - started
        notes, profiles, searches = {}, {}, {}
        for kind, item_id, text, context in jobs:
            stored = self.cache.get(cache_key(MODEL, PROMPTS[kind], _item_text(kind, item_id, text, context)))
            if kind in {"note", "support"}:
                notes[item_id] = stored if isinstance(stored, NoteFacts) else NoteFacts()
            elif kind == "profile":
                profiles[item_id] = stored if isinstance(stored, ProfileFacts) else ProfileFacts()
            else:
                searches[item_id] = stored if isinstance(stored, SearchFacts) else SearchFacts()
        return ExtractionBook(notes, profiles, searches)

    def _batch(self, kind, rows):
        pending = []
        prompt = PROMPTS[kind]
        with self._lock_import:
            for item_id, text, context in rows:
                key = cache_key(MODEL, prompt, _item_text(kind, item_id, text, context))
                if key in self.cache:
                    self.stats.cache_hits += 1
                else:
                    pending.append((item_id, text, context, key))
        if not pending:
            return
        payload = {
            "kind": kind,
            "items": [
                {"id": item_id, "text": text, "context": context}
                for item_id, text, context, _key in pending
            ],
        }
        content = usage_in = usage_out = None
        for _attempt in range(2):
            try:
                content, usage_in, usage_out = self.transport.complete(prompt, payload)
                break
            except Exception:
                content = None
        with self._lock_import:
            self.stats.http_calls += 1
            if content is None:
                self.stats.failures += len(pending)
                for _item_id, _text, _context, key in pending:
                    self.cache[key] = None
                return
            self.stats.prompt_tokens += usage_in or _estimate(prompt) + _estimate(json.dumps(payload))
            self.stats.completion_tokens += usage_out or _estimate(content)
        parsed = parse_model_json(content)
        by_id = {}
        if parsed is not None:
            for item in parsed:
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    by_id[item["id"]] = item
        sources = {item_id: text for item_id, text, _context, _key in pending}
        with self._lock_import:
            for item_id, _text, _context, key in pending:
                raw = by_id.get(item_id)
                source = sources[item_id]
                if kind in {"note", "support"}:
                    self.cache[key] = _valid_note_item(raw, source) if raw is not None else None
                elif kind == "profile":
                    self.cache[key] = _valid_profile_item(raw, source) if raw is not None else None
                else:
                    self.cache[key] = _valid_search_item(raw, source) if raw is not None else None
                if self.cache[key] is None:
                    self.stats.failures += 1


def _item_text(kind, item_id, text, context):
    return f"{kind}\n{item_id}\n{context}\n{text}"


def _estimate(text):
    return max(1, len(text) // 4)


def _clip(text):
    text = text or ""
    return text[:MAX_CHARS]


def _jobs(room, resolutions):
    jobs = []
    seen_notes = set()
    seen_search = set()
    for resolution in resolutions.values():
        for person in resolution.persons:
            for record_id in person.records:
                customer = room.customer_by_id.get(record_id) or {}
                email = (customer.get("email") or "").lower()
                for note in room.notes_by_email.get(email, []):
                    if note["note_id"] in seen_notes:
                        continue
                    seen_notes.add(note["note_id"])
                    kind = "note" if note.get("channel") == "kyc" else "support"
                    jobs.append((kind, note["note_id"], _clip(note.get("text") or ""), ""))
    from .hit import search_key
    for hit in room.hits:
        notes = (hit.get("profile_notes") or "").strip()
        first = re.split(r"\n\s*\n", notes)[0] if notes else ""
        if first and hit.get("entity_type") != "Company":
            jobs.append(("profile", hit["hit_id"], _clip(first), hit.get("name") or ""))
        if hit.get("entity_type") == "Company":
            continue
        context = f"{hit.get('name') or ''} | {hit.get('primary_category') or ''}"
        for result in room.search_by_query.get(search_key(hit.get("name") or ""), [])[:8]:
            if result["result_id"] in seen_search:
                continue
            seen_search.add(result["result_id"])
            snippet = result.get("snippet") or ""
            if snippet:
                jobs.append(("search", result["result_id"], _clip(snippet), context))
    return jobs
