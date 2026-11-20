"""Shared identity normalization. Preserve raw fields at the call site."""
import re
import unicodedata
from datetime import date

import jellyfish
import pycountry
from unidecode import unidecode

JUNK = {"", "n/a", "unknown", "none", "null", "-"}
TITLES = set("mr mrs ms miss dr prof sir sheikh general col haji".split())
SUFFIXES = set("jr sr ii iii iv".split())
PARTICLES = set("al el bin ibn bint abu van von de da".split())


def value(raw):
    if raw is None:
        return None
    cleaned = unicodedata.normalize("NFKC", str(raw)).strip()
    return None if cleaned.casefold() in JUNK else cleaned


def date_parts(raw):
    normalized = value(raw)
    if not normalized or normalized in {"1900-01-01", "1970-01-01"}:
        return None, None
    try:
        if re.fullmatch(r"\d{4}", normalized):
            return None, date(int(normalized), 1, 1).year
        parsed = date.fromisoformat(normalized)
        return parsed.isoformat(), parsed.year
    except ValueError:
        return None, None


def country_codes(raw):
    parts = raw if isinstance(raw, (list, tuple)) else str(raw or "").split(";")
    result = set()
    for part in parts:
        normalized = value(part)
        if normalized:
            try:
                result.add(pycountry.countries.lookup(normalized).alpha_2.lower())
            except LookupError:
                continue
    return sorted(result) or None


def clean_record(name, dob=None, nationality=None, id_number=None):
    tokens = [
        token for token in re.findall(r"[a-z]+", unidecode(value(name) or "").lower())
        if token not in TITLES and token not in SUFFIXES
    ]
    core = sorted(token for token in tokens if token not in PARTICLES)
    phonemes = sorted(filter(None, map(jellyfish.metaphone, core)))
    full_date, year = date_parts(dob)
    identifier = re.sub(r"[\s-]+", "", (value(id_number) or "").upper()) or None
    return {
        "name_clean": " ".join(tokens) or None,
        "name_sorted": " ".join(sorted(tokens)) or None,
        "name_tokens": sorted(set(tokens)) or None,
        "name_no_particles": " ".join(core) or None,
        "name_metaphone": " ".join(phonemes) or None,
        "first_token": min(tokens, default=None),
        "dob": full_date,
        "dob_year": year,
        "nationality": country_codes(nationality),
        "id_number": identifier,
    }
