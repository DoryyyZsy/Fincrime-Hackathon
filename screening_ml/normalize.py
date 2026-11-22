"""Normalisers for the messy parts of the data: dates, countries and names (POLICY.md principle 8)."""
import calendar
import json
import re
import unicodedata
from calendar import monthrange
from functools import lru_cache

# ---------------------------------------------------------------- dates
MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
for _name, _num in list(MONTHS.items()):
    MONTHS[_name[:3]] = _num
MONTHS["sept"] = 9
MONTHS.update(zip(
    "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split(),
    range(1, 13),
))
_MON = r"(?:" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?"


def _slash_date(match):
    first, second, year = match.groups()
    if 1 <= int(first) <= 12 < int(second):
        return year, first, second
    return year, second, first


# Each pattern yields (year, month, day) with None for parts the text does not give.
_DATE_PATTERNS = [
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})(?=\b|T\d)"), lambda m: (m[1], m[2], m[3])),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"), _slash_date),  # day first if ambiguous
    (re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b"), lambda m: (m[3], m[2], m[1])),
    (re.compile(r"(?<!\d)(\d{4})年(\d{1,2})月(\d{1,2})日"), lambda m: (m[1], m[2], m[3])),
    (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)? (" + _MON + r"),? (\d{4})\b", re.I), lambda m: (m[3], m[2], m[1])),
    (re.compile(r"\b(" + _MON + r") (\d{1,2})(?:st|nd|rd|th)?,? (\d{4})\b", re.I), lambda m: (m[3], m[1], m[2])),
    (re.compile(r"\b(\d{4})-(\d{1,2})\b(?!-)"), lambda m: (m[1], m[2], None)),
    (re.compile(r"\b(" + _MON + r"),? (\d{4})\b", re.I), lambda m: (m[2], m[1], None)),
    (re.compile(r"\b(1[89]\d{2}|20\d{2})\b"), lambda m: (m[1], None, None)),
]


def _to_int(part):
    if part is None:
        return None
    if part.isdigit():
        return int(part)
    return MONTHS.get(part.lower().rstrip("."))


def _valid(y, m, d):
    return (y is not None and 1800 <= y <= 2100
            and (m is None or 1 <= m <= 12)
            and (d is None or (m is not None and 1 <= d <= monthrange(y, m)[1])))


def is_real_date(year, month, day):
    """True when the triple is a real calendar day. Unknown month or day is allowed."""
    if not _valid(year, month, day) or month is None or day is None:
        return _valid(year, month, day)
    return 1 <= day <= calendar.monthrange(year, month)[1]


def _calendar_ok(y, m, d):
    return is_real_date(y, m, d)


_SLASH = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")


def slash_readings(first, second, year):
    """Calendar-valid readings of a numeric slash date.

    Day-first and month-first are both returned when each is a real day and they
    differ. A single component above 12 leaves only one reading. Neither reading
    valid (31/02, 02/31) returns nothing: unknown stays unknown.
    """
    readings = []
    if _calendar_ok(year, second, first):  # day / month / year
        readings.append((year, second, first))
    if first != second and _calendar_ok(year, first, second):  # month / day / year
        readings.append((year, first, second))
    return readings


def parse_date(text):
    """Parse one date string into (year, month, day); missing parts stay None. Returns None if no date.

    Numeric slash dates in structured bank fields are day-first. Every unambiguous
    slash date on customer records and KYC checks in the development set has the
    day first (the first number exceeds 12 thousands of times; the second never
    does). Hit profiles are not written that way: use `date_readings` there.
    """
    if text is None:
        return None
    text = unicodedata.normalize("NFKC", str(text)).strip()
    for pattern, pick in _DATE_PATTERNS:
        m = pattern.search(text)
        if m:
            y, mo, d = (_to_int(p) for p in pick(m))
            return (y, mo, d) if _valid(y, mo, d) else None
    return None


def date_readings(text):
    """Every defensible reading of one date expression.

    An ambiguous `a/b/year` (both readings real calendar days, a != b) contributes
    both readings so a caller can require every reading to contradict before
    clearing, and can treat one agreeing reading as enough to refuse a clearance.
    A slash date with only one calendar-valid reading (6/20/1955, 20/06/1955)
    contributes that reading. A slash date with no real reading (31/02/1980)
    contributes nothing. Any other expression falls through to `parse_date`.
    """
    if text is None:
        return []
    text = str(text).strip()
    if not text:
        return []
    sm = _SLASH.search(text)
    if sm:
        readings = slash_readings(int(sm.group(1)), int(sm.group(2)), int(sm.group(3)))
        return readings
    one = parse_date(text)
    return [one] if one else []


def date_spans(text):
    """(date, start, end, raw) in left-to-right order. More specific patterns win overlaps.

    Slash dates keep the historical day-first reading so existing callers of
    `find_dates` do not change. Use `date_readings` where a hard day-first value
    would be unsafe.
    """
    found, taken = [], []
    text = unicodedata.normalize("NFKC", text or "")
    for pattern, pick in _DATE_PATTERNS:
        for match in pattern.finditer(text or ""):
            if any(match.start() < end and start < match.end() for start, end in taken):
                continue
            year, month, day = (_to_int(part) for part in pick(match))
            taken.append((match.start(), match.end()))
            if _valid(year, month, day):
                found.append(((year, month, day), match.start(), match.end(), text[match.start():match.end()]))
    found.sort(key=lambda item: item[1])
    return found


def find_dates(text, flexible_slash=False):
    """All dates in a free-text string, most specific patterns first, without overlaps.

    `flexible_slash` is for hit prose, where 6/20/1955 is June and 05/06/1955 could
    be either day. An ambiguous slash date is returned as a year only, so the month
    and day are not a hard value. The reference year of an age sentence is unchanged
    either way.
    """
    if not flexible_slash:
        return [date for date, _, _, _ in date_spans(text)]
    found, taken = [], []
    for pattern, pick in _DATE_PATTERNS:
        for match in pattern.finditer(text or ""):
            if any(match.start() < end and start < match.end() for start, end in taken):
                continue
            taken.append((match.start(), match.end()))
            if pattern.pattern.startswith(r"\b(\d{1,2})/"):
                readings = slash_readings(int(match.group(1)), int(match.group(2)), int(match.group(3)))
                if not readings:
                    continue
                date = readings[0] if len(readings) == 1 else (readings[0][0], None, None)
            else:
                year, month, day = (_to_int(part) for part in pick(match))
                if not _valid(year, month, day):
                    continue
                date = (year, month, day)
            found.append((date, match.start()))
    return [date for date, _ in sorted(found, key=lambda item: item[1])]


def date_key(date):
    """Sortable key, unknown parts sort as the earliest possible value."""
    return (date[0], date[1] or 0, date[2] or 0) if date else (0, 0, 0)


def fmt_date(date):
    if not date:
        return "unknown"
    y, m, d = date
    return f"{y:04d}" + (f"-{m:02d}" if m else "") + (f"-{d:02d}" if m and d else "")


# ---------------------------------------------------------------- countries
_CODES = {
    "AR": "Argentina", "AT": "Austria", "AU": "Australia", "BD": "Bangladesh", "BR": "Brazil", "CA": "Canada",
    "CI": "Cote d'Ivoire", "DE": "Germany", "DK": "Denmark", "ES": "Spain", "FI": "Finland", "FR": "France",
    "GB": "United Kingdom", "UK": "United Kingdom", "GH": "Ghana", "ID": "Indonesia", "IE": "Ireland", "IN": "India",
    "IT": "Italy", "KE": "Kenya", "MX": "Mexico", "NG": "Nigeria", "NL": "Netherlands", "NO": "Norway",
    "PE": "Peru", "PK": "Pakistan", "PL": "Poland", "PT": "Portugal", "RO": "Romania", "RS": "Serbia",
    "SE": "Sweden", "SG": "Singapore", "US": "United States", "USA": "United States", "ZA": "South Africa",
    "CH": "Switzerland", "BE": "Belgium", "CN": "China", "JP": "Japan", "KR": "South Korea", "RU": "Russia",
    "TR": "Turkey", "UA": "Ukraine", "AE": "United Arab Emirates", "NZ": "New Zealand", "CZ": "Czech Republic",
    "HU": "Hungary", "GR": "Greece", "EG": "Egypt", "MA": "Morocco", "PH": "Philippines", "VN": "Vietnam",
}
_ADJECTIVES = {
    "american": "United States", "australian": "Australia", "austrian": "Austria", "bangladeshi": "Bangladesh",
    "brazilian": "Brazil", "british": "United Kingdom", "canadian": "Canada", "danish": "Denmark",
    "dutch": "Netherlands", "french": "France", "german": "Germany", "ghanaian": "Ghana", "indian": "India",
    "indonesian": "Indonesia", "irish": "Ireland", "italian": "Italy", "kenyan": "Kenya", "mexican": "Mexico",
    "nigerian": "Nigeria", "norwegian": "Norway", "pakistani": "Pakistan", "polish": "Poland",
    "portuguese": "Portugal", "serbian": "Serbia", "singaporean": "Singapore", "south african": "South Africa",
    "spanish": "Spain", "swedish": "Sweden", "finnish": "Finland", "argentine": "Argentina",
    "argentinian": "Argentina", "romanian": "Romania", "peruvian": "Peru", "swiss": "Switzerland",
    "belgian": "Belgium", "chinese": "China", "japanese": "Japan", "russian": "Russia", "turkish": "Turkey",
    "ukrainian": "Ukraine", "egyptian": "Egypt", "moroccan": "Morocco", "filipino": "Philippines",
    "vietnamese": "Vietnam", "ivorian": "Cote d'Ivoire", "greek": "Greece", "czech": "Czech Republic",
    "hungarian": "Hungary", "emirati": "United Arab Emirates", "new zealander": "New Zealand",
}
_ALIASES = {
    "united states of america": "United States", "usa": "United States", "u.s.": "United States",
    "great britain": "United Kingdom", "england": "United Kingdom", "scotland": "United Kingdom",
    "wales": "United Kingdom", "northern ireland": "United Kingdom", "cote divoire": "Cote d'Ivoire",
    "ivory coast": "Cote d'Ivoire", "palestina": "Palestine", "state of palestine": "Palestine",
    "macao special administrative region of china": "Macao", "macau": "Macao", "czechia": "Czech Republic",
    "republic of korea": "South Korea", "korea, republic of": "South Korea", "russian federation": "Russia",
    "turkiye": "Turkey", "holland": "Netherlands", "the netherlands": "Netherlands", "viet nam": "Vietnam",
    "hong kong sar": "Hong Kong", "uae": "United Arab Emirates",
}
NOT_COUNTRIES = {"international", "europe", "none", "unknown", "n/a", "n a", "na", "global", "worldwide", ""}

# Every country name we know of; extended at load time with names seen in the data.
KNOWN_COUNTRIES = set(_CODES.values()) | set(_ADJECTIVES.values()) | set(_ALIASES.values()) | {
    "Afghanistan", "Albania", "Algeria", "Armenia", "Azerbaijan", "Bahrain", "Belarus", "Bosnia and Herzegovina",
    "British Virgin Islands", "Cameroon", "Colombia", "Costa Rica", "Croatia", "Dominican Republic", "Ecuador",
    "Eritrea", "Estonia", "Ethiopia", "Georgia", "Guatemala", "Haiti", "Iceland", "Iran", "Iraq", "Israel",
    "Jamaica", "Jersey", "Jordan", "Kosovo", "Kuwait", "Latvia", "Lebanon", "Liechtenstein", "Luxembourg",
    "Malaysia", "Moldova", "Montenegro", "Nepal", "Nicaragua", "North Korea", "Northern Mariana Islands", "Oman",
    "Panama", "Paraguay", "Puerto Rico", "Qatar", "Saudi Arabia", "Sierra Leone", "Slovenia", "Somalia", "Sudan",
    "Syria", "Thailand", "Trinidad and Tobago", "Uganda", "Uruguay", "Venezuela", "Hong Kong", "Lithuania",
    "Slovakia", "Bulgaria", "Cyprus", "Malta", "Tunisia", "Libya", "Yemen", "Sri Lanka", "Myanmar", "Cambodia",
    "Zimbabwe", "Zambia", "Tanzania", "Senegal", "Mali", "Niger", "Chad", "Angola", "Mozambique", "Bolivia",
    "Chile", "Cuba", "Honduras", "El Salvador", "Kazakhstan", "Uzbekistan", "Kyrgyzstan", "Tajikistan",
    "Turkmenistan", "Mongolia", "Taiwan", "Gibraltar", "Isle of Man", "Guernsey", "Cayman Islands", "Bermuda",
    "Monaco", "Andorra", "San Marino", "North Macedonia", "Benin", "Togo", "Liberia", "Guinea", "Rwanda",
    "Burundi", "Botswana", "Namibia", "Madagascar", "Mauritius", "Seychelles", "Maldives", "Bhutan", "Brunei",
    "Laos", "Fiji", "Papua New Guinea", "Belize", "Barbados", "Bahamas", "Guyana", "Suriname",
}


def _fold(text):
    text = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in text if not unicodedata.combining(c))


def _fold_name(text):
    chars = []
    latin = False
    for char in unicodedata.normalize("NFKD", str(text)):
        if unicodedata.category(char).startswith("M"):
            if not latin:
                chars.append(char)
        else:
            latin = unicodedata.name(char, "").startswith("LATIN ")
            chars.append(char)
    return unicodedata.normalize("NFC", "".join(chars).casefold())


def _letters(text):
    """Accent-free lowercase letters and single spaces; apostrophes vanish ("Cote D'Ivoire" -> "cote divoire")."""
    text = re.sub(r"['’`´]", "", _fold(text or "").lower())
    return re.sub(r" +", " ", re.sub(r"[^a-z]", " ", text)).strip()


def _country_key(text):
    return _letters(text)


_BY_KEY = {}


def _rebuild_index():
    _BY_KEY.clear()
    for name in KNOWN_COUNTRIES:
        _BY_KEY[_country_key(name)] = name
    for alias, name in _ALIASES.items():
        _BY_KEY[_country_key(alias)] = name
    for adj, name in _ADJECTIVES.items():
        _BY_KEY[_country_key(adj)] = name


_rebuild_index()


def register_countries(names):
    """Add country names seen in structured country fields so free text can find them too."""
    added = False
    for n in names:
        if n and canon_country(n) is None and _country_key(n) not in NOT_COUNTRIES and len(n) > 3 and "," not in n:
            KNOWN_COUNTRIES.add(n.strip())
            added = True
    if added:
        _rebuild_index()


def canon_country(value):
    """Map a code, name, adjective or 'City, Region, Country [Tag]' string to one canonical country name."""
    if value is None:
        return None
    text = re.sub(r"\s*\[.*?\]", "", str(value)).strip()
    if not text:
        return None
    if text.upper() in _CODES and len(text) <= 3 and text.isupper():
        return _CODES[text.upper()]
    key = _country_key(text)
    if key in NOT_COUNTRIES:
        return None
    if key in _BY_KEY:
        return _BY_KEY[key]
    if "," in text:  # address or place: the country is the last part
        return canon_country(text.split(",")[-1])
    return None


_COUNTRY_RE = None


def countries_in_text(text):
    """Country names or adjectives mentioned anywhere in a free-text string."""
    global _COUNTRY_RE
    if _COUNTRY_RE is None or _COUNTRY_RE[0] != len(_BY_KEY):
        keys = sorted((k for k in _BY_KEY if len(k) > 3), key=len, reverse=True)
        _COUNTRY_RE = (len(_BY_KEY), re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b"))
    return {_BY_KEY[m] for m in _COUNTRY_RE[1].findall(_letters(text))}  # same folding as the keys


# ---------------------------------------------------------------- names
PLACEHOLDERS = {"na", "n/a", "test", "unknown", "xxx", "xx", "none", "null", "nil", "tbd", "name", "noname"}
_VARIANTS = {}
LEGAL_FORMS = set()


def load_name_variants(path):
    """Load name_variants.json: groups of equivalent given names and company legal forms."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    groups = raw.get("given_name_groups") or raw.get("groups") or raw.get("name_groups") or []
    forms = raw.get("company_legal_forms") or raw.get("legal_forms") or []
    if isinstance(raw, dict) and not groups:  # tolerate other layouts: any list of lists is a group set
        for value in raw.values():
            if isinstance(value, list) and value and isinstance(value[0], list):
                groups = value
            elif isinstance(value, list) and value and isinstance(value[0], str):
                forms = value
    _VARIANTS.clear()
    for gid, group in enumerate(groups):
        for name in group:
            _VARIANTS.setdefault(_fold_name(name), set()).add(gid)
    LEGAL_FORMS.clear()
    LEGAL_FORMS.update(_fold_name(f).replace(".", "") for f in forms)


def name_parts(name, company=False):
    """Distinct comparable parts: no case, accents, punctuation, brackets, initials, placeholders, legal forms."""
    if not name:
        return []
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", str(name))
    text = _fold_name(text).replace("'", "").replace("’", "").replace(".", "")
    text = "".join(c if c.isalnum() or unicodedata.category(c).startswith("M") else " " for c in text)
    parts, seen = [], set()
    for part in text.split():
        if len(part) < 2 or part.isdigit() or part in PLACEHOLDERS:
            continue
        # letters+digits: noise in a person's name, but a real part of a company name ("N26 Bank", "3M")
        if not company and any(c.isdigit() for c in part):
            continue
        if company and part in LEGAL_FORMS:
            continue
        if part not in seen:
            seen.add(part)
            parts.append(part)
    return parts


def unspaced_name(parts):
    """A multi-character East Asian name can be written as one token."""
    return len(parts) == 1 and len(parts[0]) >= 2 and all(
        unicodedata.name(c, "").startswith(("CJK ", "HIRAGANA ", "KATAKANA ", "HANGUL "))
        for c in parts[0]
    )


def name_scripts(parts):
    return {unicodedata.name(c, "").split(" ", 1)[0] for part in parts for c in part if c.isalpha()}


# A limited Russian-to-Latin spelling for review referrals. Supplied aliases
# remain the primary evidence for equivalence across writing systems.
_RUSSIAN_LATIN = str.maketrans(dict(zip(
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
    ("a", "b", "v", "g", "d", "e", "yo", "zh", "z", "i", "y", "k", "l",
     "m", "n", "o", "p", "r", "s", "t", "u", "f", "kh", "ts", "ch",
     "sh", "shch", "", "y", "", "e", "yu", "ya"),
)))


def transliterated_name_agrees(left, right):
    """Exact multi-part spelling across Latin and Cyrillic, allowing token reordering."""
    if len(left) < 2 or len(left) != len(right):
        return False
    if {frozenset(name_scripts(left)), frozenset(name_scripts(right))} != {
            frozenset({"CYRILLIC"}), frozenset({"LATIN"})}:
        return False
    return sorted(p.translate(_RUSSIAN_LATIN) for p in left) == sorted(
        p.translate(_RUSSIAN_LATIN) for p in right)


@lru_cache(maxsize=500_000)
def _edit_distance(a, b):
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@lru_cache(maxsize=500_000)
def parts_match(a, b):
    """Two name parts match: identical, listed variants, or within the allowed edit distance."""
    if a == b:
        return True
    if _VARIANTS.get(a, set()) & _VARIANTS.get(b, set()):
        return True
    if unspaced_name([a]) or unspaced_name([b]):
        return False
    shorter = min(len(a), len(b))
    if shorter < 4:
        return False
    allowed = 1 if shorter <= 6 else 2
    if abs(len(a) - len(b)) > allowed:
        return False
    return _edit_distance(a, b) <= allowed


def part_similarity(a, b):
    """Continuous 0..1 similarity, used as an NN feature (not by the SOP rules)."""
    if parts_match(a, b):
        return 1.0
    return 1.0 - _edit_distance(a, b) / max(len(a), len(b))


def _match_count(short, long):
    """Greedy one-to-one matching; a part may also match two adjacent parts written apart."""
    used, matched = set(), 0
    for part in short:
        hit = next((j for j, other in enumerate(long) if j not in used and parts_match(part, other)), None)
        if hit is not None:
            used.add(hit)
            matched += 1
            continue
        for j in range(len(long) - 1):  # Abdulsalam vs Abdul Salam
            if j not in used and j + 1 not in used and parts_match(part, long[j] + long[j + 1]):
                used |= {j, j + 1}
                matched += 1
                break
    return matched


def name_agreement(a, b):
    """POLICY 8. Returns (agree, matched_parts, fraction_of_shorter_matched). a and b are part lists."""
    if not a or not b:
        return False, 0, 0.0
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    matched = _match_count(short, long)
    if len(a) == len(b):  # same length: either side may hold the joined form
        matched = max(matched, _match_count(long, short))
    matched = min(matched, len(short))
    frac = matched / len(short)
    agree = matched >= 2 or (matched == len(short) and matched >= 1)
    return agree, matched, frac


def best_similarity(a, b):
    """Mean best part similarity of the shorter name against the longer (NN feature)."""
    if not a or not b:
        return 0.0
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return sum(max(part_similarity(p, q) for q in long) for p in short) / len(short)
