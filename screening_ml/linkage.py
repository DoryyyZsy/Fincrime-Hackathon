"""Fellegi–Sunter probability that a customer and a watchlist hit are the same person.

The rule engine in compare.py still makes every decision. This module only
estimates a match probability from the unlabelled hits in the folder being
scored, then uses it to rank and annotate the review queue.

The probability is a sum of field weights, log m(level) / u(level). m is the
probability of a comparison level when the two records are the same entity, and
u is the probability when they are not. Both are estimated by
expectation-maximisation on the hits themselves. Nothing here reads the
practice labels. Initial values, the iteration cap, the tolerance and the order
of accumulation are fixed, so the same folder always produces the same scores,
including under a different PYTHONHASHSEED.

A clearance the rules did not already make is refused, and so is a confirmation
the rules did not already make. Promoting a review on this score would risk an
unsafe clearance (cost 60) to save a review (cost 2 or 3), and an unsupervised
score is not evidence enough to take that risk. PROMOTE stays false.

Gender is not a field in the bank files, so the type comparison is person
versus company. Name evidence is a learned comparison, not a single equality:
Jaro–Winkler, token overlap, a Soundex phonetic code, alias agreement, and how
rare the name is in the customer file.
"""
import json
import math
import os
import re

from . import normalize as N

# An unsafe clearance costs 60 and the most expensive review costs 3
# (data/screening/validate.py). Clearing beats reviewing only when the
# probability of being unsafe is below this ceiling. The score is not trusted
# that far, so the ceiling is reported and never applied.
UNSAFE_CLEAR_COST = 60
REVIEW_COST = 3
CLEAR_PROBABILITY_CEILING = REVIEW_COST / UNSAFE_CLEAR_COST

EM_ITERS = 20
EM_TOL = 1e-10
PRIOR_STRENGTH = 8.0
LAMBDA0 = 0.18
PROB_EPS = 1e-6
PROMOTE = False

# Field order is the accumulation order. Do not sort these at runtime.
FIELDS = ("name", "rarity", "dob", "country", "entity_type", "document", "alias", "time")

LEVELS = {
    "name": ("exact", "strong", "partial", "weak", "disagree", "missing"),
    "rarity": ("rare", "common", "unmatched"),
    "dob": ("full", "year_month", "year", "within_1", "conflict", "missing"),
    "country": ("agree", "disagree", "missing"),
    "entity_type": ("agree", "disagree", "missing"),
    "document": ("exact", "missing"),
    "alias": ("used", "unused", "missing"),
    "time": ("impossible", "ok", "missing"),
}

# Agreement-like levels must keep m >= u (+1) and contradiction levels m <= u (-1).
# EM may move the sizes. It may not flip what a level means.
ANCHORS = {
    "name": {"exact": 1, "disagree": -1},
    "rarity": {"rare": 1},
    "dob": {"full": 1, "year_month": 1, "year": 1, "within_1": 1, "conflict": -1},
    "country": {"agree": 1, "disagree": -1},
    "entity_type": {"disagree": -1},
    "document": {"exact": 1},
    "alias": {"used": 1},
    "time": {"impossible": -1},
}

# Fixed start. Each row sums to 1. Anchors hold. Screening hits already agree on
# a loose name, so a merely strong name is common in both classes and is not
# anchored.
M0 = {
    "name": {"exact": 0.40, "strong": 0.28, "partial": 0.14, "weak": 0.08, "disagree": 0.04, "missing": 0.06},
    "rarity": {"rare": 0.50, "common": 0.30, "unmatched": 0.20},
    "dob": {"full": 0.40, "year_month": 0.14, "year": 0.16, "within_1": 0.12, "conflict": 0.03, "missing": 0.15},
    "country": {"agree": 0.62, "disagree": 0.08, "missing": 0.30},
    "entity_type": {"agree": 0.90, "disagree": 0.02, "missing": 0.08},
    "document": {"exact": 0.10, "missing": 0.90},
    "alias": {"used": 0.22, "unused": 0.70, "missing": 0.08},
    "time": {"impossible": 0.02, "ok": 0.68, "missing": 0.30},
}
U0 = {
    "name": {"exact": 0.08, "strong": 0.40, "partial": 0.22, "weak": 0.14, "disagree": 0.12, "missing": 0.04},
    "rarity": {"rare": 0.12, "common": 0.48, "unmatched": 0.40},
    "dob": {"full": 0.02, "year_month": 0.02, "year": 0.04, "within_1": 0.06, "conflict": 0.46, "missing": 0.40},
    "country": {"agree": 0.22, "disagree": 0.48, "missing": 0.30},
    "entity_type": {"agree": 0.82, "disagree": 0.12, "missing": 0.06},
    "document": {"exact": 0.01, "missing": 0.99},
    "alias": {"used": 0.06, "unused": 0.86, "missing": 0.08},
    "time": {"impossible": 0.15, "ok": 0.55, "missing": 0.30},
}

_NAME_RANK = {"exact": 6, "strong": 5, "partial": 4, "weak": 3, "disagree": 2, "missing": 1}
_DOB_RANK = {"full": 5, "year_month": 4, "year": 3, "within_1": 2, "missing": 1, "conflict": 0}
_AGREEING_NAMES = {"exact", "strong", "partial"}

_SOUNDEX = {
    "B": "1", "F": "1", "P": "1", "V": "1",
    "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
    "D": "3", "T": "3",
    "L": "4",
    "M": "5", "N": "5",
    "R": "6",
}

MISSING_PATTERN = {
    "name": "missing",
    "rarity": "unmatched",
    "dob": "missing",
    "country": "missing",
    "entity_type": "missing",
    "document": "missing",
    "alias": "missing",
    "time": "missing",
}


class Fit:
    """One EM result. Distributions are plain dicts inserted in LEVELS order."""

    def __init__(self, lambda_, m, u, iterations, converged, n_pairs):
        self.lambda_ = lambda_
        self.m = m
        self.u = u
        self.iterations = iterations
        self.converged = converged
        self.n_pairs = n_pairs


def clear_probability_ceiling():
    """P(unsafe) below which a clearance would cost less than a review. Not applied."""
    return CLEAR_PROBABILITY_CEILING


def guarded_decision(rule_decision, proposed):
    """The decision that may be written for this hit.

    DIFFERENT_ENTITY is allowed only when the rules already cleared the hit.
    SAME_ENTITY is allowed only when the rules already confirmed it. With
    PROMOTE false, which is the submitted setting, every other proposal is
    refused as well: the score never moves a decision.
    """
    if proposed == "DIFFERENT_ENTITY" and rule_decision != "DIFFERENT_ENTITY":
        return rule_decision
    if proposed == "SAME_ENTITY" and rule_decision != "SAME_ENTITY":
        return rule_decision
    if not PROMOTE:
        return rule_decision
    return proposed


# ---------------------------------------------------------------- string features
def jaro(a, b):
    """Jaro similarity in [0, 1]. Identical strings, including two empty ones, score 1."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    len_a, len_b = len(a), len(b)
    match_dist = max(len_a, len_b) // 2 - 1
    if match_dist < 0:
        match_dist = 0
    a_match = [False] * len_a
    b_match = [False] * len_b
    matches = 0
    for i, ca in enumerate(a):
        start = i - match_dist
        if start < 0:
            start = 0
        end = i + match_dist + 1
        if end > len_b:
            end = len_b
        for j in range(start, end):
            if b_match[j] or ca != b[j]:
                continue
            a_match[i] = True
            b_match[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    transpositions = 0
    j = 0
    for i, ca in enumerate(a):
        if not a_match[i]:
            continue
        while not b_match[j]:
            j += 1
        if ca != b[j]:
            transpositions += 1
        j += 1
    transpositions /= 2.0
    return (matches / len_a + matches / len_b + (matches - transpositions) / matches) / 3.0


def jaro_winkler(a, b, prefix_scale=0.1, max_prefix=4):
    """Jaro–Winkler. The shared prefix boost is capped at four characters."""
    base = jaro(a, b)
    prefix = 0
    for ca, cb in zip(a, b):
        if ca != cb or prefix >= max_prefix:
            break
        prefix += 1
    return base + prefix * prefix_scale * (1.0 - base)


def soundex(token):
    """American Soundex, four characters. Empty when the token has no letters.

    H and W are transparent: they neither code nor separate a duplicate code.
    A vowel or Y separates one. A consonant with the same code as the previous
    coded letter, including the first letter, is dropped. That is why Pfister
    is P236 and not P123.
    """
    letters = [c for c in (token or "").upper() if "A" <= c <= "Z"]
    if not letters:
        return ""
    first = letters[0]
    prev = _SOUNDEX.get(first)
    out = []
    for ch in letters[1:]:
        if ch in "HW":
            continue
        digit = _SOUNDEX.get(ch)
        if digit is None:
            prev = None
            continue
        if digit != prev:
            out.append(digit)
            if len(out) == 3:
                break
        prev = digit
    return (first + "".join(out) + "000")[:4]


def token_jaccard(a, b):
    """Share of name parts that match under principle 8, over the union."""
    if not a or not b:
        return 0.0
    used = set()
    matched = 0
    for part in a:
        for j, other in enumerate(b):
            if j in used:
                continue
            if N.parts_match(part, other):
                used.add(j)
                matched += 1
                break
    union = len(a) + len(b) - matched
    if union <= 0:
        return 0.0
    return matched / union


def _family(parts):
    longs = [p for p in parts if len(p) >= 4]
    if longs:
        return longs[-1]
    return parts[-1] if parts else ""


def phonetic_match(a, b):
    """True when a long part on each side shares a Soundex code."""
    if not a or not b:
        return False
    fa, fb = _family(a), _family(b)
    if len(fa) >= 4 and len(fb) >= 4 and soundex(fa) and soundex(fa) == soundex(fb):
        return True
    codes = [soundex(p) for p in b if len(p) >= 4 and soundex(p)]
    for part in a:
        if len(part) >= 4 and soundex(part) in codes:
            return True
    return False


def _pair_name_level(cparts, hparts):
    if not cparts or not hparts:
        return "missing"
    if sorted(cparts) == sorted(hparts):
        return "exact"
    agree = N.name_agreement(cparts, hparts)[0]
    jw = jaro_winkler(" ".join(cparts), " ".join(hparts))
    jac = token_jaccard(cparts, hparts)
    phonetic = phonetic_match(cparts, hparts)
    if agree and (jw >= 0.92 or jac >= 0.67 or phonetic):
        return "strong"
    if agree or jw >= 0.86 or jac >= 0.5:
        return "partial"
    if jw >= 0.72 or phonetic:
        return "weak"
    return "disagree"


def name_level(cust_parts, hit_names):
    """Best name tier across the customer's names and the hit's names and aliases.

    `cust_parts` is a list of part lists. `hit_names` is a list of (display, parts).
    """
    if not cust_parts or not hit_names:
        return "missing"
    best = "missing"
    for cparts in cust_parts:
        for _display, hparts in hit_names:
            level = _pair_name_level(cparts, hparts)
            if _NAME_RANK[level] > _NAME_RANK[best]:
                best = level
    return best


def _alias_key(name):
    text = N._fold(name or "").lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def alias_level(cust_parts, hit_names, alias_keys):
    """`used` when an alias agrees and the primary name does not.

    The first hit name is the listed name, never counted as an alias, even if
    the same string is repeated in the alias list.
    """
    if not cust_parts or not hit_names:
        return "missing"
    primary, aliases = [], []
    for index, (display, parts) in enumerate(hit_names):
        if index > 0 and _alias_key(display) in alias_keys:
            aliases.append(parts)
        else:
            primary.append(parts)

    def agrees(group):
        return any(N.name_agreement(cparts, hparts)[0] for cparts in cust_parts for hparts in group)

    if aliases and agrees(aliases) and not agrees(primary):
        return "used"
    return "unused"


def _one_dob(cust, hit):
    cy, cm, cd = cust
    hy, hm, hd = hit
    gap = abs(cy - hy)
    if gap > 1:
        return "conflict"
    if cy == hy and cm and hm and cm != hm:
        return "conflict"
    if cy == hy and cm and hm and cm == hm and cd and hd and cd != hd:
        return "conflict"
    if cy == hy and cm and hm and cd and hd and cm == hm and cd == hd:
        return "full"
    if cy == hy and cm and hm and cm == hm:
        return "year_month"
    if cy == hy:
        return "year"
    return "within_1"


def dob_level(cust, hit_dobs):
    """Best birth-date agreement. One agreeing reading beats the others.

    A year alone agrees with a full date. A one-year gap is its own level: an
    age in a profile note produces two neighbouring years, and the policy does
    not treat that gap as a contradiction. Every reading must conflict before
    the level is `conflict`, matching an ambiguous slash date.
    """
    if not cust or cust[0] is None or not hit_dobs:
        return "missing"
    levels = [_one_dob(cust, hit) for hit in hit_dobs if hit and hit[0] is not None]
    if not levels:
        return "missing"
    if all(level == "conflict" for level in levels):
        return "conflict"
    agreeing = [level for level in levels if level != "conflict"]
    return max(agreeing, key=lambda level: _DOB_RANK[level])


def country_level(cust_countries, hit_countries):
    if not cust_countries or not hit_countries:
        return "missing"
    if set(cust_countries) & set(hit_countries):
        return "agree"
    return "disagree"


def entity_type_level(cust_type, hit_type):
    if not cust_type or not hit_type:
        return "missing"
    if cust_type == hit_type:
        return "agree"
    return "disagree"


def name_frequencies(customers):
    """How many customers carry each name part. Cutoff is rare at or below it.

    A part seen on at most max(3, n/500) customers is rare. On the development
    file that is about 0.2 percent. Counts are integers, so the order in which
    a customer's parts are visited does not matter.
    """
    counts = {}
    n = 0
    for customer in customers:
        n += 1
        company = customer.get("entity_type") == "Company"
        parts = set()
        for field in ("full_name", "preferred_name"):
            parts.update(N.name_parts(customer.get(field), company))
        for part in sorted(parts):
            counts[part] = counts.get(part, 0) + 1
    return counts, max(3, n // 500), n


def rarity_level(cust_parts, hit_parts, frequencies, cutoff, name):
    """Rare or common only when the names already agree. Otherwise unmatched."""
    if name not in _AGREEING_NAMES:
        return "unmatched"
    flat_hit = [part for parts in hit_parts for part in parts]
    if not flat_hit:
        return "unmatched"
    freqs = []
    for parts in cust_parts:
        for part in parts:
            if any(N.parts_match(part, other) for other in flat_hit):
                freqs.append(frequencies.get(part, 0))
    if not freqs:
        return "unmatched"
    if min(freqs) <= cutoff:
        return "rare"
    return "common"


def document_numbers(room, person):
    """Approved identity-document numbers for this person, longest first."""
    found = []
    seen = set()
    for rid in person.records or []:
        for app in room.applicants_by_customer.get(rid, []):
            for check in room.checks_by_applicant.get(app, []):
                if (check.get("result") or "").lower() != "approved":
                    continue
                number = re.sub(r"[^A-Z0-9]", "", (check.get("document_number") or "").upper())
                if len(number) >= 5 and number not in seen:
                    seen.add(number)
                    found.append(number)
    found.sort(key=lambda number: (-len(number), number))
    return found


def _collect_strings(value, out):
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, list):
        for item in value:
            _collect_strings(item, out)
    elif isinstance(value, dict):
        for key in sorted(value):
            _collect_strings(value[key], out)


def hit_blob(raw):
    """Letters and digits of the hit text a document number could be quoted in."""
    if not raw:
        return ""
    chunks = []
    for key in ("name", "profile_notes", "best_matching_name", "position_or_allegation"):
        _collect_strings(raw.get(key), chunks)
    _collect_strings(raw.get("aliases"), chunks)
    for key in ("identifiers", "documents", "ids", "id_numbers"):
        if key in raw:
            _collect_strings(raw.get(key), chunks)
    return re.sub(r"[^A-Z0-9]", "", " ".join(chunks).upper())


def document_level(numbers, blob):
    """`exact` when an approved document number is quoted on the hit, else missing.

    There is no `disagree` level. A hit rarely carries its own document number,
    so absence is unknown, not a contradiction.
    """
    if not numbers:
        return "missing"
    for number in numbers:
        if number in blob:
            return "exact"
    return "missing"


def time_level(comp):
    """`impossible` when the rules already found the customer could not be the hit."""
    checks = comp.checks
    if checks["TEMPORAL_IMPOSSIBILITY"].fired or checks["DECEASED"].fired:
        return "impossible"
    if comp.person.dob and (comp.hit.earliest_report or comp.hit.deceased):
        return "ok"
    return "missing"


def pattern_from_comparison(comp, frequencies, cutoff, alias_keys, numbers, blob):
    cust_parts = [list(fact.value) for fact in comp.person.names]
    if comp.person.verified_name:
        cust_parts.append(list(comp.person.verified_name.value))
    hit_names = [(display, list(parts)) for display, parts in comp.hit.names]
    name = name_level(cust_parts, hit_names)
    hit_parts = [parts for _display, parts in hit_names]
    return {
        "name": name,
        "rarity": rarity_level(cust_parts, hit_parts, frequencies, cutoff, name),
        "dob": dob_level(comp.person.dob.value if comp.person.dob else None, list(comp.hit.dobs)),
        "country": country_level(comp.person.countries, comp.hit.countries),
        "entity_type": entity_type_level(comp.person.entity_type, comp.hit.entity_type),
        "document": document_level(numbers, blob),
        "alias": alias_level(cust_parts, hit_names, alias_keys),
        "time": time_level(comp),
    }


def alias_keys_of(raw):
    keys = set()
    for alias in (raw or {}).get("aliases") or []:
        if isinstance(alias, str):
            keys.add(_alias_key(alias))
    return keys


# ---------------------------------------------------------------- EM
def _copy_dist(source):
    copied = {}
    for field in FIELDS:
        copied[field] = {level: source[field][level] for level in LEVELS[field]}
    return copied


def initial_fit(n_pairs=0):
    return Fit(LAMBDA0, _copy_dist(M0), _copy_dist(U0), 0, True, n_pairs)


def _clip_dist(dist):
    clipped = {level: max(dist[level], PROB_EPS) for level in dist}
    total = sum(clipped[level] for level in dist)
    return {level: clipped[level] / total for level in dist}


def _anchor(field, m, u):
    """If a level has flipped meaning, freeze that field on its initial weights.

    Replacing the whole field keeps both distributions summing to 1. A later
    iteration may estimate it again. It cannot publish a flipped sign, because
    this check runs after every M-step.
    """
    for level, sign in ANCHORS.get(field, {}).items():
        if sign * (m[field][level] - u[field][level]) < 0:
            m[field] = {lv: M0[field][lv] for lv in LEVELS[field]}
            u[field] = {lv: U0[field][lv] for lv in LEVELS[field]}
            return


def _log_level(dist, level):
    return math.log(max(dist[level], PROB_EPS))


def _posterior(log_match, log_unmatch):
    if log_match >= log_unmatch:
        return 1.0 / (1.0 + math.exp(log_unmatch - log_match))
    return math.exp(log_match - log_unmatch) / (1.0 + math.exp(log_match - log_unmatch))


def _pair_logs(fit, pattern):
    log_m = math.log(max(fit.lambda_, PROB_EPS))
    log_u = math.log(max(1.0 - fit.lambda_, PROB_EPS))
    for field in FIELDS:
        log_m += _log_level(fit.m[field], pattern[field])
        log_u += _log_level(fit.u[field], pattern[field])
    return log_m, log_u


def _delta(old, new):
    diff = abs(old.lambda_ - new.lambda_)
    for field in FIELDS:
        for level in LEVELS[field]:
            diff = max(diff, abs(old.m[field][level] - new.m[field][level]))
            diff = max(diff, abs(old.u[field][level] - new.u[field][level]))
    return diff


def fit(patterns):
    """EM for m, u and the match prior. `patterns` must already be in a fixed order.

    The caller sorts pairs before calling, so the weighted sums do not depend on
    hash randomisation or on the order the hits were resolved. Iteration stops at
    EM_TOL or at EM_ITERS, whichever comes first.
    """
    if not patterns:
        return initial_fit(0)
    current = initial_fit(len(patterns))
    converged = False
    last = 0
    for last in range(1, EM_ITERS + 1):
        posts = []
        for pattern in patterns:
            log_m, log_u = _pair_logs(current, pattern)
            posts.append(_posterior(log_m, log_u))
        m = {}
        u = {}
        for field in FIELDS:
            m_counts = {level: PRIOR_STRENGTH * M0[field][level] for level in LEVELS[field]}
            u_counts = {level: PRIOR_STRENGTH * U0[field][level] for level in LEVELS[field]}
            m_total = PRIOR_STRENGTH
            u_total = PRIOR_STRENGTH
            for post, pattern in zip(posts, patterns):
                level = pattern[field]
                m_counts[level] += post
                u_counts[level] += 1.0 - post
                m_total += post
                u_total += 1.0 - post
            m[field] = _clip_dist({level: m_counts[level] / m_total for level in LEVELS[field]})
            u[field] = _clip_dist({level: u_counts[level] / u_total for level in LEVELS[field]})
            _anchor(field, m, u)
        lam = (sum(posts) + PRIOR_STRENGTH * LAMBDA0) / (len(posts) + PRIOR_STRENGTH)
        lam = min(1.0 - PROB_EPS, max(PROB_EPS, lam))
        updated = Fit(lam, m, u, last, False, len(patterns))
        change = _delta(current, updated)
        current = updated
        if change < EM_TOL:
            converged = True
            break
    current.converged = converged
    current.iterations = last
    return current


def score_pattern(model, pattern):
    """Posterior P(same | levels), the total log2 weight, and each field's weight."""
    log_m, log_u = _pair_logs(model, pattern)
    probability = _posterior(log_m, log_u)
    weights = {}
    total = 0.0
    for field in FIELDS:
        m_level = max(model.m[field][pattern[field]], PROB_EPS)
        u_level = max(model.u[field][pattern[field]], PROB_EPS)
        weight = math.log(m_level / u_level) / math.log(2.0)
        weights[field] = weight
        total += weight
    return probability, total, weights


def _q(value, places):
    return float(format(value, f".{places}f"))


def model_summary(model):
    return {
        "iterations": model.iterations,
        "converged": bool(model.converged),
        "pairs": model.n_pairs,
        "lambda": _q(model.lambda_, 6),
        "promote": PROMOTE,
        "clear_probability_ceiling": CLEAR_PROBABILITY_CEILING,
        "m": {field: {level: _q(model.m[field][level], 6) for level in LEVELS[field]} for field in FIELDS},
        "u": {field: {level: _q(model.u[field][level], 6) for level in LEVELS[field]} for field in FIELDS},
    }


def _patterns_for(room, case, raw, frequencies, cutoff):
    """One pattern per candidate person, in person order. Missing case → one empty pattern."""
    hit_id = (case.hit_id if case is not None else (raw or {}).get("hit_id")) or ""
    if case is None or not case.comparisons:
        return [(hit_id, "", 0, dict(MISSING_PATTERN))]
    keys = alias_keys_of(raw)
    blob = hit_blob(raw)
    out = []
    for index, comp in enumerate(case.comparisons):
        numbers = document_numbers(room, comp.person)
        pattern = pattern_from_comparison(comp, frequencies, cutoff, keys, numbers, blob)
        primary = comp.person.primary or ""
        out.append((hit_id, primary, index, pattern))
    return out


def score_cases(room, cases, raw_hits):
    """Fit on these hits and return (scores, fit), one score per case, input order.

    Pairs are accumulated in (hit id, person id, index) order so the fitted
    weights do not depend on which loop built the cases. The reported score for
    a hit is the candidate person with the highest probability. Review ranks
    start at 1 for the review hit most likely to be the same person.
    """
    if len(cases) != len(raw_hits):
        raise ValueError("cases and raw hits must be the same length")
    frequencies, cutoff, _n = name_frequencies(room.customers)
    grouped = []
    flat = []
    for case, raw in zip(cases, raw_hits):
        group = _patterns_for(room, case, raw, frequencies, cutoff)
        grouped.append(group)
        flat.extend(group)
    flat.sort(key=lambda item: (item[0], item[1], item[2]))
    model = fit([item[3] for item in flat])
    scores = []
    for case, group in zip(cases, grouped):
        scored = []
        for hit_id, primary, index, pattern in group:
            probability, weight, field_weights = score_pattern(model, pattern)
            scored.append((probability, weight, primary, index, pattern, field_weights, hit_id))
        best = min(scored, key=lambda item: (-item[0], -item[1], item[2], item[3]))
        probability, weight, primary, _index, pattern, field_weights, hit_id = best
        rule = case.teacher_decision if case is not None else "REVIEW"
        scores.append({
            "hit_id": hit_id,
            "alert_id": case.alert_id if case is not None else (None),
            "decision": guarded_decision(rule, rule),
            "customer_record_id": case.customer_record_id if case is not None else None,
            "scored_record": primary or None,
            "match_probability": _q(probability, 6),
            "match_weight": _q(weight, 4),
            "review_rank": None,
            "levels": {field: pattern[field] for field in FIELDS},
            "field_weights": {field: _q(field_weights[field], 4) for field in FIELDS},
            "rule_veto": True,
        })
    review = []
    for index, score in enumerate(scores):
        if score["decision"] == "REVIEW":
            review.append((index, score["match_probability"], score["match_weight"], score["hit_id"]))
    review.sort(key=lambda item: (-item[1], -item[2], item[3]))
    for rank, (index, _probability, _weight, _hit_id) in enumerate(review, 1):
        scores[index]["review_rank"] = rank
    return scores, model


def scores_path(output_path):
    """Sidecar next to the decisions file: `dev.jsonl` → `dev.scores.jsonl`."""
    base, ext = os.path.splitext(output_path)
    if ext.lower() == ".jsonl":
        return base + ".scores.jsonl"
    return output_path + ".scores.jsonl"


def meta_path(output_path):
    base, ext = os.path.splitext(scores_path(output_path))
    return base + ".meta.json"


def attach_match_scores(rows, scores):
    """Copy the probability onto output rows only when SCREENING_MATCH_SCORE=1.

    validate.py ignores unknown fields. The default stays off so the submitted
    lines stay byte-identical to a rules-only run.
    """
    if os.environ.get("SCREENING_MATCH_SCORE") != "1":
        return rows
    if len(rows) != len(scores):
        return rows
    merged = []
    for row, score in zip(rows, scores):
        item = dict(row)
        item["match_probability"] = score["match_probability"]
        merged.append(item)
    return merged


def write_scores(output_path, scores, model):
    """Write the sidecar and the fitted m/u table. Returns the sidecar path."""
    path = scores_path(output_path)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for score in scores:
            handle.write(json.dumps(score, ensure_ascii=False) + "\n")
    with open(meta_path(output_path), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(model_summary(model), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path
