"""Turn each hit into (a) a numeric feature vector for the network and (b) the case object used for output.

Features are facts, not decisions: name/DOB/country agreement signals, data coverage, link type, etc.
Add new features at the end of FEATURE_NAMES and in `featurize`; the model reads the width from the data.
"""
import hashlib
import math
import os
import pickle
import time


from .compare import decide_case
from .customer import resolve
from .data import DataRoom
from .hit import build_hit

FEATURE_NAMES = [
    # customer resolution
    "link_account", "link_login", "link_email", "link_none", "n_candidates_log", "n_persons", "customer_found",
    "n_records_of_person", "n_approved_checks", "n_rejected_checks", "has_verified_name", "dob_verified",
    # entity type
    "cust_company", "hit_company", "type_mismatch",
    # names
    "name_agree", "name_frac", "name_matched", "name_sim", "name_usable", "verified_name_agree",
    "n_hit_names_log", "screened_name_sim",
    # dates of birth
    "cust_dob_known", "cust_dob_precision", "hit_dob_field", "hit_dob_notes", "hit_dob_search", "hit_dob_none",
    "n_hit_dobs", "dob_agree", "dob_contra", "year_gap_scaled",
    # time
    "has_dated_report", "age_at_report_scaled", "under_min_age", "hit_deceased", "deceased_before_screening",
    # location
    "n_cust_countries", "n_hit_countries", "country_overlap", "country_disjoint",
    # hit profile
    "hit_rich", "match_score", "cat_sanctions", "cat_pep", "cat_enforcement", "cat_adverse_media", "cat_other",
    "hits_in_alert_log",
]


def feature_values(case, alert, hit_raw):
    res = case.resolution
    comp = case.comparisons[0]
    p, h = comp.person, comp.hit
    chk = comp.checks
    cat = (h.category or "").lower()
    dob = p.dob.value if p.dob else None
    from . import normalize as N
    screened_parts = N.name_parts(res.screened.get("name"), res.screened.get("entity_type") == "Company")
    screened_sim = max([N.best_similarity(screened_parts, hp) for _, hp in h.names] + [0.0])
    v_agree = 0.0
    if p.verified_name:
        v_agree = float(any(N.name_agreement(p.verified_name.value, hp)[0] for _, hp in h.names))
    f = {
        "link_account": res.link_type == "account", "link_login": res.link_type == "login",
        "link_email": res.link_type == "email", "link_none": res.link_type == "none",
        "n_candidates_log": math.log1p(len(res.candidates)), "n_persons": min(len(res.persons), 5) / 5,
        "customer_found": bool(res.persons), "n_records_of_person": min(len(p.records), 5) / 5,
        "n_approved_checks": min(p.n_approved_checks, 5) / 5, "n_rejected_checks": min(p.n_rejected_checks, 5) / 5,
        "has_verified_name": p.verified_name is not None, "dob_verified": p.dob_verified,
        "cust_company": p.entity_type == "Company", "hit_company": h.entity_type == "Company",
        "type_mismatch": chk["ENTITY_TYPE_MISMATCH"].fired,
        "name_agree": comp.name_agree, "name_frac": comp.name_frac, "name_matched": min(comp.name_matched, 4) / 4,
        "name_sim": comp.name_sim, "name_usable": comp.name_usable, "verified_name_agree": v_agree,
        "n_hit_names_log": math.log1p(len(h.names)), "screened_name_sim": screened_sim,
        "cust_dob_known": dob is not None, "cust_dob_precision": (sum(x is not None for x in dob) / 3) if dob else 0.0,
        "hit_dob_field": h.dob_source == "field", "hit_dob_notes": h.dob_source == "notes",
        "hit_dob_search": h.dob_source == "search", "hit_dob_none": h.dob_source == "none",
        "n_hit_dobs": min(len(h.dobs), 4) / 4, "dob_agree": comp.dob_agree, "dob_contra": comp.dob_contra,
        "year_gap_scaled": min(comp.year_gap, 30) / 30,
        "has_dated_report": h.earliest_report is not None, "age_at_report_scaled": min(comp.age_at_report, 99) / 99,
        "under_min_age": chk["TEMPORAL_IMPOSSIBILITY"].fired, "hit_deceased": h.deceased is not None,
        "deceased_before_screening": chk["DECEASED"].fired,
        "n_cust_countries": min(len(p.countries), 6) / 6, "n_hit_countries": min(len(h.countries), 6) / 6,
        "country_overlap": bool(comp.country_overlap),
        "country_disjoint": bool(p.countries) and bool(h.countries) and not comp.country_overlap,
        "hit_rich": h.rich, "match_score": (hit_raw.get("match_score") or 0) / 100,
        "cat_sanctions": cat == "sanctions", "cat_pep": cat == "pep", "cat_enforcement": cat == "enforcement",
        "cat_adverse_media": cat == "adverse media",
        "cat_other": cat not in ("sanctions", "pep", "enforcement", "adverse media"),
        "hits_in_alert_log": math.log1p(len(alert.get("hit_ids") or [])),
    }
    return [float(f[name]) for name in FEATURE_NAMES]


def featurize(case, alert, hit_raw):
    import numpy as np
    return np.asarray(feature_values(case, alert, hit_raw), dtype=np.float32)


def build_cases(folder, cache_dir="artifacts/cache", use_cache=True, verbose=True):
    """Resolve every alert and hit in a data folder. Cached per folder contents (rebuild with use_cache=False)."""
    import numpy as np
    stamp = hashlib.sha1()
    for name in sorted(set(os.listdir(folder)) - {"practice_labels.jsonl"}):
        st = os.stat(os.path.join(folder, name))
        stamp.update(f"{name}:{st.st_size}:{int(st.st_mtime)}".encode())
    code_dir = os.path.dirname(__file__)
    for name in sorted(os.listdir(code_dir)):
        if name.endswith(".py"):
            stamp.update(open(os.path.join(code_dir, name), "rb").read())
    path = os.path.join(cache_dir, f"cases_{stamp.hexdigest()[:12]}.pkl")
    if use_cache and os.path.exists(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)

    t0 = time.time()
    room = DataRoom(folder)
    if verbose:
        print(f"loaded {folder} in {time.time() - t0:.1f}s: {len(room.alerts)} alerts, {len(room.hits)} hits, "
              f"{len(room.customers)} customers")
    cases, X = [], []
    for alert in room.alerts:
        resolution = resolve(room, alert)
        for hid in alert["hit_ids"]:
            hit_raw = room.hit_by_id[hid]
            case = decide_case(resolution, build_hit(room, hit_raw), alert)
            cases.append(case)
            X.append(featurize(case, alert, hit_raw))
    out = {"cases": cases, "X": np.stack(X), "feature_names": FEATURE_NAMES}
    if verbose:
        print(f"built {len(cases)} cases in {time.time() - t0:.1f}s")
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(out, handle)
    return out
