from functools import lru_cache
from threading import Lock

import pandas as pd

from src.clean import clean_record
from src.io import COLUMNS, PROCESSED, config, read_json
from src.match import BLOCKS, best_pairs, linker_for
from src.route import route

LOCK = Lock()


@lru_cache(maxsize=1)
def open_index(version):
    model = read_json("model")
    linker, db = linker_for(model, live=True)
    db.read_parquet(str(PROCESSED / "entities.parquet")).project(
        "record_id, entity_id, caption, source_url"
    ).create("public_entities")
    return model, linker, db


def live_screen(name, dob=None, nationality=None):
    row = clean_record(name, dob, nationality)
    if not row["name_clean"]:
        raise ValueError("Enter a name with at least one usable token.")
    if dob and row["dob_year"] is None:
        raise ValueError("Enter a valid date or year of birth.")
    if nationality and row["nationality"] is None:
        raise ValueError("Enter a recognized nationality or ISO country code.")
    row["record_id"] = "LIVE"
    record = pd.DataFrame([{key: row[key] for key in COLUMNS}])
    with LOCK:
        model, linker, db = open_index((PROCESSED / "model.json").stat().st_mtime_ns)
        pairs = linker.inference.find_matches_to_new_records(
            record, blocking_rules=BLOCKS, match_weight_threshold=-1000,
        ).as_record_dict()
        ids = {
            pair[side] for pair in pairs for side in ("record_id_l", "record_id_r")
            if pair[side] != "LIVE"
        }
        if not ids:
            return []
        entities = {rid: {"entity_id": eid, "caption": caption, "url": url}
                    for rid, eid, caption, url in db.execute(
                        "SELECT * FROM public_entities WHERE record_id IN (SELECT unnest(?))",
                        [sorted(ids)],
                    ).fetchall()}
        scored = best_pairs(pairs, entities, model["probability_two_random_records_match"])
    thresholds = config()
    return [{
        "Listed person": entities[row["entity_record_id"]]["caption"],
        "Identity score": row["match_probability"],
        "Route": route(row, thresholds).replace("_", " ").capitalize(),
    } for row in scored[:5]]
