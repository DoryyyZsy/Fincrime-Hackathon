from collections import defaultdict
from math import ceil, floor

import duckdb
from rapidfuzz import fuzz, process

from src.io import PROCESSED, config, write_json


def baseline():
    threshold = config()["baseline_threshold"]
    ratio = threshold / (200 - threshold)
    blocks = defaultdict(list)
    with duckdb.connect() as db:
        entities = db.read_parquet(str(PROCESSED / "entities.parquet"))
        customers = db.read_parquet(str(PROCESSED / "customers.parquet"))
        for entity_id, name, initial, length in entities.query("rows", """
            SELECT DISTINCT entity_id, name_sorted, left(name_sorted, 1), length(name_sorted)
            FROM rows WHERE name_sorted IS NOT NULL
        """).fetchall():
            blocks[initial, length].append((entity_id, name))
        hits = []
        for cid, name in customers.project("customer_id, name_sorted").fetchall():
            if not name:
                continue
            best = {}
            for length in range(ceil(ratio * len(name)), floor(len(name) / ratio) + 1):
                rows = blocks[name[0], length]
                for _, score, index in process.extract_iter(
                    name, [pair[1] for pair in rows], scorer=fuzz.token_sort_ratio,
                    score_cutoff=threshold,
                ):
                    eid = rows[index][0]
                    best[eid] = max(score, best.get(eid, 0))
            hits.extend({"customer_id": cid, "entity_id": eid, "score": score}
                        for eid, score in sorted(best.items()))
    write_json("baseline", hits)
    print(f"Baseline raised {len(hits):,} customer–entity alerts")


if __name__ == "__main__":
    baseline()
