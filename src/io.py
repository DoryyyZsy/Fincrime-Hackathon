import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"
RAW = ROOT / "data/raw"
COLUMNS = [
    "record_id", "name_sorted", "name_tokens", "name_no_particles",
    "name_metaphone", "first_token", "dob", "dob_year", "nationality", "id_number",
]


def config():
    # JSON syntax is also valid YAML 1.2.
    return json.loads((ROOT / "config.yaml").read_text(encoding="utf-8"))


def read_rows(name):
    with duckdb.connect() as db:
        query = db.read_parquet(str(PROCESSED / f"{name}.parquet"))
        keys = query.columns
        return [dict(zip(keys, row)) for row in query.fetchall()]


def write_rows(name, rows):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / f"{name}.parquet"
    frame = pd.DataFrame(rows)
    if "dob_year" in frame:
        frame["dob_year"] = frame["dob_year"].astype("Int64")
    with duckdb.connect() as db:
        db.from_df(frame).write_parquet(str(path))


def read_json(name, default=None):
    path = PROCESSED / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(name, payload):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=PROCESSED, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
    os.replace(handle.name, PROCESSED / f"{name}.json")
