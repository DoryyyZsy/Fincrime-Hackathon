"""The committed submission file follows the brief: one line per hit, reasons of two to four sentences."""
import json
import os
import re

import pytest

OUT = os.path.join(os.path.dirname(__file__), "..", "output", "dev.jsonl")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


@pytest.fixture(scope="module")
def rows():
    if not os.path.exists(OUT):
        pytest.skip("output/dev.jsonl not generated")
    with open(OUT, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_reasons_have_two_to_four_sentences(rows):
    bad = [(r["hit_id"], r["reason"]) for r in rows if not 2 <= len(SENT.split(r["reason"].strip())) <= 4]
    assert not bad, bad[:3]


def test_reasons_are_clean_text(rows):
    assert not [r["hit_id"] for r in rows if ".." in r["reason"] or "  " in r["reason"] or "None" in r["reason"]]


def test_every_line_has_the_required_fields(rows):
    need = {"hit_id", "alert_id", "decision", "reason_code", "customer_record_id", "reason", "evidence"}
    assert all(need <= set(r) for r in rows)
    assert len({r["hit_id"] for r in rows}) == len(rows)
