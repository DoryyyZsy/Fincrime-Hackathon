import pytest

from src.route import route

THRESHOLDS = {"grey": 0.2, "likely": 0.9}


@pytest.mark.parametrize(("probability", "exact_id", "expected"), [
    (0.19, False, "auto_dismissed"), (0.20, False, "grey_zone"),
    (0.89, False, "grey_zone"), (0.90, False, "likely_true_hit"),
    (0.01, True, "likely_true_hit"),
])
def test_routing(probability, exact_id, expected):
    assert route({"match_probability": probability, "exact_id": exact_id}, THRESHOLDS) == expected


@pytest.mark.parametrize(("decision", "confidence", "expected"), [
    ("dismiss", "low", "human_review"), ("dismiss", "high", "auto_dismissed"),
    ("request_info", "high", "human_review"), ("human_review", "medium", "human_review"),
])
def test_grey_zone(decision, confidence, expected):
    pair = {"match_probability": 0.5, "exact_id": False}
    assert route(pair, THRESHOLDS, {"decision": decision, "confidence": confidence}) == expected
