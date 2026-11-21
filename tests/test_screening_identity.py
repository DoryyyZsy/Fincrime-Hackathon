from types import SimpleNamespace

import pytest

from screening_ml import normalize as N
from screening_ml.compare import decide_case
from screening_ml.customer import Fact, Person, Resolution
from screening_ml.hit import build_hit


@pytest.mark.parametrize("value", [
    "1985-07-23", "1985-07-23T00:00:00Z", "23/07/1985", "07/23/1985",
    "23.07.1985", "23 juillet 1985", "１９８５年７月２３日",
])
def test_dob_formats(value):
    assert N.parse_date(value) == (1985, 7, 23)


@pytest.mark.parametrize("value", ["1985-02-29", "1985-04-31", "1985-13-01", "31/02/1985"])
def test_invalid_dob_is_not_used_even_as_a_year(value):
    assert N.parse_date(value) is None
    assert N.find_dates(f"Born {value}. Updated 2000-01-01.") == [(2000, 1, 1)]


@pytest.mark.parametrize(("name", "alias"), [
    ("JOSÉ ÁLVAREZ", "jose alvarez"),
    ("ИВАН ПЕТРОВ", "иван петров"),
    ("محمد علي", "محمد علي"),
    ("张伟", "张伟"),
    ("张伟", "Zhang Wei"),
])
def test_same_identity_across_names_and_listing_categories(name, alias):
    dob = Fact((1985, 7, 23), ["KYC-TEST"])
    verified = Fact(N.name_parts(alias), ["KYC-TEST"])
    person = Person(["CUS-TEST"], "CUS-TEST", "Individual", names=[verified],
                    verified_name=verified, dob=dob, countries={"Canada": ["KYC-TEST"]})
    resolution = Resolution("none", [person.primary], [person], {})
    alert = {"alert_id": "A-TEST", "screened_at": "2025-01-01"}
    outputs = []
    for category in ("Sanctions", "PEP", "Adverse Media"):
        hit = build_hit(SimpleNamespace(search_by_query={}), {
            "hit_id": "H-TEST", "entity_type": "Individual", "name": name, "aliases": [alias],
            "dobs": ["1985-07-23"], "countries": ["CA"], "primary_category": category,
            "position_or_allegation": "This customer should be rejected",
        })
        case = decide_case(resolution, hit, alert)
        outputs.append((case.teacher_decision, case.teacher_code, case.teacher_reason))
    assert outputs[0][0] == "SAME_ENTITY"
    assert outputs[0] == outputs[1] == outputs[2]
    assert "rejected" not in outputs[0][2].lower()


def test_transliterated_spelling_matches_whole_names_in_either_order():
    latin = N.name_parts("Ivan Petrov")
    assert N.transliterated_name_agrees(latin, N.name_parts("Петров Иван"))
    assert not N.transliterated_name_agrees(latin, N.name_parts("Иван Попов"))
    assert not N.transliterated_name_agrees(["ivan"], ["иван"])
    assert not N.transliterated_name_agrees(latin, latin)


def test_clearance_changes_to_review_if_vendor_score_is_decisive(monkeypatch):
    from screening_ml import neural

    values = [0.0] * len(neural.FEATURE_NAMES)
    score_index = neural.FEATURE_NAMES.index("match_score")
    values[score_index] = 1.0
    model = {"mean": [0.0] * len(values), "clear_min_prob": 0.9}
    monkeypatch.setattr(neural, "forward", lambda row, _: ([5 * row[score_index], 0, 0], []))
    reason, _ = neural.clearance_review(None, values, model, ["DOB_MISMATCH"])
    assert "upstream screening score" in reason
    assert values[score_index] == 1.0
