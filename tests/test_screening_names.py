"""Name rules from POLICY.md principle 8 (screening_ml.normalize)."""
import os

import pytest

from screening_ml import normalize as N

VARIANTS = os.path.join(os.path.dirname(__file__), "..", "data", "screening", "dev", "name_variants.json")


@pytest.fixture(autouse=True, scope="module")
def variants():
    if os.path.exists(VARIANTS):
        N.load_name_variants(VARIANTS)
    else:  # the data folder is git-ignored; fall back to the forms these tests need
        N._VARIANTS.clear()
        N._VARIANTS.update({"bob": {0}, "robert": {0}})
        N.LEGAL_FORMS.clear()
        N.LEGAL_FORMS.update({"se", "as", "gmbh", "ltd"})


def agree(a, b, company=False):
    return N.name_agreement(N.name_parts(a, company), N.name_parts(b, company))[0]


def test_company_alphanumeric_part_is_kept():
    assert N.name_parts("N26 Bank SE", company=True) == ["n26", "bank"]


def test_different_banks_do_not_agree_on_the_word_bank():
    assert not agree("N26 Bank SE", "Danske Bank A/S", company=True)


def test_same_company_with_other_legal_form_agrees():
    assert agree("N26 Bank SE", "N26 Bank GmbH", company=True)


def test_person_digits_and_placeholders_are_not_parts():
    assert N.name_parts("John 123 Smith2 N/A") == ["john"]


def test_variants_typos_and_joined_parts():
    assert agree("Bob Smith", "Robert Smith")
    assert agree("Elvira Velasco Mooillo", "Elvira Velasco Morillo")      # one edit
    assert agree("Abdulsalam Bello", "Abdul Salam Bello")                  # written together or apart
    assert not agree("Daniel Sargsyan", "Davit Vahan Sargsyan")            # only one part matches
