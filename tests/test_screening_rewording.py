"""The unseen set rewords free text. These are paraphrases we wrote ourselves, not dev-set sentences."""
import pytest

from screening_ml import customer as CU
from screening_ml import hit as HI
from screening_ml import normalize as N
from screening_ml.compare import Check, REASON_CODES, _safety_net
from screening_ml.customer import Fact, Person

RESIDENCE = [
    ("From 2008 to 2018, the individual established their life and career inside the United Kingdom.", "United Kingdom"),
    ("The client spent 2011-2015 living in Norway.", "Norway"),
    ("Customer says they relocated to Côte d’Ivoire in 2012 and stayed until 2019.", "Cote d'Ivoire"),
    ("Between 2009 and 2017 the user was based out of Cote D'Ivoire.", "Cote d'Ivoire"),
    ("Per the customer, home was Brazil for most of 2003 through 2010.", "Brazil"),
    ("The account holder had professional and personal ties to Singapore from 2014 through 2018.", "Singapore"),
    ("Worked as a nurse in the Philippines, 1999-2004.", "Philippines"),
    ("They emigrated from Pakistan in 2015.", "Pakistan"),
    ("Address history shows a German address 2016-2020.", "Germany"),
]


@pytest.mark.parametrize("text,country", RESIDENCE)
def test_residence_notes_are_recognised_and_their_country_read(text, country):
    assert CU.RESIDENCE_WORDS.search(text) or CU.YEAR.search(text)
    assert country in N.countries_in_text(text)


KYC = [
    ("Veriff approved the passport of FERNÁNDEZ ISABEL (born 23/04/1970), issued in Spain, on 2023-06-17.",
     "2023-06-17", "FERNÁNDEZ ISABEL", (1970, 4, 23), "Spain"),
    ("Identity confirmed for ASAD ULLAH KHAN, year of birth 1966, using a United Kingdom driving licence.",
     "2024-10-03", "ASAD ULLAH KHAN", (1966, None, None), "United Kingdom"),
    ("On 2025-01-09 Persona matched JOSÉ ÁLVAREZ-RUIZ to a Mexican national ID; date of birth 2 March 1988.",
     "2025-01-09", "JOSÉ ÁLVAREZ-RUIZ", (1988, 3, 2), "Mexico"),
    ("Onfido check OK for O'BRIEN SEAN, DOB Sept 5, 1979, document: Irish passport.",
     "2024-02-02", "O'BRIEN SEAN", (1979, 9, 5), "Ireland"),
]


@pytest.mark.parametrize("text,created,name,dob,country", KYC)
def test_kyc_notes_in_new_wordings(text, created, name, dob, country):
    ident = CU._kyc_note_identity({"text": text, "created_at": created})
    assert ident["name"] == name
    assert ident["dob"] == dob
    assert country in ident["countries"]


AGES = [
    ("Formal charges were brought against Jane Doe, aged 45, in March 2019.", {1973, 1974}),
    ("In March 2019 the 45-year-old was charged.", {1973, 1974}),
    ("When he was arrested in June 2010 he was 33.", {1976, 1977}),
    ("Investigators questioned her in January 2025, when she reached the age of 38.", {1986, 1987}),
    ("A May 2002 notice listed Aleksandar Raković, 29, a politically exposed person in Serbia.", {1972, 1973}),
    ("John Smith was born on 17 July 1939.", {1939}),
    ("By October 2015 John Smith was 50 years old.", {1964, 1965}),
    ("John Smith (41) was sentenced in April 2008.", {1966, 1967}),
    ("At 52, John Smith was appointed minister in 2011.", {1958, 1959}),
]


@pytest.mark.parametrize("text,years", AGES)
def test_age_sentences(text, years):
    dobs, _ = HI.extract_ages(text)
    assert {d[0] for d in dobs} == years


def _person(**kw):
    p = Person(records=["CUS-000001"], primary="CUS-000001", entity_type="Individual")
    p.names = [Fact(["tyler", "harris"], ["CUS-000001"])]
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _checks(**fired):
    checks = {code: Check(False) for code in REASON_CODES}
    for code in fired:
        checks[code] = Check(True, ["X"], "fired")
    return checks


def test_safety_net_holds_location_clear_when_a_residence_note_is_unreadable():
    checks = _checks(LOCATION_MISMATCH=True)
    held, ev = _safety_net(_person(unread_notes=["NOTE-AAAAAA"]), checks)
    assert not checks["LOCATION_MISMATCH"].fired and held and ev == ["NOTE-AAAAAA"]


def test_safety_net_keeps_location_clear_when_everything_was_read():
    checks = _checks(LOCATION_MISMATCH=True)
    held, _ = _safety_net(_person(), checks)
    assert checks["LOCATION_MISMATCH"].fired and not held


def test_safety_net_holds_name_clear_from_a_note_name_unrelated_to_the_records():
    checks = _checks(NAME_MISMATCH_VERIFIED=True)
    person = _person(verified_name=Fact(["national", "identity"], ["NOTE-BBBBBB"]))
    _safety_net(person, checks)
    assert not checks["NAME_MISMATCH_VERIFIED"].fired


def test_safety_net_keeps_name_clear_when_note_name_shares_the_surname():
    checks = _checks(NAME_MISMATCH_VERIFIED=True)
    person = _person(verified_name=Fact(["amanda", "harris"], ["NOTE-CCCCCC"]))
    _safety_net(person, checks)
    assert checks["NAME_MISMATCH_VERIFIED"].fired
