"""extract_ages must read a hit's birth year from many wordings (the unseen set rewords free text) and must
never turn an unrelated date into a birth date: a false birth date can fake a DOB contradiction and clear a
real match (cost 60)."""
import pytest

from screening_ml.hit import extract_ages


def years(text):
    return sorted({d[0] for d in extract_ages(text)[0]})


@pytest.mark.parametrize("text, expected", [
    # ages at a dated event: two possible birth years
    ("Smith, 45, was sentenced in March 2019.", [1973, 1974]),
    ("In March 2019 a court sentenced Smith, then aged 45.", [1973, 1974]),
    ("Smith was 45 years of age when charged in March 2019.", [1973, 1974]),
    ("Smith, who was 45 at the time, was charged in March 2019.", [1973, 1974]),
    ("Questioned in March 2019 shortly after turning 45, Smith denied the claims.", [1973, 1974]),
    ("Smith had turned 45 by March 2019, the court heard.", [1973, 1974]),
    ("A 45-year-old named Smith was arrested in March 2019.", [1973, 1974]),
    ("Local reports from March 2019 say Smith was 45 at the time of the charge.", [1973, 1974]),
    ("When Smith was charged in March 2019, his age was 45.", [1973, 1974]),
    ("March 2019 marked the date Smith, then 45, was charged.", [1973, 1974]),
    ("Smith, at 45, was taken into custody in March 2019.", [1973, 1974]),
    ("A notice from a regulator in March 2019 included Smith, 45.", [1973, 1974]),
    # birth dates in any position
    ("Smith was born on 17 July 1939.", [1939]),
    ("In 1952, Smith was born.", [1952]),
    ("Smith came into the world on 17 July 1939.", [1939]),
    ("Records give Smith's date of birth as 17 July 1939.", [1939]),
    ("Smith (DOB 17 July 1939) is listed.", [1939]),
    ("Smith's birth date, 17 July 1939, appears in the filing.", [1939]),
    ("Smith entered the world on 17 July 1939.", [1939]),
    ("Records list 1939 as Smith's year of birth.", [1939]),
])
def test_reads_varied_wordings(text, expected):
    assert years(text) == expected


@pytest.mark.parametrize("text", [
    "Smith celebrated a birthday in Madrid in June 2010.",                  # birthday is not a birth date
    "Smith, born in Spain, was charged in 2010 and convicted in 2012.",       # birth word, two other dates
    "The company was registered in 2001.",                                    # no age, no birth word
    "Sales rose by a figure which was 45 percent higher in 2019.",            # number is not an age
    "Turnover turned 45 percent higher in 2019.",                             # 'turned 45 percent'
    "Smith, who was 3 times convicted, appeared in court in 2019.",           # count, not age
    "In 2019 the fine was 50.",                                               # not a person
    "In 2019 the court met at 10, then adjourned.",                           # a time, not an age
    "In 2019 the company employed 120 people, then 45 more.",                 # 'then 45 more'
    "Smith was arrested at 10 Main Street in 2019.",                          # an address
])
def test_does_not_invent_birth_dates(text):
    assert years(text) == []
