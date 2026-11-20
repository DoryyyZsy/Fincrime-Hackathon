from src.clean import clean_record


def test_title_suffix_and_particle():
    row = clean_record("Dr. Muhammad bin Rashid Jr.")
    assert row["name_clean"] == "muhammad bin rashid"
    assert row["name_no_particles"] == "muhammad rashid"


def test_order():
    assert clean_record("Wang Wei")["name_sorted"] == clean_record("Wei Wang")["name_sorted"]


def test_normalization():
    assert clean_record("Müller")["name_clean"] == "muller"
    assert clean_record("N/A", "1900-01-01")["name_clean"] is None
    assert clean_record("N/A", "1900-01-01")["dob_year"] is None
    assert clean_record("Name", "1980", "GBR", "ab- 12")["dob_year"] == 1980
    assert clean_record("Name", "1980", "GBR", "ab- 12")["nationality"] == ["gb"]
    assert clean_record("Name", "1980", "GBR", "ab- 12")["id_number"] == "AB12"
