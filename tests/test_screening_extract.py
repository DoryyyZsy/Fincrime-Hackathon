"""Stdlib checks for the free-text extractors. No development data."""
import unittest

from screening_ml.compare import compare, teacher_code
from screening_ml.customer import Fact, Person, _dob_from_kyc_text, kyc_note_failed
from screening_ml.hit import HitFacts, extract_ages
from screening_ml.normalize import date_readings, find_dates, parse_date


class SlashDates(unittest.TestCase):
    def test_day_first_structured_fields_stay_day_first(self):
        self.assertEqual(parse_date("10/06/1962"), (1962, 6, 10))
        self.assertEqual(parse_date("20/06/1955"), (1955, 6, 20))

    def test_ambiguous_slash_has_both_readings(self):
        self.assertEqual(date_readings("05/06/1980"), [(1980, 6, 5), (1980, 5, 6)])

    def test_unambiguous_slash_picks_the_only_real_day(self):
        self.assertEqual(date_readings("6/20/1955"), [(1955, 6, 20)])
        self.assertEqual(date_readings("20/06/1955"), [(1955, 6, 20)])
        self.assertEqual(date_readings("11/11/1980"), [(1980, 11, 11)])

    def test_impossible_slash_is_unknown(self):
        self.assertEqual(date_readings("31/02/1980"), [])
        self.assertEqual(date_readings("02/31/1980"), [])

    def test_prose_date_is_unchanged(self):
        self.assertEqual(date_readings("born on 17 July 1939"), [(1939, 7, 17)])
        self.assertEqual(date_readings("1952"), [(1952, None, None)])

    def test_flexible_finder_keeps_the_year_of_an_ambiguous_slash(self):
        self.assertEqual(find_dates("on 05/06/2019", flexible_slash=True), [(2019, None, None)])
        self.assertEqual(find_dates("on 6/20/2019", flexible_slash=True), [(2019, 6, 20)])


class KycNotes(unittest.TestCase):
    def test_dob_cue_beats_the_day_the_check_ran(self):
        text = (
            "Onfido verified the passport for KARIM KHAN on 2024-07-12; "
            "subject date of birth is December 5, 1992."
        )
        self.assertEqual(_dob_from_kyc_text(text, (2024, 7, 12), (2026, 11, 15)), (1992, 12, 5))
        # Shifting the note's created_at must not make the check date look like a birth date.
        self.assertEqual(_dob_from_kyc_text(text, (2024, 7, 14), (2026, 11, 15)), (1992, 12, 5))

    def test_parenthetical_dob_and_year_only_dob(self):
        paren = "JAVIER BIANCHI (19/12/2003) from Germany was verified via Veriff on 2024-12-17."
        self.assertEqual(_dob_from_kyc_text(paren, (2024, 12, 17), (2026, 11, 15)), (2003, 12, 19))
        year = "Sumsub verified the passport for ABDUL RAHMAN, DOB 1999, on 2025-10-17."
        self.assertEqual(_dob_from_kyc_text(year, (2025, 10, 17), (2026, 11, 15)), (1999, None, None))

    def test_born_in_country_on_date(self):
        text = "The subject was born in France on 14 February 1953. Verified on 2025-10-23."
        self.assertEqual(_dob_from_kyc_text(text, (2025, 10, 23), (2026, 11, 15)), (1953, 2, 14))

    def test_impossible_and_conflicting_dobs_are_unknown(self):
        future = "DOB 2027-01-01 confirmed by Sumsub on 2024-01-01."
        self.assertIsNone(_dob_from_kyc_text(future, (2024, 1, 1), (2026, 11, 15)))
        ancient = "DOB 1801-01-01 confirmed by Sumsub on 2024-01-01."
        self.assertIsNone(_dob_from_kyc_text(ancient, (2024, 1, 1), (2026, 11, 15)))
        conflict = "DOB 14 May 1969 and also born 2 June 1970, verified on 2024-07-20."
        self.assertIsNone(_dob_from_kyc_text(conflict, (2024, 7, 20), (2026, 11, 15)))

    def test_rejection_phrases_are_not_evidence(self):
        passed = "Onfido verified the passport of NILS MARTIN, born December 13, 1977, on 2023-02-19."
        self.assertFalse(kyc_note_failed(passed))
        for phrase in (
            "Identity was not verified; the applicant did not pass and must resubmit.",
            "The check was unsuccessful.",
            "Verification failed and the customer was declined.",
            "Documents were rejected as expired.",
            "The applicant could not be matched and should resubmit.",
            "Outcome pending.",
        ):
            self.assertTrue(kyc_note_failed(phrase + " " + passed), phrase)
        self.assertTrue(kyc_note_failed("A filing that names the customer and gives no outcome."))

    def test_document_dated_without_a_birth_cue_is_unknown(self):
        text = (
            "RASEL KHAN was successfully verified using an Onfido check on 2024-06-04, "
            "which authenticated a Bangladesh national ID dated 1952-01-21."
        )
        self.assertIsNone(_dob_from_kyc_text(text, (2024, 6, 4), (2026, 11, 15)))


class UnreadableIdentity(unittest.TestCase):
    def test_a_discarded_kyc_note_does_not_clear_on_country_or_declared_dob(self):
        person = Person(
            records=["CUS-1"], primary="CUS-1", entity_type="Individual",
            names=[Fact(["jane", "doe"], ["CUS-1"])],
            countries={"Nigeria": ["CUS-1"]},
            identity_unreadable=True,
        )
        hit = HitFacts(
            hit_id="H-000001", entity_type="Individual",
            names=[("Jane Doe", ["jane", "doe"])],
            countries={"Pakistan"},
        )
        comp = compare(person, hit, "2026-11-01")
        self.assertFalse(comp.checks["LOCATION_MISMATCH"].fired)
        self.assertEqual(teacher_code(comp), "INSUFFICIENT_DATA")

        readable = Person(
            records=["CUS-1"], primary="CUS-1", entity_type="Individual",
            names=[Fact(["jane", "doe"], ["CUS-1"])],
            countries={"Nigeria": ["CUS-1"]},
        )
        cleared = compare(readable, hit, "2026-11-01")
        self.assertTrue(cleared.checks["LOCATION_MISMATCH"].fired)
        self.assertEqual(teacher_code(cleared), "LOCATION_MISMATCH")


class Ages(unittest.TestCase):
    def test_development_wordings(self):
        samples = {
            "Formal charges were brought against her, aged 30, in November 1998.": [(1967, None, None), (1968, None, None)],
            "At the time he faced charges in March 1999, he was 24.": [(1974, None, None), (1975, None, None)],
            "He was born in 1952.": [(1952, None, None)],
            "He was born on 17 July 1939.": [(1939, 7, 17)],
            "Questioned in January 2025, when he reached the age of 38.": [(1986, None, None), (1987, None, None)],
        }
        for text, expected in samples.items():
            got, _ = extract_ages(text)
            self.assertEqual(got, expected, text)

    def test_paraphrases_keep_the_same_years(self):
        pairs = [
            ("She was aged 46 in January 2023.", "She was age 46 in January 2023."),
            ("In January 2025, investigators questioned the 38-year-old man.",
             "In January 2025, investigators questioned the 38 year old man."),
            ("He was 32 when arrested during September 2001.",
             "He was age 32 when arrested during September 2001."),
            ("Back then she reached the age of 23 in September 2001.",
             "Back then she reached age 23 in September 2001."),
            ("He turned up, aged 30 years old, in February 2011.",
             "He turned up, age 30 years of age, in February 2011."),
        ]
        for original, rewritten in pairs:
            self.assertEqual(extract_ages(original)[0], extract_ages(rewritten)[0], rewritten)

    def test_month_first_reference_date_still_yields_a_year(self):
        got, _ = extract_ages("He was age 45 on 6/20/2019 when sentenced.")
        self.assertEqual(got, [(1973, None, None), (1974, None, None)])

    def test_ambiguous_born_date_keeps_both_readings(self):
        got, _ = extract_ages("She was born on 05/06/1980 in the capital.")
        self.assertEqual(got, [(1980, 6, 5), (1980, 5, 6)])


if __name__ == "__main__":
    unittest.main()
