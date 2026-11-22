"""Model-extractor tests. No network and no API key: a fake transport stands in."""
import json
import os
import unittest
import urllib.error
from collections import defaultdict

from screening_ml.compare import compare, teacher_code
from screening_ml.customer import Person, build_person
from screening_ml.hit import HitFacts, build_hit
from screening_ml.llm_extract import (
    MODEL,
    ExtractionBook,
    LlmClient,
    NoteFacts,
    ProfileFacts,
    SearchFacts,
    cache_key,
    merge_kyc_dob,
    merge_year_dobs,
    note_is_evidence,
    parse_model_json,
    quoted_countries,
    reconcile,
    verbatim,
)


class Quotes(unittest.TestCase):
    def test_verbatim_allows_only_whitespace_differences(self):
        self.assertTrue(verbatim("aged 45 in March 2019", "He was aged 45 in March 2019."))
        self.assertTrue(verbatim("aged  45\nin March 2019", "He was aged 45 in March 2019."))
        self.assertFalse(verbatim("aged 46 in March 2019", "He was aged 45 in March 2019."))
        self.assertFalse(verbatim("", "text"))

    def test_country_quote_must_name_the_country_and_sit_in_the_source(self):
        source = "She lived in France after 2010."
        self.assertEqual(quoted_countries([{"name": "France", "quote": "lived in France"}], source), ["France"])
        self.assertEqual(quoted_countries([{"name": "France", "quote": "lived in Spain"}], source), [])
        self.assertEqual(quoted_countries([{"name": "France", "quote": "not in the note"}], source), [])


class Dates(unittest.TestCase):
    def test_model_year_must_follow_the_quote(self):
        source = "He was aged 45 when sentenced in March 2019."
        quote = "aged 45 when sentenced in March 2019"
        agreed, only, conflict = merge_year_dobs([], [1973, 1974], quote, source)
        self.assertEqual(agreed, [(1973, None, None), (1974, None, None)])
        self.assertTrue(only)
        self.assertFalse(conflict)
        # The quote cannot support 1900, so the model fact is dropped.
        kept, only, conflict = merge_year_dobs([(1973, None, None)], [1900], quote, source)
        self.assertEqual(kept, [(1973, None, None)])
        self.assertFalse(conflict)

    def test_disagreement_drops_the_date(self):
        source = "born in 1980"
        merged, only, conflict = merge_year_dobs([(1974, None, None)], [1980], "born in 1980", source)
        self.assertEqual(merged, [])
        self.assertTrue(conflict)
        self.assertFalse(only)

    def test_regex_stands_when_the_model_is_silent(self):
        kept, only, conflict = merge_kyc_dob((1977, 1, 11), None, "", "DOB 1977-01-11")
        self.assertEqual(kept, (1977, 1, 11))
        self.assertFalse(only)
        self.assertFalse(conflict)

    def test_kyc_note_is_evidence_only_when_both_allow_it(self):
        self.assertTrue(note_is_evidence(True, "passed"))
        self.assertTrue(note_is_evidence(True, None))
        self.assertFalse(note_is_evidence(True, "unclear"))
        self.assertFalse(note_is_evidence(True, "rejected"))
        self.assertFalse(note_is_evidence(False, "passed"))


class Reconcile(unittest.TestCase):
    def _case(self, decision, code):
        return type("Case", (), {"teacher_decision": decision, "teacher_code": code})()

    def test_a_new_clearance_needs_a_quoted_birth_date(self):
        regex = self._case("REVIEW", "INSUFFICIENT_DATA")
        cleared = self._case("DIFFERENT_ENTITY", "DOB_MISMATCH")
        hit = type("Hit", (), {"llm_dob_only": True, "dob_withheld": False})()
        resolution = type("Res", (), {"persons": []})()
        self.assertIs(reconcile(regex, cleared, hit, resolution), cleared)
        hit.llm_dob_only = False
        self.assertIs(reconcile(regex, cleared, hit, resolution), regex)
        location = self._case("DIFFERENT_ENTITY", "LOCATION_MISMATCH")
        hit.llm_dob_only = True
        self.assertIs(reconcile(regex, location, hit, resolution), regex)

    def test_a_withheld_date_cannot_open_a_clearance(self):
        regex = self._case("REVIEW", "INSUFFICIENT_DATA")
        cleared = self._case("DIFFERENT_ENTITY", "DOB_MISMATCH")
        hit = type("Hit", (), {"llm_dob_only": True, "dob_withheld": True})()
        resolution = type("Res", (), {"persons": []})()
        self.assertIs(reconcile(regex, cleared, hit, resolution), regex)

    def test_dropping_a_clearance_is_kept(self):
        regex = self._case("DIFFERENT_ENTITY", "DOB_MISMATCH")
        review = self._case("REVIEW", "INSUFFICIENT_DATA")
        hit = type("Hit", (), {"llm_dob_only": False, "dob_withheld": True})()
        resolution = type("Res", (), {"persons": []})()
        self.assertIs(reconcile(regex, review, hit, resolution), review)


class Client(unittest.TestCase):
    def test_cache_and_schema_and_failure(self):
        calls = {"n": 0}

        class Transport:
            def complete(self, prompt, payload):
                calls["n"] += 1
                if payload["items"][0]["text"] == "bad":
                    return "not json", 3, 1
                if payload["items"][0]["text"] == "boom":
                    raise urllib.error.URLError("down")
                return json.dumps({"items": [{
                    "id": payload["items"][0]["id"],
                    "outcome": "passed",
                    "dob": None,
                    "countries": [],
                    "cities": [],
                }]}), 10, 4

        client = LlmClient(Transport())
        client._batch("note", [("N1", "Onfido verified the passport.", "")])
        client._batch("note", [("N1", "Onfido verified the passport.", "")])
        self.assertEqual(calls["n"], 1)
        self.assertEqual(client.stats.cache_hits, 1)
        self.assertEqual(len(cache_key(MODEL, "p", "t")), 64)
        client._batch("note", [("N2", "bad", "")])
        self.assertGreaterEqual(client.stats.failures, 1)
        self.assertIsNone(parse_model_json("not json"))
        client._batch("note", [("N3", "boom", "")])
        from screening_ml.llm_extract import PROMPT_NOTE
        self.assertIsNone(client.cache[cache_key(MODEL, PROMPT_NOTE, "note\nN3\n\nboom")])

    def test_pinned_model_is_grok_4_fast(self):
        from screening_ml.llm_extract import PRICE_INPUT_PER_TOKEN, PRICE_OUTPUT_PER_TOKEN
        self.assertEqual(MODEL, "x-ai/grok-4-fast")
        self.assertEqual(PRICE_INPUT_PER_TOKEN, 0.20 / 1_000_000)
        self.assertEqual(PRICE_OUTPUT_PER_TOKEN, 0.50 / 1_000_000)

    def test_no_key_means_no_client(self):
        saved = {name: os.environ.pop(name, None) for name in ("OPENROUTER_API_KEY", "SCREENING_LLM", "SCREENING_LLM_MOCK")}
        try:
            self.assertIsNone(LlmClient.from_env())
            os.environ["OPENROUTER_API_KEY"] = "test-key"
            os.environ["SCREENING_LLM"] = "0"
            self.assertIsNone(LlmClient.from_env())
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


class OnAHit(unittest.TestCase):
    def test_quoted_age_fills_a_missing_birth_year_and_can_clear(self):
        # a wording the regex cannot read, so only the model's checked quote supplies the birth year
        text = "Investigators noted that by the March 2019 sentencing she had 45 birthdays behind her."
        quote = "by the March 2019 sentencing she had 45 birthdays"
        hit = {
            "hit_id": "H-000001", "entity_type": "Individual", "name": "Jane Doe",
            "profile_notes": text, "categories": ["adverse media"], "primary_category": "adverse media",
        }
        room = type("Room", (), {"search_by_query": defaultdict(list)})()
        book = ExtractionBook(
            notes={},
            profiles={"H-000001": ProfileFacts(years=[1973, 1974], quote=quote)},
            searches={},
        )
        facts = build_hit(room, hit, book)
        self.assertTrue(facts.llm_dob_only)
        self.assertEqual(facts.dobs, [(1973, None, None), (1974, None, None)])
        person = Person(
            records=["CUS-1"], primary="CUS-1", entity_type="Individual",
            names=[], countries={"France": ["CUS-1"]},
        )
        from screening_ml.customer import Fact
        person.names = [Fact(["jane", "doe"], ["CUS-1"])]
        person.dob = Fact((1990, 1, 1), ["CUS-1"])
        comp = compare(person, facts, "2026-11-01")
        self.assertEqual(teacher_code(comp), "DOB_MISMATCH")

    def test_a_quote_that_is_not_in_the_text_does_not_clear(self):
        hit = {
            "hit_id": "H-000001", "entity_type": "Individual", "name": "Jane Doe",
            "profile_notes": "No age is stated here.", "categories": [], "primary_category": "",
        }
        room = type("Room", (), {"search_by_query": defaultdict(list)})()
        book = ExtractionBook(
            notes={},
            profiles={"H-000001": ProfileFacts(years=[1973, 1974], quote="aged 45 in March 2019")},
            searches={},
        )
        facts = build_hit(room, hit, book)
        self.assertEqual(facts.dobs, [])
        self.assertFalse(facts.llm_dob_only)

    def test_rejected_model_outcome_discards_the_note(self):
        note = {
            "note_id": "NOTE-1", "channel": "kyc", "requester_email": "a@b.c",
            "created_at": "2024-01-02",
            "text": "Onfido verified JANE DOE, born 1 January 1990, on 2024-01-02.",
        }
        room = type("Room", (), {})()
        room.person_of = {}
        room.records_of_person = {}
        room.customer_by_id = {"C1": {
            "record_id": "C1", "entity_type": "Individual", "full_name": "Jane Doe",
            "email": "a@b.c", "dob": "1990-01-01", "residence_country": "Nigeria",
            "nationality": "Nigeria",
        }}
        room.applicants_by_customer = defaultdict(list)
        room.checks_by_applicant = defaultdict(list)
        room.notes_by_email = {"a@b.c": [note]}
        book = ExtractionBook(
            notes={"NOTE-1": NoteFacts(outcome="rejected")},
            profiles={}, searches={},
        )
        person = build_person(room, "C1", "2026-11-01", book)
        self.assertTrue(person.identity_unreadable)
        self.assertIsNone(person.dob)
        hit = HitFacts(hit_id="H-1", entity_type="Individual", names=[("Jane Doe", ["jane", "doe"])], countries={"Pakistan"})
        from screening_ml.customer import Fact
        person.names = [Fact(["jane", "doe"], ["C1"])]
        comp = compare(person, hit, "2026-11-01")
        self.assertFalse(comp.checks["LOCATION_MISMATCH"].fired)


if __name__ == "__main__":
    unittest.main()
