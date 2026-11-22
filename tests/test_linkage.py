"""Fellegi–Sunter match score: comparison levels, EM, the decision veto, determinism.

The tests build their own fixtures. They do not read data/screening/dev, which is
git-ignored and is not in every checkout.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from screening_ml import linkage as L
from screening_ml import normalize as N


def _pattern(**overrides):
    pattern = {
        "name": "strong", "rarity": "common", "dob": "missing", "country": "missing",
        "entity_type": "agree", "document": "missing", "alias": "unused", "time": "missing",
    }
    pattern.update(overrides)
    return pattern


class StringFeatures(unittest.TestCase):
    def test_jaro_winkler_known_pairs(self):
        self.assertEqual(L.jaro("MARTHA", "MARTHA"), 1.0)
        self.assertEqual(L.jaro("", ""), 1.0)
        self.assertEqual(L.jaro("MARTHA", ""), 0.0)
        self.assertAlmostEqual(L.jaro("MARTHA", "MARHTA"), 0.944, places=2)
        self.assertAlmostEqual(L.jaro_winkler("MARTHA", "MARHTA"), 0.961, places=2)
        self.assertAlmostEqual(L.jaro("DIXON", "DICKSONX"), 0.767, places=2)
        self.assertAlmostEqual(L.jaro_winkler("DIXON", "DICKSONX"), 0.813, places=2)
        self.assertAlmostEqual(L.jaro("JELLYFISH", "SMELLYFISH"), 0.896, places=2)

    def test_soundex_reference_codes(self):
        self.assertEqual(L.soundex("Robert"), "R163")
        self.assertEqual(L.soundex("Rupert"), "R163")
        self.assertEqual(L.soundex("Rubin"), "R150")
        self.assertEqual(L.soundex("Ashcraft"), "A261")
        self.assertEqual(L.soundex("Ashcroft"), "A261")
        self.assertEqual(L.soundex("Tymczak"), "T522")
        self.assertEqual(L.soundex("Pfister"), "P236")
        self.assertEqual(L.soundex("Mohammed"), L.soundex("Muhammad"))
        self.assertEqual(L.soundex(""), "")

    def test_phonetic_agreement_survives_a_spelling_the_edit_rule_misses(self):
        self.assertTrue(L.phonetic_match(["steven"], ["stephen"]))
        level = L.name_level([["steven"]], [("Stephen", ["stephen"])])
        self.assertIn(level, ("strong", "partial", "exact"))
        self.assertNotEqual(level, "disagree")

    def test_name_tiers(self):
        self.assertEqual(L.name_level([["elvira", "velasco"]], [("Elvira Velasco", ["elvira", "velasco"])]), "exact")
        self.assertEqual(L.name_level([["alice", "chen"]], [("Robert Smith", ["robert", "smith"])]), "disagree")
        self.assertEqual(L.name_level([], [("Elvira", ["elvira"])]), "missing")

    def test_alias_level_when_only_the_alias_agrees(self):
        # The primary name shares nothing with the customer. The alias is the customer's name.
        # This must not depend on name_variants: Bob/Robert may already be loaded by another test.
        hit_names = [("Alice Chen", ["alice", "chen"]), ("Bob Smith", ["bob", "smith"])]
        keys = {L._alias_key("Bob Smith")}
        self.assertEqual(L.alias_level([["bob", "smith"]], hit_names, keys), "used")
        self.assertEqual(L.alias_level([["alice", "chen"]], hit_names, keys), "unused")

    def test_dob_levels_including_partial_dates(self):
        full = (1968, 3, 5)
        self.assertEqual(L.dob_level(full, [(1968, 3, 5)]), "full")
        self.assertEqual(L.dob_level(full, [(1968, 3, None)]), "year_month")
        self.assertEqual(L.dob_level(full, [(1968, None, None)]), "year")
        self.assertEqual(L.dob_level(full, [(1967, None, None)]), "within_1")
        self.assertEqual(L.dob_level(full, [(1968, 4, 5)]), "conflict")
        self.assertEqual(L.dob_level(full, [(1950, 1, 1)]), "conflict")
        self.assertEqual(L.dob_level(None, [(1968, 3, 5)]), "missing")
        self.assertEqual(L.dob_level(full, []), "missing")
        # One agreeing reading is enough. Every reading must conflict to disagree.
        self.assertEqual(L.dob_level(full, [(1968, 6, 5), (1968, 3, 5)]), "full")
        self.assertEqual(L.dob_level(full, [(1950, 1, 1), (1940, None, None)]), "conflict")

    def test_country_and_entity_type(self):
        self.assertEqual(L.country_level(["Spain"], ["Spain", "France"]), "agree")
        self.assertEqual(L.country_level(["Spain"], ["France"]), "disagree")
        self.assertEqual(L.country_level([], ["France"]), "missing")
        self.assertEqual(L.entity_type_level("Individual", "Individual"), "agree")
        self.assertEqual(L.entity_type_level("Individual", "Company"), "disagree")
        self.assertEqual(L.entity_type_level("", "Company"), "missing")

    def test_rarity_uses_the_customer_file_frequency(self):
        freq = {"smith": 10, "john": 10, "velasco": 1, "elvira": 1}
        rare = L.rarity_level([["elvira", "velasco"]], [["elvira", "velasco"]], freq, 3, "exact")
        common = L.rarity_level([["john", "smith"]], [["john", "smith"]], freq, 3, "exact")
        self.assertEqual(rare, "rare")
        self.assertEqual(common, "common")
        self.assertEqual(L.rarity_level([["john", "smith"]], [["john", "smith"]], freq, 3, "disagree"), "unmatched")

    def test_document_level_is_exact_or_unknown(self):
        self.assertEqual(L.document_level(["AB123456"], "XXAB123456YY"), "exact")
        self.assertEqual(L.document_level(["AB123456"], "NOPE"), "missing")
        self.assertEqual(L.document_level([], "AB123456"), "missing")


class Model(unittest.TestCase):
    def test_initial_distributions_sum_to_one_and_keep_their_meaning(self):
        for field, levels in L.LEVELS.items():
            self.assertAlmostEqual(sum(L.M0[field][level] for level in levels), 1.0)
            self.assertAlmostEqual(sum(L.U0[field][level] for level in levels), 1.0)
            for level, sign in L.ANCHORS.get(field, {}).items():
                self.assertGreaterEqual(sign * (L.M0[field][level] - L.U0[field][level]), 0.0)

    def test_dob_conflict_lowers_the_score_and_full_agreement_raises_it(self):
        model = L.initial_fit()
        _p, _w, conflict = L.score_pattern(model, _pattern(dob="conflict"))
        _p, _w, full = L.score_pattern(model, _pattern(dob="full"))
        self.assertLess(conflict["dob"], 0.0)
        self.assertGreater(full["dob"], 0.0)

    def test_em_separates_agreement_from_contradiction(self):
        match = _pattern(name="exact", rarity="rare", dob="full", country="agree")
        other = _pattern(name="strong", rarity="common", dob="conflict", country="disagree", time="ok")
        model = L.fit([match] * 8 + [other] * 24)
        p_match, _, _ = L.score_pattern(model, match)
        p_other, _, _ = L.score_pattern(model, other)
        self.assertGreater(p_match, p_other)
        self.assertGreater(p_match, 0.5)
        self.assertLess(p_other, 0.5)
        self.assertTrue(model.converged)
        self.assertGreaterEqual(model.iterations, 1)
        self.assertLess(model.iterations, 15, f"separated mixture took {model.iterations} iterations")
        self.assertTrue(L.anchors_hold(model))
        self.assertTrue(L.weights_are_ordered(model))
        self.assertLessEqual(model.lambda_, L.LAMBDA_MAX)

    def test_starting_weights_are_ordered(self):
        model = L.initial_fit()
        self.assertTrue(L.weights_are_ordered(model))
        self.assertGreater(L.level_weight(model, "dob", "full"), L.level_weight(model, "dob", "missing"))
        self.assertGreater(L.level_weight(model, "dob", "missing"), L.level_weight(model, "dob", "conflict"))
        self.assertLessEqual(L.level_weight(model, "name", "missing"), 0.0)

    def test_a_missing_date_cannot_outrank_an_agreeing_date(self):
        """The mixture that used to label-switch: many rare exact names and no date of birth.

        An unconstrained fit called that pattern the match class, pushed the match
        rate to about 0.6, and scored it level with pairs whose dates agree. The
        ordered starting weights have to keep a full date strictly above a missing
        date, and a contradiction below both, and the fit has to settle.
        """
        rows = []
        rows += [_pattern(name="exact", rarity="rare", dob="missing", country="agree")] * 220
        rows += [_pattern(name="exact", rarity="rare", dob="full", country="agree", time="ok")] * 80
        rows += [_pattern(name="strong", rarity="common", dob="full", country="agree", time="ok")] * 40
        rows += [_pattern(name="exact", rarity="common", dob="conflict", country="disagree", time="ok")] * 250
        rows += [_pattern(name="strong", rarity="rare", dob="conflict", country="disagree", time="impossible")] * 120
        rows += [_pattern(name="exact", rarity="rare", dob="within_1", country="disagree", time="ok")] * 90
        rows += [_pattern(name="partial", rarity="common", dob="missing", country="missing", alias="used")] * 76
        model = L.fit(rows)
        self.assertTrue(model.converged, f"still moving after {model.iterations} iterations")
        self.assertLess(model.iterations, L.EM_ITERS)
        self.assertLessEqual(model.lambda_, L.LAMBDA_MAX)
        self.assertTrue(L.anchors_hold(model))
        self.assertTrue(L.weights_are_ordered(model))
        same = dict(name="exact", rarity="rare", country="agree", time="ok")
        full = L.score_pattern(model, _pattern(dob="full", **same))[0]
        year = L.score_pattern(model, _pattern(dob="year", **same))[0]
        missing = L.score_pattern(model, _pattern(dob="missing", **same))[0]
        conflict = L.score_pattern(model, _pattern(
            name="exact", rarity="common", dob="conflict", country="disagree", time="ok"))[0]
        self.assertGreater(full, year)
        self.assertGreater(year, missing)
        self.assertGreater(missing, conflict)
        # The pattern that saturated at 0.98: rare exact name, country agrees, no date.
        bare = L.score_pattern(model, _pattern(name="exact", rarity="rare", dob="missing", country="agree"))[0]
        self.assertGreater(full, bare)
        self.assertLess(bare, 0.85)
        self.assertGreater(full, 0.90)
        self.assertLess(conflict, 0.15)
        again = L.fit(list(rows))
        self.assertEqual(model.iterations, again.iterations)
        self.assertEqual(L.model_summary(model), L.model_summary(again))

    def test_fit_is_deterministic_for_a_fixed_order(self):
        rows = [_pattern(dob="full", country="agree"), _pattern(dob="conflict", country="disagree")] * 6
        first = L.model_summary(L.fit(rows))
        second = L.model_summary(L.fit(list(rows)))
        self.assertEqual(first, second)

    def test_hash_seed_does_not_change_the_fit(self):
        script = (
            "import json,sys\n"
            "sys.path.insert(0, %r)\n"
            "from screening_ml.linkage import fit, score_pattern, model_summary\n"
            "match = {'name':'exact','rarity':'rare','dob':'full','country':'agree',"
            "'entity_type':'agree','document':'missing','alias':'unused','time':'missing'}\n"
            "other = {'name':'strong','rarity':'common','dob':'conflict','country':'disagree',"
            "'entity_type':'agree','document':'missing','alias':'unused','time':'ok'}\n"
            "rows = [match, other] * 6\n"
            "model = fit(rows)\n"
            "p, w, fields = score_pattern(model, match)\n"
            "print(json.dumps({'summary': model_summary(model), 'p': p, 'w': w, 'dob': fields['dob']}))\n"
        ) % ROOT
        outputs = []
        for seed in ("0", "12345"):
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = seed
            proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            outputs.append(proc.stdout)
        self.assertEqual(outputs[0], outputs[1])

    def test_module_imports_nothing_outside_the_standard_library(self):
        script = (
            "import sys\n"
            "sys.path.insert(0, %r)\n"
            "import screening_ml.linkage, screening_ml.pipeline\n"
            "bad = sorted(m.split('.')[0] for m in sys.modules if m.split('.')[0] in "
            "{'numpy','torch','pandas','yaml','requests','sklearn'})\n"
            "print(','.join(bad))\n"
        ) % ROOT
        proc = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "")


class Veto(unittest.TestCase):
    def test_score_cannot_clear_or_confirm_what_the_rules_did_not(self):
        self.assertFalse(L.PROMOTE)
        self.assertEqual(L.guarded_decision("REVIEW", "DIFFERENT_ENTITY"), "REVIEW")
        self.assertEqual(L.guarded_decision("REVIEW", "SAME_ENTITY"), "REVIEW")
        self.assertEqual(L.guarded_decision("SAME_ENTITY", "DIFFERENT_ENTITY"), "SAME_ENTITY")
        self.assertEqual(L.guarded_decision("DIFFERENT_ENTITY", "SAME_ENTITY"), "DIFFERENT_ENTITY")
        self.assertEqual(L.guarded_decision("DIFFERENT_ENTITY", "REVIEW"), "DIFFERENT_ENTITY")
        self.assertEqual(L.guarded_decision("REVIEW", "REVIEW"), "REVIEW")

    def test_veto_holds_even_if_promotion_is_switched_on(self):
        saved = L.PROMOTE
        L.PROMOTE = True
        try:
            self.assertEqual(L.guarded_decision("REVIEW", "DIFFERENT_ENTITY"), "REVIEW")
            self.assertEqual(L.guarded_decision("REVIEW", "SAME_ENTITY"), "REVIEW")
            self.assertEqual(L.guarded_decision("SAME_ENTITY", "DIFFERENT_ENTITY"), "SAME_ENTITY")
        finally:
            L.PROMOTE = saved

    def test_clear_ceiling_matches_the_cost_table(self):
        # Unsafe clear 60, dearest review 3. The score does not use the ceiling.
        self.assertAlmostEqual(L.clear_probability_ceiling(), 3 / 60)


def _customer(record_id, name, dob, country, email, phone):
    return {
        "record_id": record_id, "entity_type": "Individual", "full_name": name,
        "preferred_name": "", "login_name": "", "dob": dob, "residence_country": country,
        "nationality": country, "email": email, "phone": phone, "city": "Madrid", "address": "1 Calle",
    }


def _write_room(folder):
    customers = [
        _customer("CUS-000001", "Elvira Velasco", "1968-03-05", "Spain", "elvira@example.com", "+34000000001"),
        _customer("CUS-000002", "John Smith", "1980-01-15", "United States", "john@example.com", "+19990000002"),
    ]
    accounts = [
        {"account_id": "ACC-000001", "customer_id": "CUS-000001", "login_handle": "elvira"},
        {"account_id": "ACC-000002", "customer_id": "CUS-000002", "login_handle": "john"},
    ]
    alerts = [
        {"alert_id": "A-000001", "screened_at": "2024-06-01", "account_ref": "ACC-000001",
         "customer": {"name": "Elvira Velasco", "entity_type": "Individual", "dob": "1968-03-05", "country": "Spain", "city": "Madrid"},
         "hit_ids": ["H-000001"]},
        {"alert_id": "A-000002", "screened_at": "2024-06-01", "account_ref": "ACC-000002",
         "customer": {"name": "John Smith", "entity_type": "Individual", "dob": "1980-01-15", "country": "United States"},
         "hit_ids": ["H-000002"]},
        {"alert_id": "A-000003", "screened_at": "2024-06-01", "account_ref": "ACC-000002",
         "customer": {"name": "John Smith", "entity_type": "Individual", "dob": "1980-01-15", "country": "United States"},
         "hit_ids": ["H-000003"]},
    ]
    hits = [
        {"hit_id": "H-000001", "alert_id": "A-000001", "entity_type": "Individual", "name": "Elvira Velasco",
         "countries": ["Spain"], "dobs": ["1968-03-05"], "primary_category": "pep", "categories": ["pep"], "aliases": []},
        {"hit_id": "H-000002", "alert_id": "A-000002", "entity_type": "Individual", "name": "John Smith",
         "countries": ["Spain"], "dobs": ["1950-01-01"], "primary_category": "sanctions", "categories": ["sanctions"], "aliases": []},
        {"hit_id": "H-000003", "alert_id": "A-000003", "entity_type": "Individual", "name": "John Smith",
         "countries": ["United States"], "dobs": [], "primary_category": "adverse media", "categories": ["adverse media"],
         "aliases": ["Johnny Smith"]},
    ]
    tables = {
        "customers": customers, "accounts": accounts, "alerts": alerts, "hits": hits,
        "onboarding": [], "kyc_checks": [], "notes": [], "search_results": [],
    }
    for name, rows in tables.items():
        with open(os.path.join(folder, name + ".jsonl"), "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")


class Engine(unittest.TestCase):
    def test_score_ranks_without_changing_decisions_and_is_repeatable(self):
        from screening_ml.compare import decide_case
        from screening_ml.customer import resolve
        from screening_ml.data import DataRoom
        from screening_ml.hit import build_hit
        from screening_ml.linkage import score_cases
        from screening_ml.neural import classify
        from screening_ml.pipeline import adjudicate

        with tempfile.TemporaryDirectory() as folder:
            _write_room(folder)
            room = DataRoom(folder)
            rows, errors, _stats, scores, fit = adjudicate(room)
            hitnet = []
            for hit in room.hits:
                alert = room.alert_by_id[hit["alert_id"]]
                case = decide_case(resolve(room, alert), build_hit(room, hit), alert)
                hitnet.append(classify(case, alert, hit).teacher_decision)
            self.assertEqual(errors, 0)
            self.assertEqual([row["decision"] for row in rows], hitnet)
            self.assertTrue(all("match_probability" not in row for row in rows))
            self.assertEqual([score["decision"] for score in scores], [row["decision"] for row in rows])
            self.assertEqual([score["hit_id"] for score in scores], ["H-000001", "H-000002", "H-000003"])
            by_id = {score["hit_id"]: score for score in scores}
            self.assertGreater(by_id["H-000001"]["match_probability"], by_id["H-000003"]["match_probability"])
            self.assertGreater(by_id["H-000003"]["match_probability"], by_id["H-000002"]["match_probability"])
            review_ids = [row["hit_id"] for row in rows if row["decision"] == "REVIEW"]
            ranked = sorted((score for score in scores if score["review_rank"] is not None), key=lambda score: score["review_rank"])
            self.assertEqual(
                [score["hit_id"] for score in ranked],
                sorted(review_ids, key=lambda hit_id: (-by_id[hit_id]["match_probability"], hit_id)),
            )
            self.assertTrue(all(0.0 <= score["match_probability"] <= 1.0 for score in scores))
            self.assertTrue(all(score["rule_veto"] for score in scores))
            self.assertTrue(fit.converged)
            self.assertGreaterEqual(fit.iterations, 1)
            self.assertLess(fit.iterations, L.EM_ITERS)

            again, _, _, scores_again, _ = adjudicate(room)
            self.assertEqual(rows, again)
            self.assertEqual(scores, scores_again)

            reversed_cases, reversed_raws = [], []
            for hit in reversed(room.hits):
                alert = room.alert_by_id[hit["alert_id"]]
                reversed_cases.append(decide_case(resolve(room, alert), build_hit(room, hit), alert))
                reversed_raws.append(hit)
            reversed_scores, _ = score_cases(room, reversed_cases, reversed_raws)
            self.assertEqual(
                {score["hit_id"]: score["match_probability"] for score in scores},
                {score["hit_id"]: score["match_probability"] for score in reversed_scores},
            )

            out = os.path.join(folder, "out.jsonl")
            env = dict(os.environ)
            env.pop("SCREENING_MATCH_SCORE", None)
            env["SCREENING_LLM"] = "0"
            first = second = None
            for seed in ("0", "12345"):
                env["PYTHONHASHSEED"] = seed
                proc = subprocess.run(
                    [sys.executable, os.path.join(ROOT, "system", "run"), folder, out],
                    capture_output=True, text=True, env=env, cwd=ROOT,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn("decisions unchanged", proc.stderr)
                with open(out, encoding="utf-8") as handle:
                    written = handle.read()
                with open(L.scores_path(out), encoding="utf-8") as handle:
                    sidecar = handle.read()
                self.assertTrue(all("match_probability" not in json.loads(line) for line in written.splitlines() if line.strip()))
                if first is None:
                    first = (written, sidecar)
                else:
                    second = (written, sidecar)
            self.assertEqual(first, second)

            env["SCREENING_MATCH_SCORE"] = "1"
            env["PYTHONHASHSEED"] = "0"
            flagged = os.path.join(folder, "flagged.jsonl")
            proc = subprocess.run(
                [sys.executable, os.path.join(ROOT, "system", "run"), folder, flagged],
                capture_output=True, text=True, env=env, cwd=ROOT,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(flagged, encoding="utf-8") as handle:
                flagged_rows = [json.loads(line) for line in handle if line.strip()]
            self.assertTrue(all("match_probability" in row for row in flagged_rows))
            self.assertEqual([row["decision"] for row in flagged_rows], [row["decision"] for row in rows])


if __name__ == "__main__":
    unittest.main()
