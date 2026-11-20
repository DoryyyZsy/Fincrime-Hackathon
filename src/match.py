import logging
import math
from importlib.metadata import version

import altair as alt
import duckdb
from splink import DuckDBAPI, Linker, SettingsCreator, block_on
import splink.comparison_library as cl
import splink.comparison_level_library as level

from src.io import COLUMNS, PROCESSED, config, read_rows, write_json

BLOCKS = [
    block_on("name_metaphone"), block_on("first_token", "dob_year"),
    block_on("id_number"), block_on("name_no_particles"),
]
FIELDS = {
    "name_sorted": ("Name", {3: "exact", 2: "very similar", 1: "similar", 0: "mismatch"}),
    "name_tokens": ("Name tokens", {2: "two or more shared", 1: "one shared", 0: "none shared"}),
    "dob": ("DOB", {4: "exact", 3: "same year and month", 2: "same year", 1: "within one year", 0: "mismatch"}),
    "nationality": ("Nationality", {1: "overlap", 0: "mismatch"}),
    "id_number": ("ID number", {1: "exact", 0: "mismatch"}),
}
TYPES = {"name_tokens": "VARCHAR[]", "nationality": "VARCHAR[]", "dob_year": "INTEGER"}


def custom_level(sql, label):
    return level.CustomLevel(sql, label)


def settings():
    if version("splink").split(".")[0] != "4":
        raise RuntimeError("This project requires Splink v4.")
    return SettingsCreator(
        link_type="link_only",
        unique_id_column_name="record_id",
        comparisons=[
            cl.CustomComparison([
                level.NullLevel("name_sorted"),
                level.ExactMatchLevel("name_sorted", term_frequency_adjustments=True),
                level.JaroWinklerLevel("name_sorted", 0.92),
                level.JaroWinklerLevel("name_sorted", 0.85), level.ElseLevel(),
            ], "name_sorted"),
            cl.CustomComparison([
                level.NullLevel("name_tokens"),
                custom_level("len(list_intersect(name_tokens_l, name_tokens_r)) >= 2", "Two or more shared"),
                custom_level("list_has_any(name_tokens_l, name_tokens_r)", "One shared"),
                level.ElseLevel(),
            ], "name_tokens"),
            cl.CustomComparison([
                level.NullLevel("dob_year"),
                level.ExactMatchLevel("dob"),
                custom_level("left(dob_l, 7) = left(dob_r, 7)", "Same year and month"),
                level.ExactMatchLevel("dob_year"),
                custom_level("abs(dob_year_l - dob_year_r) <= 1", "Years within one"),
                level.ElseLevel(),
            ], "dob"),
            cl.CustomComparison([
                level.NullLevel("nationality"),
                custom_level("list_has_any(nationality_l, nationality_r)", "Nationality overlap"),
                level.ElseLevel(),
            ], "nationality"),
            cl.ExactMatch("id_number"),
        ],
        blocking_rules_to_generate_predictions=BLOCKS,
        probability_two_random_records_match=config()["prior"],
        retain_matching_columns=True,
        retain_intermediate_calculation_columns=True,
    )


def linker_for(model, live=False):
    db = duckdb.connect(config={"threads": 4})
    columns = ", ".join(f'CAST("{col}" AS {TYPES.get(col, "VARCHAR")}) AS "{col}"' for col in COLUMNS)
    for table in ("customers", "entities"):
        db.read_parquet(str(PROCESSED / f"{table}.parquet")).project(columns).create(table)
    if live:
        db.execute("DELETE FROM customers")
    return Linker(["customers", "entities"], model, DuckDBAPI(connection=db), set_up_basic_logging=False), db


def field_details(pair):
    res = []
    for code, (field, descriptions) in FIELDS.items():
        gamma = pair[f"gamma_{code}"]
        factor = pair[f"bf_{code}"]
        if code == "name_sorted":
            factor *= pair.get("bf_tf_adj_name_sorted") or 1
        if code == "dob":
            left = pair.get("dob_l") or pair.get("dob_year_l")
            right = pair.get("dob_r") or pair.get("dob_year_r")
        else:
            left, right = pair[f"{code}_l"], pair[f"{code}_r"]
        res.append({
            "field": field, "customer": left, "entity": right,
            "result": descriptions.get(gamma, "missing"),
            "weight": math.log2(factor) if factor else 0.0,
        })
    return res


def best_pairs(pairs, entities, prior):
    best = {}
    for pair in pairs:
        customer_id, record_id = pair["record_id_l"], pair["record_id_r"]
        if record_id not in entities:
            customer_id, record_id = record_id, customer_id
            # find_matches_to_new_records may put the new record on either side.
            pair = {((key[:-2] + "_r") if key.endswith("_l") else
                      (key[:-2] + "_l") if key.endswith("_r") else key): val
                    for key, val in pair.items()}
        eid = entities[record_id]["entity_id"]
        key = (customer_id, eid)
        probability = pair["match_probability"]
        exact_id = pair["gamma_id_number"] == 1
        order = (exact_id, probability, record_id)
        if key not in best or order > best[key][0]:
            best[key] = (order, {
                "customer_id": customer_id, "entity_id": eid, "entity_record_id": record_id,
                "match_probability": probability, "exact_id": exact_id,
                "prior_weight": math.log2(prior / (1 - prior)),
                "field_comparisons": field_details(pair),
            })
    return [entry for _, entry in sorted(best.values(), key=lambda item: (
        item[1]["customer_id"], -item[1]["match_probability"], item[1]["entity_id"]
    ))]


def waterfall(comparisons, prior_weight):
    entries = []
    end = 0
    for field, weight in [("Prior", prior_weight), *((v["field"], v["weight"]) for v in comparisons)]:
        entries.append({"Field": field, "Start": end, "End": end + weight, "Weight": weight})
        end += weight
    return alt.Chart(alt.Data(values=entries)).mark_bar().encode(
        x=alt.X("Field:N", sort=[row["Field"] for row in entries], axis=alt.Axis(labelAngle=-20)),
        y=alt.Y("Start:Q", title="Cumulative match weight (log₂ odds)"),
        y2="End:Q",
        color=alt.condition("datum.Weight >= 0", alt.value("#176b68"), alt.value("#b84e43")),
        tooltip=[alt.Tooltip("Field:N"), alt.Tooltip("Weight:Q", format=".2f")],
    ).properties(height=230)


def stabilize_dob(model, mismatch):
    # EM sees many same-name decoys, so it overestimates DOB conflicts among
    # genuine matches. Set a stated prior distribution for DOB agreement.
    comparison = next(item for item in model["comparisons"] if item["output_column_name"] == "dob")
    for entry, probability in zip(
        comparison["comparison_levels"][1:], [0.85 - mismatch, 0.02, 0.10, 0.03, mismatch]
    ):
        entry["m_probability"] = probability


def main():
    logging.basicConfig(level=logging.INFO)
    entities = {v["record_id"]: v for v in read_rows("entities")}
    linker, db = linker_for(settings())
    try:
        linker.training.estimate_u_using_random_sampling(max_pairs=1_000_000, seed=42)
        for rule in (block_on("first_token", "dob_year"), block_on("name_no_particles")):
            linker.training.estimate_parameters_using_expectation_maximisation(rule)
        model = linker.misc.save_model_to_json()
        stabilize_dob(model, config()["dob_conflict_m"])
        write_json("model", model)
    finally:
        db.close()
    linker, db = linker_for(model)
    try:
        pairs = linker.inference.predict().as_record_dict()
        scored = best_pairs(pairs, entities, model["probability_two_random_records_match"])
        write_json("scored", scored)
        write_json("funnel", {"customers": db.sql("SELECT count(*) FROM customers").fetchall()[0][0],
                              "candidates": len(pairs), "scored": len(scored)})
        print(f"{len(pairs):,} candidate rows → {len(scored):,} scored customer–entity pairs")
    finally:
        db.close()


if __name__ == "__main__":
    main()
