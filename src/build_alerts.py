import argparse
from collections import Counter
import json

from src.io import ROOT, config, read_json, read_rows, write_json
from src.route import route


def build(adjudicator=None):
    customers = {row["customer_id"]: row for row in read_rows("customers")}
    entities = {row["record_id"]: row for row in read_rows("entities")}
    thresholds = config()
    scored = read_json("scored")
    alerts = []
    funnel = read_json("funnel")
    funnel["grey_zone"] = 0
    for index, pair in enumerate(scored, 1):
        customer = customers[pair["customer_id"]]
        entity = entities[pair["entity_record_id"]]
        result = route(pair, thresholds)
        if result == "grey_zone":
            funnel["grey_zone"] += 1
            outcome = adjudicator(customer, entity, pair) if adjudicator else None
            outcome = outcome or {
                "decision": "human_review", "confidence": "low", "missing_info": [],
                "reasoning": "The identity score is within the review range. An analyst should review the available identity evidence.",
                "sources": [],
            }
            result = route(pair, thresholds, outcome)
        else:
            evidence = [
                f"{field['field']} ({field['result']})"
                for field in sorted(pair["field_comparisons"], key=lambda f: abs(f["weight"]), reverse=True)
                if field["result"] != "missing"
            ]
            why = (
                "The pair has an exact scoped identifier match."
                if pair["exact_id"] else
                f"The identity score of {pair['match_probability']:.2%} is "
                + ("at or above the likely-hit threshold." if result == "likely_true_hit"
                   else "below the review threshold.")
            )
            outcome = {
                "decision": result, "reasoning": f"{why} Comparisons: {', '.join(evidence) or 'no populated identity fields'}.",
                "missing_info": [], "sources": [],
            }
        alerts.append({
            "alert_id": f"A-{index:04d}", "route": result,
            "match_probability": pair["match_probability"], "prior_weight": pair["prior_weight"],
            "customer": {**{key: customer[key] for key in ("customer_id", "name", "nationality", "occupation")},
                         "dob": customer["dob_raw"]},
            "entity": {**{key: entity[key] for key in ("entity_id", "nationality", "countries", "datasets", "source_url")},
                       "name": entity["caption"], "compared_name": entity["name"],
                       "dob_year": str(entity["dob_year"]) if entity["dob_year"] else None},
            "field_comparisons": pair["field_comparisons"], "agent": outcome, "analyst_decision": None,
        })
    demos = []
    by_pair = {(a["customer"]["customer_id"], a["entity"]["entity_id"]): a for a in alerts}
    for demo in json.loads((ROOT / "data" / "demo_cases.json").read_text()):
        alert = by_pair.get((demo["customer_id"], demo["entity_id"]))
        if not alert or alert["route"] != demo["route"]:
            continue
        if demo.get("require_source") and not alert["agent"]["sources"]:
            continue
        if evidence := demo.get("identity_evidence"):
            alert["public_identity_note"] = evidence
        demos.append({"title": demo["title"], "alert_id": alert["alert_id"]})
    write_json("demo_cases", demos)
    routes = Counter(row["route"] for row in alerts)
    funnel["human_queue"] = routes["human_review"]
    write_json("funnel", funnel)
    write_json("alerts", alerts)
    print(f"Wrote {len(alerts):,} scored pairs with dispositions and explanations")
    return alerts


if __name__ == "__main__":
    from src.adjudicate import Adjudicator

    parser = argparse.ArgumentParser()
    parser.add_argument("--research", action="store_true", help="Search public sources using Tavily.")
    parser.add_argument("--grok-synthetic", action="store_true", help="Send generated demo records to Grok.")
    args = parser.parse_args()
    build(Adjudicator(research=args.research, grok_synthetic=args.grok_synthetic))
