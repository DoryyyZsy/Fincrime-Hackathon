from src.io import PROCESSED, ROOT, config, read_json, read_rows, write_json
from src.route import route


def evaluate():
    labels = read_rows("labels")
    truth = {(row["customer_id"], row["true_entity_id"]) for row in labels if row["is_true_hit"]}
    baseline, scored = read_json("baseline"), read_json("scored")
    alerts_path = PROCESSED / "alerts.json"
    cutoff = max((PROCESSED / "scored.json").stat().st_mtime_ns, (ROOT / "config.yaml").stat().st_mtime_ns)
    alerts = read_json("alerts") if alerts_path.exists() and alerts_path.stat().st_mtime_ns >= cutoff else None
    current = alerts if alerts is not None else [
        {**row, "route": route(row, config())} for row in scored
    ]
    baseline_keys = {(row["customer_id"], row["entity_id"]) for row in baseline}
    current_keys = {
        (row["customer"]["customer_id"], row["entity"]["entity_id"]) if alerts is not None
        else (row["customer_id"], row["entity_id"])
        for row in current if row["route"] != "auto_dismissed"
    }

    def summary(keys):
        return {"alerts": len(keys), "caught": len(keys & truth),
                "missed": len(truth - keys), "false_positives": len(keys - truth)}

    metrics = {"baseline": summary(baseline_keys), "ours": summary(current_keys)}
    metrics["alert_reduction_pct"] = 100 * (1 - len(current_keys) / len(baseline_keys)) if baseline_keys else None
    metrics["baseline_missed_pairs"] = [
        {"customer_id": cid, "entity_id": eid, "reason": "No blocked name scored at the fuzzy threshold"}
        for cid, eid in sorted(truth - baseline_keys)
    ]
    scores = {(row["customer_id"], row["entity_id"]): row for row in scored}
    metrics["missed_pairs"] = [
        {"customer_id": cid, "entity_id": eid, "reason": (
            f"Scored {scores[cid, eid]['match_probability']:.6f} and was dismissed"
            if (cid, eid) in scores else "No alias or date rows passed blocking")}
        for cid, eid in sorted(truth - current_keys)
    ]
    write_json("metrics", metrics)
    print(f"{'Metric':<20}{'Baseline':>12}{'Ours':>12}")
    for key in metrics["baseline"]:
        print(f"{key:<20}{metrics['baseline'][key]:>12}{metrics['ours'][key]:>12}")
    if metrics["alert_reduction_pct"] is not None:
        print(f"Alert reduction: {metrics['alert_reduction_pct']:.1f}%")
    for system, missed in (("Baseline", "baseline_missed_pairs"), ("Splink", "missed_pairs")):
        for row in metrics[missed]:
            print(f"{system} missed {row['customer_id']} → {row['entity_id']}: {row['reason']}")
    return metrics


if __name__ == "__main__":
    evaluate()
