def route(pair, thresholds, adjudication=None):
    probability = pair["match_probability"]
    if pair["exact_id"] or probability >= thresholds["likely"]:
        return "likely_true_hit"
    if probability < thresholds["grey"]:
        return "auto_dismissed"
    if adjudication is None:
        return "grey_zone"
    decision = adjudication.get("decision")
    if decision == "likely_true_hit":
        return "likely_true_hit"
    if decision == "dismiss" and adjudication.get("confidence") == "high":
        return "auto_dismissed"
    return "human_review"
