"""Local inference for the selected two-head HitNet, using only Python's standard library."""
import json
import math
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from .compare import CLEAR_ORDER, CODE_DECISION, DECISIONS, REASON_CODES, clean_evidence, evidence_for
from .features import FEATURE_NAMES, feature_values
from .normalize import transliterated_name_agrees


@lru_cache(maxsize=1)
def parameters():
    with Path(__file__).with_name("hitnet_weights.json").open() as handle:
        model = json.load(handle)
    if model["features"] != FEATURE_NAMES or model["decisions"] != DECISIONS or model["codes"] != REASON_CODES:
        raise ValueError("HitNet feature or output order differs from its trained checkpoint")
    return model


def dense(x, weights, prefix):
    return [bias + sum(w * value for w, value in zip(row, x))
            for row, bias in zip(weights[prefix + ".weight"], weights[prefix + ".bias"])]


def forward(values, model):
    weights = model["weights"]
    x = [(v - m) / s for v, m, s in zip(values, model["mean"], model["std"])]
    for linear, norm in (("backbone.0", "backbone.1"), ("backbone.4", "backbone.5")):
        x = dense(x, weights, linear)
        mean = sum(x) / len(x)
        scale = math.sqrt(sum((v - mean) ** 2 for v in x) / len(x) + 1e-5)
        x = [(v - mean) / scale * w + b
             for v, w, b in zip(x, weights[norm + ".weight"], weights[norm + ".bias"])]
        x = [0.5 * v * (1 + math.erf(v / math.sqrt(2))) for v in x]
    return dense(x, weights, "decision_head"), dense(x, weights, "reason_head")


def classify(case, alert, raw):
    model = parameters()
    values = feature_values(case, alert, raw)
    logits, reasons = forward(values, model)
    index = max(range(len(logits)), key=logits.__getitem__)
    decision = DECISIONS[index]
    probability = 1 / sum(math.exp(v - logits[index]) for v in logits)
    if decision == "DIFFERENT_ENTITY" and probability < model["clear_min_prob"]:
        decision = "REVIEW"
    code = max((c for c in REASON_CODES if CODE_DECISION[c] == decision),
               key=lambda c: reasons[REASON_CODES.index(c)])
    checks = case.comparisons[0].checks
    fired = [c for c in CLEAR_ORDER if checks[c].fired]
    if decision == "DIFFERENT_ENTITY":
        if case.teacher_decision != decision or not fired:
            decision = "REVIEW"
        elif code not in fired:
            code = fired[0]
    elif decision == "SAME_ENTITY" and (fired or case.teacher_code == "AMBIGUOUS_CUSTOMER_RECORD"):
        decision = "REVIEW"

    if decision == "DIFFERENT_ENTITY":
        hold = clearance_review(case, values, model, fired)
        if hold:
            explanation, sources = hold
            return replace(case, teacher_decision="REVIEW", teacher_code="INSUFFICIENT_DATA",
                           teacher_reason=checks[code].text.strip() + " " + explanation,
                           teacher_evidence=clean_evidence(checks[code].evidence + sources))
    if decision == "REVIEW":
        code = case.teacher_code if case.teacher_decision == "REVIEW" else "INSUFFICIENT_DATA"
    if decision == "REVIEW" and case.teacher_decision != decision:
        evidence = case.teacher_evidence
        reason = ("HitNet referred this hit for identity review. The evidence comparison found "
                  + case.teacher_code.lower().replace("_", " ") + ", but the model did not return a supported final match or clearance.")
    elif decision == "SAME_ENTITY" and not checks[code].fired:
        if case.teacher_decision == "SAME_ENTITY":
            code = case.teacher_code
            evidence, reason = evidence_for(case, code)
        else:
            decision, code = "REVIEW", case.teacher_code
            evidence, reason = case.teacher_evidence, case.teacher_reason
    else:
        evidence, reason = evidence_for(case, code)
    return replace(case, teacher_decision=decision, teacher_code=code, teacher_reason=reason, teacher_evidence=evidence)


def clearance_review(case, values, model, fired):
    """Check alternate spellings and dependence on the upstream screening score."""
    if set(fired) <= {"NAME_MISMATCH_VERIFIED", "LOCATION_MISMATCH"}:
        for comparison in case.comparisons:
            person = comparison.person
            names = list(person.names)
            if person.verified_name:
                names.append(person.verified_name)
            for fact in names:
                for display, parts in comparison.hit.names:
                    if transliterated_name_agrees(fact.value, parts):
                        return (f"The customer name and the listed name {display.rstrip('. ')} have the same "
                                "transliterated spelling, so an analyst should review the identity.", fact.sources)

    # Ditto's field-removal idea, applied at inference. Replacing the vendor
    # score by its training mean zeros that input after standardization.
    neutral = values.copy()
    column = FEATURE_NAMES.index("match_score")
    neutral[column] = model["mean"][column]
    logits, _ = forward(neutral, model)
    peak = max(logits)
    probabilities = [math.exp(v - peak) for v in logits]
    p_clear = probabilities[DECISIONS.index("DIFFERENT_ENTITY")] / sum(probabilities)
    if p_clear < model["clear_min_prob"]:
        return ("With the upstream screening score neutralized, HitNet falls below its clearance "
                "threshold and refers the hit for review.", [])
    return None
