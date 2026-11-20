import hashlib
import json
import logging

from src.enrich import PublicResearch, allowed, env
from src.io import PROCESSED, ROOT, read_json, write_json

LOG = logging.getLogger(__name__)
PROMPT = ROOT / "src" / "adjudication_prompt.txt"
DECISIONS = {"likely_true_hit", "dismiss", "request_info", "human_review"}
CONFIDENCES = {"high", "medium", "low"}
LISTS = {"key_evidence_for_match", "key_evidence_against_match", "missing_info"}


def valid(result, sources):
    if not isinstance(result, dict) or set(result) != LISTS | {"decision", "confidence", "reasoning", "sources"}:
        return False
    if not isinstance(result["decision"], str) or not isinstance(result["confidence"], str):
        return False
    if result["decision"] not in DECISIONS or result["confidence"] not in CONFIDENCES:
        return False
    if not isinstance(result["reasoning"], str) or not result["reasoning"].strip():
        return False
    if not all(isinstance(result[key], list) and all(isinstance(s, str) for s in result[key]) for key in LISTS):
        return False
    return isinstance(result["sources"], list) and all(
        isinstance(source, dict) and set(source) == {"title", "url"}
        and isinstance(source["title"], str) and isinstance(source["url"], str)
        and source["url"] in sources and allowed(source["url"])
        for source in result["sources"]
    )


def cite_conflict(customer, findings):
    verified = [finding for finding in findings if finding.get("verified")]
    if len(verified) != 1 or not customer["dob"]:
        return None
    fact = verified[0]
    countries = set(customer["nationality"] or [])
    listed = set(fact["nationality"])
    if customer["dob"][:4] != fact["dob"][:4] or customer["dob"] == fact["dob"]:
        return None
    if not countries or not listed or not countries.isdisjoint(listed):
        return None
    return {
        "decision": "dismiss", "confidence": "high", "missing_info": [],
        "key_evidence_for_match": ["Birth years overlap."],
        "key_evidence_against_match": ["Full date of birth differs.", "Recorded nationalities differ."],
        "reasoning": (
            f"The birth years overlap, which put this pair in the grey zone. "
            f"The listed person's source gives {fact['dob']} and nationality {', '.join(sorted(listed))}. "
            f"The customer record gives {customer['dob']} and nationality {', '.join(sorted(countries))}, "
            "so both identity fields conflict."
        ),
        "sources": [{"title": fact["title"], "url": fact["url"]}],
        "method": "Reviewed public evidence",
    }


def validate_synthetic_input():
    manifest = read_json("synthetic_manifest", {})
    digest = hashlib.sha256((PROCESSED / "customers.parquet").read_bytes()).hexdigest()
    if manifest.get("customers_sha256") != digest:
        raise ValueError("Run make data to regenerate the synthetic customer file for Grok.")


class Adjudicator:
    def __init__(self, research=False, grok_synthetic=False):
        self.research = PublicResearch(enabled=research)
        self.client = None
        self.cache = read_json("adjudication_cache", {})
        self.model = env().get("GROK_MODEL")
        self.prompt = PROMPT.read_text(encoding="utf-8")
        if grok_synthetic:
            validate_synthetic_input()
            if env().get("XAI_API_KEY") and self.model:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=env()["XAI_API_KEY"], base_url="https://api.x.ai/v1",
                    timeout=15, max_retries=0,
                )

    def __call__(self, customer, entity, pair):
        findings = self.research.for_entity(entity)
        offline = cite_conflict(customer, findings)
        if not self.client:
            if offline:
                return offline
            return None
        payload = {
            "customer": {key: customer[key] for key in ("name", "dob", "nationality", "id_number", "occupation")},
            "entity": {key: entity[key] for key in ("caption", "name", "dob", "dob_year", "nationality", "source_url")},
            "field_comparisons": pair["field_comparisons"],
            "enrichment": findings,
        }
        encoded = json.dumps(payload, sort_keys=True)
        fingerprint = hashlib.sha256((self.prompt + self.model + encoded).encode()).hexdigest()
        key = json.dumps([pair["customer_id"], pair["entity_id"]])
        sources = {item["url"] for item in findings}
        cached = self.cache.get(key, {})
        if cached.get("fingerprint") == fingerprint and valid(cached.get("result"), sources):
            result = cached["result"]
        else:
            try:
                reply = self.client.chat.completions.create(
                    model=self.model, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": self.prompt},
                              {"role": "user", "content": encoded}],
                )
                result = json.loads(reply.choices[0].message.content)
                if not valid(result, sources):
                    raise ValueError("Invalid adjudication JSON")
            except Exception as exc:
                LOG.warning("Adjudication failed for %s (%s)", key, type(exc).__name__)
                return None
            self.cache[key] = {"fingerprint": fingerprint, "result": result}
            write_json("adjudication_cache", self.cache)
        if result["decision"] == "dismiss" and offline is None:
            # A model-only dismissal stays in the analyst queue.
            result = {**result, "decision": "human_review"}
        if result["decision"] == "likely_true_hit":
            comparison = {field["field"]: field["result"] for field in pair["field_comparisons"]}
            confirmed_dob = comparison["DOB"] == "exact" or any(
                finding.get("verified") and finding.get("dob") == customer["dob"]
                for finding in findings if customer["dob"]
            )
            if not confirmed_dob or comparison["Name"] not in {"exact", "very similar"}:
                result = {**result, "decision": "human_review"}
        return {**result, "method": "Grok · synthetic data"}
