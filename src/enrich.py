import json
import logging
import os
from functools import lru_cache
from urllib.parse import urlparse

from src.io import ROOT, read_json, write_json

DOMAINS = ("gov.uk", "treasury.gov", "un.org", "europa.eu", "wikipedia.org", "opensanctions.org")
LOGGER = logging.getLogger(__name__)
FIXTURE = ROOT / "data" / "public_evidence.json"


@lru_cache(maxsize=1)
def env():
    result = {}
    path = ROOT / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, _, value = line.partition("=")
                result[key.strip()] = value.strip().strip("'\"")
    return {**result, **os.environ}


def allowed(url):
    parsed = urlparse(url)
    host = parsed.hostname or ""
    return parsed.scheme == "https" and any(host == domain or host.endswith("." + domain) for domain in DOMAINS)


@lru_cache(maxsize=1)
def curated():
    return json.loads(FIXTURE.read_text(encoding="utf-8")) if FIXTURE.exists() else {}


class PublicResearch:
    def __init__(self, enabled=False):
        self.cache = read_json("enrich_cache", {})
        self.client = None
        key = env().get("TAVILY_API_KEY")
        if enabled and key:
            from tavily import TavilyClient
            self.client = TavilyClient(api_key=key)

    def for_entity(self, entity):
        eid = entity["entity_id"]
        reviewed = curated().get(eid, [])
        if eid not in self.cache and self.client:
            query = " ".join(filter(None, [
                entity["caption"], " ".join(entity["countries"] or []),
                entity.get("role"), "sanctions",
            ]))
            try:
                response = self.client.search(query, include_domains=list(DOMAINS), max_results=5, timeout=10)
                self.cache[eid] = [{
                    "title": result["title"], "url": result["url"], "finding": result["content"][:1500],
                } for result in response["results"] if allowed(result.get("url", ""))]
                write_json("enrich_cache", self.cache)
            except Exception as exc:
                LOGGER.warning("Public research unavailable for %s (%s)", eid, type(exc).__name__)
        return [*reviewed, *self.cache.get(eid, [])]
