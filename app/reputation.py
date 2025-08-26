# app/reputation.py
import os, json, time
from typing import Dict, List
import tldextract

from app.search import _tavily_search  # we already wrote this (HTTP version)
from openai import OpenAI

CACHE_PATH = os.path.join(os.path.dirname(__file__), "../data/reputation_cache.json")
TTL_SECONDS = 30 * 24 * 3600  # 30 days

def _load_cache() -> Dict:
    try:
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_cache(cache: Dict):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)

def _domain_from_url(url: str) -> str:
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}".lower()

def _openai():
    # prefers Streamlit secrets, then env
    try:
        import streamlit as st
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY for reputation analysis.")
    return OpenAI(api_key=key)

SYSTEM = (
    "You are a cautious media reliability rater.\n"
    "Given a news/publisher domain and a list of snippets/URLs from reputable sources, "
    "decide CREDIBILITY and (optionally) POLITICAL_LEANING.\n"
    "Rules:\n"
    "• Credibility reflects editorial standards: corrections policy, transparency, ownership, history, reputation in academia/libraries, Wikipedia reliability notes, IFCN affiliation, etc.\n"
    "• Political_leaning must be inferred ONLY if supported by strong evidence (e.g., Wikipedia infobox, peer-reviewed/academic/industry analyses). Otherwise leave Unknown.\n"
    "• NEVER use random blogs, forum opinions, or user comments as sole evidence.\n"
    "• Use only the provided items as evidence; if insufficient, return Unknown.\n"
    "Output STRICT JSON with keys: credibility (High|Medium|Low|Unknown), bias (Left|Centre-Left|Centre|Centre-Right|Right|Unknown), rationale (<=120 words), citations (array of up to 3 URLs used)."
)

USER_TMPL = """Domain: {domain}

Candidate evidence (title — url — snippet):
{bullets}

Task: Decide credibility and possible political leaning following the rules.
"""

def _fetch_signals(domain: str) -> List[Dict]:
    # Compose a small bundle of focused queries
    queries = [
        f"{domain} site:wikipedia.org",
        f"{domain} ownership",
        f"{domain} about corrections policy",
        f"{domain} editorial standards",
        f"{domain} site:ifcncodeofprinciples.poynter.org",
        f"{domain} reliability site:edu OR site:ac.uk OR site:gov",
    ]
    hits: List[Dict] = []
    seen = set()
    for q in queries:
        for r in _tavily_search(q, max_results=5):
            url = r.get("link") or ""
            if not url or url in seen:
                continue
            seen.add(url)
            hits.append(r)
    return hits[:12]  # keep prompt small

def analyze_domain(domain_or_url: str) -> Dict:
    """
    Returns:
      {
        "credibility": "High|Medium|Low|Unknown",
        "bias": "Left|Centre-Left|Centre|Centre-Right|Right|Unknown",
        "rationale": "...",
        "citations": [urls...],
        "ts": epoch_seconds
      }
    """
    domain = _domain_from_url(domain_or_url)
    cache = _load_cache()
    now = int(time.time())
    if domain in cache and now - cache[domain].get("ts", 0) < TTL_SECONDS:
        return cache[domain]

    signals = _fetch_signals(domain)
    if not signals:
        result = {
            "credibility": "Unknown",
            "bias": "Unknown",
            "rationale": "Insufficient reputable evidence retrieved.",
            "citations": [],
            "ts": now,
        }
        cache[domain] = result
        _save_cache(cache)
        return result

    bullets = "\n".join(
        f"- {s.get('title','')[:120]} — {s.get('link','')} — {s.get('snippet','')[:240]}"
        for s in signals
    )

    client = _openai()
    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_REP_MODEL", "gpt-4o-mini"),
            response_format={"type": "json_object"},
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TMPL.format(domain=domain, bullets=bullets)},
            ],
        )
        import json
        data = json.loads(resp.choices[0].message.content)
        out = {
            "credibility": data.get("credibility", "Unknown"),
            "bias": data.get("bias", "Unknown"),
            "rationale": data.get("rationale", "")[:600],
            "citations": data.get("citations", [])[:3],
            "ts": now,
        }
    except Exception as e:
        out = {
            "credibility": "Unknown",
            "bias": "Unknown",
            "rationale": f"Reputation model error: {e}",
            "citations": [],
            "ts": now,
        }

    cache[domain] = out
    _save_cache(cache)
    return out
