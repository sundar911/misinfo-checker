# app/search.py
import os
import json
import time
import random
from typing import List, Dict, Any, Tuple
import requests
import tldextract

# --- Config ------------------------------------------------------------------

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # optional but recommended

# country preference for "Localize India" etc. ISO-2 (IN, US, AU ...)
DEFAULT_COUNTRY = os.getenv("MISINFO_COUNTRY", "IN")

# Hard-coded “credible by default” domains (gov/intl/academia)
FACTUAL_DOMAINS = [
    "gov.in", "nic.in", "mea.gov.in", "mha.gov.in",
    "who.int", "un.org", "oecd.org", "worldbank.org",
    "imf.org", "data.gov", "census.gov", "nih.gov", "nber.org",
    "gov.au", "abs.gov.au", "gov.uk", "ons.gov.uk", "europa.eu", "ec.europa.eu",
    "ac.in", "edu", "ox.ac.uk", "cam.ac.uk", "harvard.edu", "mit.edu", "anu.edu.au",
]

# Quick bias heuristics (loose). If bias_lookup.csv exists you can enrich later.
LEAN_RIGHT_HINTS = ["breitbart", "foxnews", "thefederalist", "newsmax", "washingtontimes"]
LEAN_LEFT_HINTS  = ["theguardian", "cnn.com", "msnbc", "vox.com", "huffpost"]

# --- OpenAI small helper ------------------------------------------------------

def _oai_chat(system: str, user: str) -> str:
    import openai  # uses v1 client (the new SDK)
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing")
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()

# --- Query generation (now includes *implied* claims) ------------------------

SYSTEM_CLAIMS = (
    "You extract concise, Googleable claims from text. "
    "Return JSON with `claims` (list of short claim strings). "
    "Include both explicit factual claims *and* possible implied claims that a reader might infer."
)

SYSTEM_QUERIES = (
    "You are a search-strategy assistant. For each claim, produce 2-4 efficient web search queries. "
    "Diversify intents: facts/data (.gov/.edu/intl orgs), recent news, and background explainer. "
    "Return JSON as {queries: [{claim: str, q: [strings]}]}."
)

def _extract_claims(text: str) -> List[str]:
    js = _oai_chat(SYSTEM_CLAIMS, text)
    try:
        data = json.loads(js)
        claims = [c.strip() for c in data.get("claims", []) if c.strip()]
        return claims[:6] if claims else [text[:128]]
    except Exception:
        # very defensive fallback
        return [text[:128]]

def _build_queries(claims: List[str]) -> List[str]:
    payload = {"claims": claims}
    js = _oai_chat(SYSTEM_QUERIES, json.dumps(payload, ensure_ascii=False))
    try:
        data = json.loads(js)
        out = []
        for item in data.get("queries", []):
            out += [q.strip() for q in item.get("q", []) if q.strip()]
        # de-dupe, keep order
        seen = set(); uniq = []
        for q in out:
            if q.lower() not in seen:
                seen.add(q.lower()); uniq.append(q)
        return uniq[:12] if uniq else claims
    except Exception:
        return claims

# --- Provider: Tavily (primary) ----------------------------------------------

def _tavily_search(q: str, region: str | None, max_results: int = 5) -> List[Dict[str, Any]]:
    if not TAVILY_API_KEY:
        return []
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": q,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
                "include_domains": None,
                "topic": "news" if "news" in q.lower() else "general",
                "country": region,  # ISO-2 or None
            },
            timeout=18,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        results = data.get("results", [])
        normalized = []
        for r in results:
            url = r.get("url", "")
            title = r.get("title", "") or r.get("url", "")
            snippet = r.get("content", "")[:280]
            if url:
                normalized.append({"title": title, "link": url, "snippet": snippet})
        return normalized
    except Exception:
        return []

# --- Util: normalize + tag bias/credibility ----------------------------------

def _domain(url: str) -> str:
    ext = tldextract.extract(url)
    dom = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
    return dom.lower()

def _guess_bias(domain: str) -> str:
    d = domain.lower()
    if any(h in d for h in LEAN_LEFT_HINTS):  return "Leans left (heuristic)"
    if any(h in d for h in LEAN_RIGHT_HINTS): return "Leans right (heuristic)"
    return "Unknown"

def _guess_credibility(domain: str) -> str:
    d = domain.lower()
    if any(d.endswith(suffix) or d == suffix for suffix in FACTUAL_DOMAINS):
        return "High (gov/intl/academia)"
    return "Unrated"

def _dedupe_keep_order(items: List[Dict[str, Any]], key="link") -> List[Dict[str, Any]]:
    seen = set(); out = []
    for it in items:
        k = it.get(key)
        if k and k not in seen:
            seen.add(k); out.append(it)
    return out

# --- Challenge Modes ----------------------------------------------------------

MODES = {
    # factual, low-controversy first
    "Just the facts": dict(
        prefer_factual=True, opposing=False, localize=False, per_query=4, total=8
    ),
    # slight local preference + balanced mix
    "Balanced mix": dict(
        prefer_factual=False, opposing=True, localize=True, per_query=5, total=10
    ),
    # push users out of silo with clear opposing viewpoints included
    "Break my silo": dict(
        prefer_factual=True, opposing=True, localize=False, per_query=6, total=12
    ),
    # short + explanatory about manipulation
    "Explain manipulative framing": dict(
        prefer_factual=True, opposing=False, localize=False, per_query=4, total=8
    ),
    # India-first lens (change DEFAULT_COUNTRY to user setting in UI)
    "Localize India": dict(
        prefer_factual=False, opposing=True, localize=True, per_query=5, total=10
    ),
}

def _blend_sources(rows: List[Dict[str, Any]], opposing: bool) -> List[Dict[str, Any]]:
    """Optionally ensure ideologically diverse mix."""
    if not opposing:
        return rows
    left, right, unknown = [], [], []
    for r in rows:
        b = r.get("bias", "Unknown")
        if "left" in b.lower(): left.append(r)
        elif "right" in b.lower(): right.append(r)
        else: unknown.append(r)
    # Interleave right/left, then fill unknown
    mixed = []
    while left or right:
        if left: mixed.append(left.pop(0))
        if right: mixed.append(right.pop(0))
    mixed += unknown
    return mixed

# --- Public API ---------------------------------------------------------------

def search_message(message: str,
                   mode: str = "Balanced mix",
                   per_query: int = 5) -> List[Dict[str, Any]]:
    """
    Returns normalized results: [{title, link, snippet, bias, credibility, why_included}]
    """
    cfg = MODES.get(mode, MODES["Balanced mix"])
    per_query = cfg.get("per_query", per_query)

    claims = _extract_claims(message)
    queries = _build_queries(claims)

    # Bias queries toward factual when requested
    factual_queries = []
    if cfg["prefer_factual"]:
        for c in claims[:3]:
            factual_queries += [
                f"{c} site:.gov OR site:.edu OR site:who.int OR site:un.org",
                f"{c} methodology site:.gov OR site:.edu",
            ]
    queries = factual_queries + queries

    all_rows: List[Dict[str, Any]] = []

    for q in queries:
        # local vs global runs if localize True
        regions: List[Tuple[str | None, str]] = [ (None, "global") ]
        if cfg["localize"]:
            regions.insert(0, (DEFAULT_COUNTRY, f"local:{DEFAULT_COUNTRY}"))

        for region_code, tag in regions:
            rows = _tavily_search(q, region_code, max_results=per_query)
            for r in rows:
                dom = _domain(r["link"])
                row = {
                    "title": r["title"],
                    "link": r["link"],
                    "snippet": r["snippet"],
                    "domain": dom,
                    "bias": _guess_bias(dom),
                    "credibility": _guess_credibility(dom),
                    "why_included": f"Query='{q}' scope={tag}",
                }
                all_rows.append(row)
        # be nice to APIs
        time.sleep(0.25 + random.random() * 0.25)

    # De-dupe and blend per opposing flag
    all_rows = _dedupe_keep_order(all_rows)
    all_rows = _blend_sources(all_rows, opposing=cfg["opposing"])

    # Cap total
    total = cfg.get("total", 10)
    return all_rows[:total]

# Back-compat shim used by verifier.py
def google_search(message, k=6, mode: str = "Balanced mix"):
    results = search_message(message, mode=mode, per_query=max(3, k))
    # Keep only fields verifier expects (title/link/snippet) + our tags
    return [{"title": r["title"],
             "link": r["link"],
             "snippet": r["snippet"],
             "bias": r["bias"],
             "credibility": r["credibility"],
             "why_included": r["why_included"]} for r in results]
