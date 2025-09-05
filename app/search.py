# app/search.py
# Three-stage pipeline with 2-LLM loops:
# 1) Claims: EXTRACTOR -> REVIEWER
# 2) Queries: FRAMER -> REVIEWER
# 3) Retrieval: FETCHER -> LLM SOURCE-JUDGE per claim
#
# Public API:
#   plan_search(text) -> {"country","claims":[...]}  # after review
#   run_trusty_retrieval(plan,k=6) -> {
#       "per_claim": [{"id","claim","sources":[{title,link,snippet,domain,trust,why}] , "note"}],
#       "flat": [ ... top-k merged ... ]
#   }
#   google_search(text,k=6) -> flat list (back-compat)

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import tldextract

# -------------------------
# Config & env
# -------------------------
OPENAI_MODEL_EXTRACTOR = os.getenv("OPENAI_EXTRACTOR_MODEL", "gpt-4o-mini")
OPENAI_MODEL_REVIEWER  = os.getenv("OPENAI_REVIEWER_MODEL",  "gpt-4o")
OPENAI_MODEL_FRAMER    = os.getenv("OPENAI_FRAMER_MODEL",    OPENAI_MODEL_EXTRACTOR)
OPENAI_MODEL_QREVIEW   = os.getenv("OPENAI_QREVIEW_MODEL",   OPENAI_MODEL_REVIEWER)
OPENAI_MODEL_SOURCEJ   = os.getenv("OPENAI_SOURCEJ_MODEL",   OPENAI_MODEL_REVIEWER)

TAVILY_URL = "https://api.tavily.com/search"
TAVILY_TIMEOUT = 18

MAX_CLAIMS = 6
MAX_QUERIES_PER_CLAIM = 3
QUERY_MAX_WORDS = 12
MAX_RESULTS_PER_QUERY = 6   # API fetch fan-out
MAX_MERGED_RESULTS = 30     # before judging

DEBUG = os.getenv("SEARCH_DEBUG") == "1"

# -------------------------
# Secrets
# -------------------------
def _get_secret(name: str) -> Optional[str]:
    try:
        import streamlit as st  # type: ignore
        v = st.secrets.get(name)
        if v:
            return str(v)
    except Exception:
        pass
    return os.getenv(name)

def _openai_key() -> Optional[str]:
    return _get_secret("OPENAI_API_KEY")

def _tavily_key() -> Optional[str]:
    return _get_secret("TAVILY_API_KEY")

# -------------------------
# Clients
# -------------------------
def _openai():
    from openai import OpenAI
    key = _openai_key()
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return OpenAI(api_key=key)

# -------------------------
# Helpers
# -------------------------
def _shorten(text: str, max_words: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\x20-\x7E]+", " ", text)
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text

def _dedupe(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        k = s.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(s.strip())
    return out

def _domain(url: str) -> str:
    try:
        ext = tldextract.extract(url)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}".lower()
        return (ext.domain or "").lower()
    except Exception:
        return ""

def _is_gov(d: str) -> bool:
    return d.endswith(".gov.in") or d.endswith(".nic.in") or d.endswith(".gov")

# -------------------------
# Trust lists (extend freely)
# -------------------------
INDIA_TIER2 = {
    "timesofindia.com", "hindustantimes.com", "indianexpress.com", "thehindu.com",
    "livemint.com", "business-standard.com", "ndtv.com", "pti.in", "economictimes.com",
    "theprint.in", "scroll.in", "moneycontrol.com",
    # fact-checkers & official comms portals
    "altnews.in", "boomlive.in", "factly.in", "pib.gov.in",
    # data portals / law & police
    "data.gov.in", "ncrb.gov.in", "mha.gov.in", "prsindia.org"
}
GLOBAL_TIER1 = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "theguardian.com",
    "nytimes.com", "wsj.com", "ft.com", "washingtonpost.com", "aljazeera.com"
}
INTL_ORG_ACAD = {
    "who.int", "un.org", "worldbank.org", "imf.org", "oecd.org",
    "journals.plos.org", "nature.com", "science.org", "thelancet.com",
}
FACTCHECK_GLOBAL = {"snopes.com", "politifact.com", "factcheck.org", "fullfact.org"}

WEAK_OR_OPEN = {
    "wikipedia.org", "medium.com", "quora.com", "reddit.com", "blogspot.com",
    "wordpress.com", "substack.com", "stackexchange.com"
}

def _trust_score(d: str, country: str) -> Tuple[int, str]:
    d = d.lower()
    # Tier 3: official/judiciary/statistics
    if _is_gov(d) or d.endswith(".parliament.in") or d.endswith(".supremecourt.nic.in"):
        return 3, "official/government or judiciary"
    if country.lower() == "india":
        if d in INDIA_TIER2:
            return 2, "major Indian outlet / fact-check / official portal"
    if d in INTL_ORG_ACAD:
        return 1, "international org / academic"
    if d in GLOBAL_TIER1 or d in FACTCHECK_GLOBAL:
        return 1, "reputable global media / fact-checker"
    if any(d.endswith(x) or d == x for x in WEAK_OR_OPEN):
        return 0, "low-signal/openly editable/community"
    return 0, "unvetted/other"

# -------------------------
# LLM prompts
# -------------------------
SYSTEM_CLAIMS = (
    "You extract claims & polarising framings from user text for fact-checking.\n"
    "Return STRICT JSON with fields: country, claims[]. Each claim has: id, claim, type(explicit|implied), "
    "polarising(bool), entities[], numbers[], time_refs[].\n"
    "Max 6 claims."
)

SYSTEM_REVIEW = (
    "Review extracted claims:\n"
    "- Merge duplicates, split over-broad claims.\n"
    "- Ensure implied claims exist where a reasonable reader would infer them.\n"
    "- Keep ≤ 6 claims.\n"
    "Return SAME JSON."
)

SYSTEM_FRAME = (
    "You craft short, high-recall Google queries for each claim.\n"
    "For each claim, propose up to 3 queries, each ≤ 12 words, grounded in entities/numbers/places/time.\n"
    "Prefer neutral phrasing (no opinion words). If country provided, steer to that jurisdiction.\n"
    "Return STRICT JSON: {claims:[{id, queries:[]}]}"
)

SYSTEM_QREVIEW = (
    "You review the proposed queries:\n"
    "- Remove duplicates, remove opinionated phrasing, ensure ≤ 12 words each.\n"
    "- Ensure queries are sufficient to verify or falsify the claim.\n"
    "Return SAME JSON."
)

SYSTEM_SOURCEJ = (
    "You are a strict source judge.\n"
    "Input: a single claim, a list of fetched results (title, url, snippet, domain, trust_tier, trust_reason).\n"
    "Goal: choose ONLY trustworthy, directly relevant sources (ideally gov/official/data, major outlets, fact-checkers). "
    "Reject unvetted/weak domains unless nothing else exists.\n"
    "Return STRICT JSON: {keep:[indices...], note:\"why\"}. If none are trustworthy, keep:[], add a brief note."
)

# -------------------------
# 1) Claims (extract -> review)
# -------------------------
def plan_search(user_text: str) -> Dict[str, Any]:
    oai = _openai()
    # Extract
    try:
        r = oai.chat.completions.create(
            model=OPENAI_MODEL_EXTRACTOR,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_CLAIMS},
                {"role": "user", "content": user_text},
            ],
        )
        data = json.loads(r.choices[0].message.content or "{}")
    except Exception as e:
        if DEBUG: print("[CLAIMS] extractor error:", e)
        data = {}

    country = (data.get("country") or "Unknown") if isinstance(data.get("country"), str) else "Unknown"
    claims = data.get("claims") or []

    # Review
    if claims:
        try:
            rr = oai.chat.completions.create(
                model=OPENAI_MODEL_REVIEWER,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_REVIEW},
                    {"role": "user", "content": json.dumps({"country": country, "claims": claims}, ensure_ascii=False)},
                ],
            )
            data2 = json.loads(rr.choices[0].message.content or "{}")
            country = (data2.get("country") or country) if isinstance(data2.get("country"), str) else country
            claims = data2.get("claims") or claims
        except Exception as e:
            if DEBUG: print("[CLAIMS] reviewer error:", e)

    # normalize ids
    out = []
    for i, c in enumerate(claims[:MAX_CLAIMS], start=1):
        if not c.get("claim"): continue
        out.append({
            "id": c.get("id") or f"C{i}",
            "claim": c["claim"].strip(),
            "type": (c.get("type") or "explicit").lower(),
            "polarising": bool(c.get("polarising", False)),
            "entities": c.get("entities") or [],
            "numbers": c.get("numbers") or [],
            "time_refs": c.get("time_refs") or [],
        })
    if DEBUG:
        print("[CLAIMS] country:", country)
        print("[CLAIMS] final:", json.dumps(out, ensure_ascii=False)[:800])
    return {"country": country, "claims": out}

# -------------------------
# 2) Queries (framer -> reviewer)
# -------------------------
def _frame_queries(country: str, claims: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    oai = _openai()
    try:
        r = oai.chat.completions.create(
            model=OPENAI_MODEL_FRAMER,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_FRAME},
                {"role": "user", "content": json.dumps({"country": country, "claims": claims}, ensure_ascii=False)},
            ],
        )
        data = json.loads(r.choices[0].message.content or "{}")
    except Exception as e:
        if DEBUG: print("[QUERIES] frame error:", e)
        data = {}

    raw = {c.get("id"): c.get("queries", []) for c in (data.get("claims") or [])}
    # Review pass
    if raw:
        try:
            rr = oai.chat.completions.create(
                model=OPENAI_MODEL_QREVIEW,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_QREVIEW},
                    {"role": "user", "content": json.dumps({"country": country, "claims": [{"id": cid, "queries": q} for cid, q in raw.items()]}, ensure_ascii=False)},
                ],
            )
            data2 = json.loads(rr.choices[0].message.content or "{}")
            raw = {c.get("id"): c.get("queries", []) for c in (data2.get("claims") or [])}
        except Exception as e:
            if DEBUG: print("[QUERIES] review error:", e)

    # Normalize & limit
    out: Dict[str, List[str]] = {}
    for c in claims:
        cid = c["id"]
        qs = [ _shorten(q, QUERY_MAX_WORDS) for q in (raw.get(cid, []) or []) if isinstance(q, str) and q.strip() ]
        qs = _dedupe(qs)[:MAX_QUERIES_PER_CLAIM]
        out[cid] = qs
    if DEBUG:
        print("[QUERIES] per-claim:", json.dumps(out, ensure_ascii=False))
    return out

# -------------------------
# 3) Retrieval (fetch -> source-judge)
# -------------------------
def _tavily_search(query: str, limit: int) -> List[Dict[str, str]]:
    key = _tavily_key()
    if not key:
        if DEBUG: print("[TAVILY] missing key")
        return []
    payload = {
        "api_key": key,
        "query": query,
        "max_results": min(limit, 10),
        "search_depth": "advanced",
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
    }
    try:
        r = requests.post(TAVILY_URL, json=payload, timeout=TAVILY_TIMEOUT)
    except Exception as e:
        if DEBUG: print("[TAVILY] request error:", e)
        return []
    if r.status_code != 200:
        if DEBUG:
            try: print(f"[TAVILY] HTTP {r.status_code}:", r.json())
            except Exception: print(f"[TAVILY] HTTP {r.status_code}:", r.text[:300])
        return []
    try:
        data = r.json()
    except Exception:
        return []
    out = []
    for it in data.get("results") or []:
        out.append({
            "title": (it.get("title") or "")[:220],
            "link": it.get("url") or "",
            "snippet": (it.get("content") or "")[:700],
        })
    return out

def _score_and_merge(hits: List[Dict[str, str]], country: str) -> List[Dict[str, Any]]:
    bucket = []
    for h in hits:
        url = h.get("link", "")
        if not url: continue
        d = _domain(url)
        trust, why = _trust_score(d, country)
        e = dict(h)
        e["domain"] = d
        e["trust"] = trust
        e["trust_reason"] = why
        bucket.append(e)
    return bucket

def _judge_sources_for_claim(claim: str, candidates: List[Dict[str, Any]]) -> Tuple[List[int], str]:
    """Ask an LLM to pick truly relevant, trustworthy sources only."""
    if not candidates:
        return [], "no candidates"
    oai = _openai()
    # Prepare compact JSON for the judge
    compact = []
    for i, c in enumerate(candidates):
        compact.append({
            "i": i,
            "title": c.get("title", "")[:160],
            "url": c.get("link", ""),
            "snippet": c.get("snippet", "")[:600],
            "domain": c.get("domain", ""),
            "trust_tier": c.get("trust", 0),
            "trust_reason": c.get("trust_reason", ""),
        })
    try:
        r = oai.chat.completions.create(
            model=OPENAI_MODEL_SOURCEJ,
            temperature=0,
            response_format={"type":"json_object"},
            messages=[
                {"role":"system","content": SYSTEM_SOURCEJ},
                {"role":"user","content": json.dumps({
                    "claim": claim,
                    "candidates": compact
                }, ensure_ascii=False)}
            ]
        )
        data = json.loads(r.choices[0].message.content or "{}")
        keep = [int(x) for x in (data.get("keep") or []) if isinstance(x, int)]
        note = (data.get("note") or "").strip()
        return keep, note
    except Exception as e:
        if DEBUG: print("[SOURCEJ] error:", e)
        # fallback: keep top trustworthy
        strong_idxs = [i for i, c in enumerate(candidates) if c.get("trust",0) >= 1]
        return strong_idxs[:3], "fallback: kept highest trust"

def run_trusty_retrieval(plan: Dict[str,Any], k: int = 6) -> Dict[str, Any]:
    """For each claim: fetch with queries, trust-score, then LLM judge."""
    country = plan.get("country") or "Unknown"
    claims = plan.get("claims") or []
    # 2) frame queries
    queries_by_claim = _frame_queries(country, claims)

    per_claim = []
    merged_urls = set()
    flat: List[Dict[str, Any]] = []

    for c in claims:
        cid = c["id"]
        text = c["claim"]
        qs = queries_by_claim.get(cid, [])
        # gather hits for this claim
        hits_all: List[Dict[str, Any]] = []
        for q in qs:
            hits = _tavily_search(q, limit=MAX_RESULTS_PER_QUERY)
            hits_all.extend(_score_and_merge(hits, country))
            time.sleep(0.2)
        # truncate before judge to keep token small
        hits_all = hits_all[:MAX_MERGED_RESULTS]

        keep_idx, note = _judge_sources_for_claim(text, hits_all)
        kept = [hits_all[i] for i in keep_idx if 0 <= i < len(hits_all)]

        # add to per-claim list
        per_claim.append({
            "id": cid,
            "claim": text,
            "sources": kept,
            "note": note if kept else (note or "no trustable source found")
        })

        # build flat (unique by URL) with trust-priority
        for s in kept:
            url = s.get("link")
            if url and url not in merged_urls:
                merged_urls.add(url)
                flat.append(s)

    # sort flat by trust desc then domain alpha
    flat.sort(key=lambda x: (-int(x.get("trust",0)), x.get("domain","")))
    return {"per_claim": per_claim, "flat": flat[:k]}

# -------------------------
# Compatibility wrapper
# -------------------------
def google_search(user_text: str, k: int = 6) -> List[Dict[str, str]]:
    plan = plan_search(user_text)
    res = run_trusty_retrieval(plan, k=k)
    return res.get("flat", [])
