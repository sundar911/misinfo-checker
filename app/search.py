# app/search.py
import os
import re
import requests
from typing import List, Dict

# -------------------------
# Config
# -------------------------
OPENAI_MODEL = os.getenv("OPENAI_SEARCH_MODEL", "gpt-4o-mini")  # keep it cheap/fast
SERPER_URL = "https://google.serper.dev"
SERPER_TIMEOUT = 18
MAX_QUERIES = 6

# -------------------------
# Secrets / Clients
# -------------------------
def _get_tavily_key():
    try:
        import streamlit as st
        v = st.secrets.get("TAVILY_API_KEY")
        if v: return v
    except Exception:
        pass
    import os
    return os.getenv("TAVILY_API_KEY")

def _get_openai_key():
    try:
        import streamlit as st
        v = st.secrets.get("OPENAI_API_KEY")
        if v:
            return v
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")

def _get_serper_key():
    try:
        import streamlit as st
        v = st.secrets.get("SERPER_API_KEY")
        if v:
            return v
    except Exception:
        pass
    return os.getenv("SERPER_API_KEY")

def _openai_client():
    from openai import OpenAI
    key = _get_openai_key()
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY for query generation.")
    return OpenAI(api_key=key)

# -------------------------
# GPT query generation
# -------------------------
_SYSTEM = (
    "You are a web research planner. Given a user message that may contain rumors, "
    "opinions, or complex claims, produce 3-6 concise Google queries that a skilled "
    "human would use to verify/understand the claims. Include a mix of: "
    "1) fact-check queries, 2) data/official context queries, 3) broader background/context queries. "
    "Queries must be short (<= 12 words), self-contained, and free of quotes or special characters "
    "that hurt search matching. Avoid leading/trailing punctuation. "
    "Return STRICT JSON: {\"queries\":[{\"q\":\"string\",\"intent\":\"fact-check|data|context\"}, ...]} "
    "No extra text."
)

_USER_TMPL = """Message:
\"\"\"{text}\"\"\"

Constraints:
- 3 to 6 queries total
- max 12 words each
- use plain words that will match well on Google
- choose intents from: fact-check, data, context
"""

def _generate_queries_via_gpt(text: str) -> List[Dict]:
    client = _openai_client()
    # Use JSON mode to force valid output
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _USER_TMPL.format(text=text)},
        ],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content
    import json
    data = json.loads(raw or "{}")
    out = []
    for item in (data.get("queries") or []):
        q = (item.get("q") or "").strip()
        intent = (item.get("intent") or "context").strip().lower()
        if not q:
            continue
        if intent not in ("fact-check", "data", "context"):
            intent = "context"
        out.append({"q": q, "intent": intent})
    # Conservative cap
    return out[:MAX_QUERIES]

# -------------------------
# Fallback heuristics (if GPT fails)
# -------------------------
def _shorten_for_query(t: str, max_words=12) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"[^\x20-\x7E]+", " ", t)  # strip emojis/control chars
    words = t.split()
    return " ".join(words[:max_words]) if words else t

def _fallback_queries(text: str) -> List[Dict]:
    short = _shorten_for_query(text)
    base = [{"q": short, "intent": "context"}]
    # add two mild variants
    base.append({"q": f"{short} fact check", "intent": "fact-check"})
    base.append({"q": f"{short} data report", "intent": "data"})
    return base

# --- Serper helpers -------------------------------------------------------
def _serper(endpoint: str, payload: dict) -> dict:
    key = _get_serper_key()
    if not key:
        return {}
    headers = {"X-API-KEY": key, "Content-Type": "application/json"}
    url = f"{SERPER_URL}/{endpoint}"
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=SERPER_TIMEOUT)
        if not r.ok:
            # Leave a breadcrumb so we can see 403s in the terminal
            print(f"[SERPER] {endpoint} HTTP {r.status_code}: {r.text[:300]}")
            return {}
        return r.json()
    except Exception as e:
        print(f"[SERPER] exception: {e}")
        return {}

# --- DDG with backoff ----------------------------------------------------
def _ddg_search(query: str, max_results: int = 10):
    try:
        from duckduckgo_search import DDGS
        import time
        out = []
        # small batches + backoff to dodge 202 RateLimit
        with DDGS() as ddgs:
            got = 0
            for r in ddgs.text(query, region="in-en", max_results=max_results):
                out.append({
                    "title": (r.get("title") or "")[:220],
                    "link": r.get("href") or "",
                    "snippet": (r.get("body") or "")[:300],
                })
                got += 1
                if got % 5 == 0:
                    time.sleep(1.2)
        return out
    except Exception as e:
        print("[DDG] fallback error:", e)
        return []

# --- Tavily --------------------------------------------------------------
def _tavily_search(query: str, max_results: int = 10):
    key = _get_tavily_key()
    if not key:
        return []
    try:
        from tavily import TavilyClient
        tv = TavilyClient(api_key=key)
        res = tv.search(query=query, max_results=min(max_results, 10), include_answer=False)
        out = []
        for item in res.get("results", []):
            out.append({
                "title": (item.get("title") or "")[:220],
                "link": item.get("url") or "",
                "snippet": (item.get("content") or "")[:300],
            })
        return out
    except Exception as e:
        print("[TAVILY] error:", e)
        return []


# -------------------------
# Public API
# -------------------------
# --- Main search orchestrator -------------------------------------------
def google_search(user_text: str, k: int = 6) -> list[dict]:
    try:
        plan = _generate_queries_via_gpt(user_text)
    except Exception:
        plan = []
    if not plan:
        plan = _fallback_queries(user_text)

    results = {}
    def _add(item):
        link = item.get("link")
        if link and link not in results:
            results[link] = item

    for item in plan:
        q = item["q"]

        # 1) Tavily first
        if len(results) < k:
            for it in _tavily_search(q, max_results=10):
                _add(it)
                if len(results) >= k:
                    break

        # 2) Serper (if your key/plan works later)
        if len(results) < k:
            data = _serper("search", {"q": q, "num": 10, "gl": "in", "hl": "en"})
            for it in (data.get("organic") or []):
                _add({
                    "title": (it.get("title") or "")[:220],
                    "link": it.get("link") or "",
                    "snippet": (it.get("snippet") or "")[:300],
                })
                if len(results) >= k:
                    break

        # 3) DDG fallback
        if len(results) < k:
            for it in _ddg_search(q, max_results=10):
                _add(it)
                if len(results) >= k:
                    break

        if len(results) >= k:
            break

    return list(results.values())[:k]

