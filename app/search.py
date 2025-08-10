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

# -------------------------
# Serper helpers
# -------------------------
def _serper(endpoint: str, payload: Dict) -> Dict:
    key = _get_serper_key()
    if not key:
        return {}
    headers = {"X-API-KEY": key, "Content-Type": "application/json"}
    url = f"{SERPER_URL}/{endpoint}"
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=SERPER_TIMEOUT)
        return r.json() if r.ok else {}
    except Exception:
        return {}

# -------------------------
# Public API
# -------------------------
def google_search(user_text: str, k: int = 6) -> List[Dict]:
    """
    Generate search queries via GPT, execute on Serper (web then news),
    merge/dedupe, return up to k items of {title, link, snippet}.
    """
    # 1) Build queries (GPT → fallback)
    try:
        plan = _generate_queries_via_gpt(user_text)
    except Exception:
        plan = []
    if not plan:
        plan = _fallback_queries(user_text)

    # 2) Execute plan
    results: Dict[str, Dict] = {}
    for item in plan:
        q = item["q"]
        # Try general web first
        data = _serper("search", {"q": q, "num": 10})
        for it in (data.get("organic") or []):
            link = it.get("link")
            if not link or link in results:
                continue
            results[link] = {
                "title": (it.get("title") or "")[:220],
                "link": link,
                "snippet": (it.get("snippet") or "")[:300],
            }
            if len(results) >= k:
                return list(results.values())[:k]
        # Then try news if we still need more
        if len(results) < k:
            nd = _serper("news", {"q": q, "num": 10})
            for it in (nd.get("news") or []):
                link = it.get("link")
                if not link or link in results:
                    continue
                results[link] = {
                    "title": (it.get("title") or "")[:220],
                    "link": link,
                    "snippet": (it.get("snippet") or "")[:300],
                }
                if len(results) >= k:
                    return list(results.values())[:k]

    out = list(results.values())[:k]
    return out
