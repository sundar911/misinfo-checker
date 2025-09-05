# app/verifier.py
from __future__ import annotations

import os
from typing import Dict, List

from openai import OpenAI
from app.search import plan_search, run_trusty_retrieval

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_VERIFIER = (
    "You are a calm, empathetic misinformation checker.\n"
    "TONE: kind, neutral, non-confrontational; acknowledge why message feels plausible.\n"
    "METHOD: Given the parsed claims (explicit + implied + polarising flags) and per-claim trusted sources,\n"
    "assess each claim with a label in {Verified, Partially supported, Unsubstantiated, Unsupported & polarising, Uncertain}.\n"
    "Cite sources as [C{id}-S{n}]. If none for a claim, say so and suggest what evidence would help.\n"
    "OUTPUT (Markdown):\n"
    "### 🧘 Calm review\n"
    "1–2 gentle sentences acknowledging concerns.\n\n"
    "**Claims in the message:**\n"
    "1) <claim text>  \n"
    "   - **Status:** <label>\n"
    "   - **Why:** 1–3 bullets citing [C{id}-S{n}] when possible.\n\n"
    "**Overall:** 1 short integrative paragraph.\n"
    "**What would settle it:** 1–3 specific data/doc pointers."
)

def _sources_md(per_claim: List[Dict]) -> str:
    blocks = []
    for c in per_claim:
        cid = c["id"]
        items = c.get("sources", [])
        if not items:
            blocks.append(f"**Claim {cid}:** _no trustable source found_. Note: {c.get('note','')}")
            continue
        lines = [f"**Claim {cid}:** {c['claim']}"]
        for i, s in enumerate(items, start=1):
            title = s.get("title") or "(untitled)"
            url = s.get("link") or ""
            snip = s.get("snippet") or ""
            trust = s.get("trust", 0)
            reason = s.get("trust_reason","source")
            lines.append(f"- [C{cid}-S{i}] **[{title}]({url})** — _tier {trust}: {reason}_\n  {snip}")
        if c.get("note"):
            lines.append(f"_Note:_ {c['note']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)

def verify_claim(text: str):
    plan = plan_search(text)
    retrieval = run_trusty_retrieval(plan, k=6)

    per_claim = retrieval.get("per_claim", [])
    sources_md = _sources_md(per_claim)

    # Build a compact per-claim reference map for the model
    claims_for_llm = []
    for c in per_claim:
        claims_for_llm.append({
            "id": c["id"],
            "claim": c["claim"],
            "has_sources": bool(c.get("sources")),
        })

    user_prompt = f"""User message:
\"\"\"{text}\"\"\"

Parsed claims:
{claims_for_llm}

Trusted sources (grouped by claim, cite as [C{{id}}-S{{n}}]):
{sources_md}
"""

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_VERIFIER},
                {"role": "user", "content": user_prompt},
            ],
        )
        verdict_md = resp.choices[0].message.content
    except Exception as e:
        verdict_md = f"Error verifying claim: {e}"

    # also return flat list if UI needs it
    return {
        "verdict_md": verdict_md,
        "per_claim": per_claim,
        "flat_sources": retrieval.get("flat", []),
        # back-compat for older UI expecting `sources`
        "sources": retrieval.get("flat", []),
    }
