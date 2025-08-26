# app/verifier.py
import os
from openai import OpenAI
from app.search import google_search

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_VERIFIER = (
    "You are a careful misinformation checker. "
    "Given a user message and a set of sources (titles+urls+snippets), "
    "produce a succinct verdict and a short structured explanation. "
    "Possible verdicts: True, False, Misleading, Uncertain. "
    "If sources don’t directly resolve a claim, say so and advise what would be needed."
)

def verify_claim(text: str, mode: str = "Balanced mix"):
    # 1) search
    results = google_search(text, k=6, mode=mode)

    if not results:
        return {
            "verdict_md": "### ⚠️ No sources found\nTry rephrasing your message or switching modes.",
            "sources": []
        }

    # 2) prepare prompt
    src_lines = []
    for r in results:
        src_lines.append(
            f"- {r['title']} ({r['link']})\n  {r['snippet']}\n  "
            f"Bias: {r.get('bias','Unknown')} | Credibility: {r.get('credibility','Unrated')} | {r.get('why_included','')}"
        )
    src_block = "\n".join(src_lines)

    user_prompt = f"""Message to assess:
\"\"\"{text}\"\"\"

Sources:
{src_block}

Return Markdown with:
- **Verdict:** <True/False/Misleading/Uncertain>
- **Justification:** 3-6 bullet points
- **Missing context or open questions:** (optional, if applicable)
"""

    # 3) LLM call
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_VERIFIER},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        answer = f"Error verifying claim: {e}"

    # 4) Return for UI
    return {
        "verdict_md": answer,
        "sources": results
    }
