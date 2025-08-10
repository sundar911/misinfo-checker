# app/verifier.py (just replace verify_claim + helpers if needed)
import tldextract
import streamlit as st
from openai import OpenAI
from app.search import google_search
from app.bias_lookup import get_bias_info

def _get_openai_client():
    key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
    if not key:
        import os
        key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=key)

def extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    return ".".join([p for p in [ext.domain, ext.suffix] if p])

def _render_sources_block(results):
    parts = []
    for r in results:
        domain = extract_domain(r["link"])
        bias, cred = get_bias_info(domain)
        parts.append(
            f"**Source**: [{r['title']}]({r['link']})\n"
            f"> {r.get('snippet','')}\n"
            f"> 🧭 Bias: `{bias}` | 🛡️ Credibility: `{cred}`"
        )
    return "\n\n".join(parts) if parts else "_No relevant sources found._"

def verify_claim(text: str) -> str:
    client = _get_openai_client()

    results = google_search(text, k=6)
    sources_md = _render_sources_block(results)

    prompt = f"""
You are a careful fact-checking assistant.

Message to assess:
\"\"\"{text}\"\"\"

From open web snippets (titles/snippets) below and your knowledge, provide:
1) One-line verdict: True / False / Misleading / Unverified.
2) 3–6 sentence justification grounded in the snippets or explain lack of reliable data.
3) Missing context users should know.
Avoid moralizing; be neutral and specific.

Web snippets:
{sources_md}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Be rigorous, cite concrete facts, keep it concise."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        answer = f"Error verifying claim: {e}"

    return f"### ✅ Verdict\n\n{answer}\n\n---\n### 🔎 Sources\n{sources_md}"
