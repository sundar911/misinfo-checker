# app/verifier.py
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import os
from app.search import google_search
from app.bias_lookup import get_bias_info
import tldextract


def extract_domain(url):
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}"

def verify_claim(text):
    search_results = google_search(text)
    sources_summary = ""
    urls = []

    for result in search_results[:3]:
        domain = extract_domain(result['link'])
        bias, credibility = get_bias_info(domain)
        sources_summary += f"\n**Source**: [{result['title']}]({result['link']})\n"
        sources_summary += f"> {result['snippet']}\n"
        sources_summary += f"> 🧭 Bias: `{bias}` | 🛡️ Credibility: `{credibility}`\n"
        urls.append(result['link'])

    # Prepare LLM prompt
    prompt = f"""
    Given the following message:
    """
    {text}
    """
    And the following information from web sources:
    {sources_summary}

    Assess whether the original message is true, false, or misleading. Provide a short justification.
    """

    try:
        response = client.chat.completions.create(model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a fact-checking assistant."},
            {"role": "user", "content": prompt}
        ])
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"Error verifying claim: {str(e)}"

    return f"### ✅ Verdict\n\n{answer}\n\n---\n### 🔎 Sources\n{sources_summary}"
