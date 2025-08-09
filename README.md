# misinfo-checker

Lightweight misinformation & source-bias checker. Enter a claim, we search, attach bias/credibility metadata, and ask an LLM for a verdict.

## Quickstart (local)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p .streamlit
# add your keys to .streamlit/secrets.toml
streamlit run app.py
