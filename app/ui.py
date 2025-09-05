# app/ui.py
import streamlit as st
from app.verifier import verify_claim

def render_ui():
    st.set_page_config(page_title="Misinformation & Bias Checker", layout="centered")
    st.title("🔎 Misinformation & Bias Checker")
    st.caption("Check credibility of forwarded messages and online claims.")

    text = st.text_area("Paste a message or claim:", height=180)

    if st.button("Verify"):
        if not text.strip():
            st.warning("Paste a message first.")
            return

        with st.spinner("Checking sources and assessing..."):
            out = verify_claim(text)

        st.markdown("## ✅ Verdict")
        st.markdown(out["verdict_md"])

        st.divider()
        st.markdown("## 🔎 Sources")
        if not out["sources"]:
            st.info("No sources found.")
        else:
            for r in out["sources"]:
                st.markdown(
                    f"**[{r['title']}]({r['link']})**  \n"
                    f"{r['snippet']}"
                )