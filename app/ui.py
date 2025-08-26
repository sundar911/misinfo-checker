# app/ui.py
import streamlit as st
from app.verifier import verify_claim

MODE_HELP = {
    "Just the facts": "Prioritize gov/intl/academic sources. Lower controversy.",
    "Balanced mix": "Blend local & global, some ideological diversity.",
    "Break my silo": "Actively include opposing viewpoints to challenge assumptions.",
    "Explain manipulative framing": "Short read + highlights common persuasion tactics.",
    "Localize India": "Prefer India sources where possible + balanced mix."
}

def render_ui():
    st.set_page_config(page_title="Misinformation & Bias Checker", layout="centered")
    st.title("🔎 Misinformation & Bias Checker")
    st.caption("Check credibility & bias of forwarded messages and online claims.")

    col1, col2 = st.columns([3, 2])
    with col2:
        mode = st.selectbox(
            "Challenge mode",
            list(MODE_HELP.keys()),
            index=1,
            help=MODE_HELP["Balanced mix"]
        )
        st.caption(MODE_HELP.get(mode, ""))

    with col1:
        text = st.text_area("Paste a message or claim:", height=170)

    if st.button("Verify"):
        if not text.strip():
            st.warning("Paste a message first.")
            return

        with st.spinner("Checking sources and assessing..."):
            out = verify_claim(text, mode=mode)

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
                    f"{r['snippet']}\n\n"
                    f"🧭 Bias: `{r.get('bias','Unknown')}` | 🛡️ Credibility: `{r.get('credibility','Unrated')}`  \n"
                    f"Why included: _{r.get('why_included','')}_"
                )
