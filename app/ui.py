# app/ui.py
import streamlit as st
from app.verifier import verify_claim


def render_ui():
    st.title("🔍 Misinformation & Bias Checker")
    st.markdown("Check the credibility and bias of forwarded messages and online claims.")

    user_input = st.text_area("Paste a message or claim:", height=150)

    if st.button("Verify") and user_input.strip():
        with st.spinner("Verifying..."):
            result = verify_claim(user_input)
            st.markdown(result)
