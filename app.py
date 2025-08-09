# app.py
import streamlit as st
from app.ui import render_ui

st.set_page_config(page_title="Misinformation & Bias Checker", layout="centered")

def main():
    render_ui()

if __name__ == "__main__":
    main()
