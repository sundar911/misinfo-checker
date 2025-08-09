# app/bias_lookup.py
import pandas as pd
import os

bias_path = os.path.join(os.path.dirname(__file__), "../data/media_bias.csv")
bias_df = pd.read_csv(bias_path)

bias_dict = bias_df.set_index("domain").to_dict(orient="index")

def get_bias_info(domain):
    info = bias_dict.get(domain, {})
    return info.get("bias", "Unknown"), info.get("credibility", "Unknown")
