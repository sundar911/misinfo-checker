# app/search.py
import os
import requests

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

headers = {
    "X-API-KEY": SERPER_API_KEY,
    "Content-Type": "application/json"
}

def google_search(query):
    url = "https://google.serper.dev/search"
    payload = {"q": query, "num": 5}
    try:
        res = requests.post(url, json=payload, headers=headers)
        results = res.json().get("organic", [])
        return results
    except Exception as e:
        print(f"Search error: {e}")
        return []
