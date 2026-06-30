"""
gemini.py
---------
Thin client for Google's Gemini API (the free-tier LLM doing sentiment scoring).

Uses the stable REST endpoint so there is no SDK-version churn:
    POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
    header: x-goog-api-key: <key>

Default model is gemini-2.5-flash-lite (most generous free quota, ~1,000/day).
Set GEMINI_MODEL in .env to change it.

`score_headlines` sends a batch of a stock's headlines in ONE call and asks for a
single 0-10 score (5 = neutral) -- batching keeps us well inside the free quota.
"""

import re
import time
import requests
import config
import random

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

PROMPT = (
    "You are a financial news sentiment analyst. Based ONLY on the headlines "
    "below about {company}, rate the outlook for its stock on a scale of 0 to 10, "
    "where 0 = very negative, 5 = neutral/average, 10 = very positive. "
    "Reply with ONLY the number (it may have one decimal).\n\nHeadlines:\n{headlines}"
)


def available() -> bool:
    return bool(config.GEMINI_API_KEY)


def score_headlines(company: str, headlines: list[str], retries: int = 4) -> float:
    """Return a 0-10 sentiment score for one stock. Raises on hard failure."""
    if not headlines:
        return 5.0
    text = "\n".join(f"- {h}" for h in headlines[:15])
    body = {"contents": [{"parts": [{"text": PROMPT.format(company=company, headlines=text)}]}]}
    url = ENDPOINT.format(model=config.GEMINI_MODEL)
    headers = {"x-goog-api-key": config.GEMINI_API_KEY, "Content-Type": "application/json"}

    last = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=30)
        except requests.RequestException as e:        # network blip
            last = str(e)
            time.sleep(min(30, 2 ** attempt) + random.random())
            continue
        # 429 = rate limited, 5xx (e.g. 503) = server overloaded -> both transient
        if resp.status_code == 429 or resp.status_code >= 500:
            last = f"{resp.status_code} {resp.reason}"
            time.sleep(min(30, 2 ** attempt) + random.random())
            continue
        resp.raise_for_status()
        out = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        m = re.search(r"\d+(\.\d+)?", out)
        return max(0.0, min(10.0, float(m.group()))) if m else 5.0
    raise RuntimeError(f"Gemini unavailable after {retries} retries (last: {last})")