"""
news_client.py
--------------
Fetches recent news headlines per stock and caches them to disk.

Default provider is Marketaux (free tier, covers Indian tickers, simple JSON).
The interface returns a uniform list of dicts: {date, ticker, text}. Swapping
providers only means writing another `_fetch_*` function with the same output.

IMPORTANT (free-tier reality): free news APIs only return RECENT news (roughly
the last days to ~1 month) and are rate-limited (~100 requests/day). That is
fine for the live recommendation, but it means a multi-year *sentiment* backtest
needs either a paid/historical news source or a cache you build up over time.
This client always caches what it fetches, so history accumulates as you run it.
"""

import json
import time
import requests
import pandas as pd
import config


def _cache_path(ticker, after):
    safe = ticker.replace(".", "_")
    return config.CACHE / f"news_{safe}_{after}.json"


def _fetch_marketaux(symbol, company, after, limit=20):
    url = "https://api.marketaux.com/v1/news/all"
    params = {
        "symbols": symbol,
        "filter_entities": "true",
        "language": "en",
        "published_after": after,
        "limit": limit,
        "api_token": config.NEWS_API_KEY,
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json().get("data", [])
    out = []
    for a in data:
        text = a.get("title", "") + ". " + (a.get("description") or a.get("snippet") or "")
        out.append({"date": a.get("published_at", "")[:10], "text": text.strip()})
    # fallback: some Indian tickers resolve better by company-name search
    if not out:
        params.pop("symbols"); params["search"] = company
        r = requests.get(url, params=params, timeout=20)
        if r.ok:
            for a in r.json().get("data", []):
                text = a.get("title", "") + ". " + (a.get("description") or "")
                out.append({"date": a.get("published_at", "")[:10], "text": text.strip()})
    return out


def _fetch_newsdata(symbol, company, after, limit=20):
    # NewsData.io has strong India coverage. Free tier = latest news (the
    # historical 'archive' endpoint is paid), so `after` isn't applied here.
    url = "https://newsdata.io/api/1/news"
    params = {
        "apikey": config.NEWS_API_KEY,
        "q": company,
        "country": "in",
        "language": "en",
        "category": "business",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    out = []
    for a in r.json().get("results", [])[:limit]:
        text = (a.get("title", "") + ". " + (a.get("description") or "")).strip()
        out.append({"date": (a.get("pubDate") or "")[:10], "text": text})
    return out


_PROVIDERS = {"marketaux": _fetch_marketaux, "newsdata": _fetch_newsdata}


def get_news(symbol, company, after, refresh=False):
    """Recent news for one stock since `after` (YYYY-MM-DD). Cached to disk."""
    path = _cache_path(symbol, after)
    if path.exists() and not refresh:
        return json.loads(path.read_text())

    if not config.NEWS_API_KEY:
        return []  # no key -> no news; sentiment agent will treat as neutral

    try:
        fetch = _PROVIDERS.get(config.NEWS_PROVIDER)
        if fetch is None:
            raise ValueError(f"Unknown NEWS_PROVIDER: {config.NEWS_PROVIDER}")
        items = fetch(symbol, company, after)
    except Exception as e:
        print(f"  [news] fetch failed for {symbol}: {e}")
        items = []

    for it in items:
        it["ticker"] = symbol
    path.write_text(json.dumps(items))
    time.sleep(0.3)  # be gentle with the free-tier rate limit
    return items
