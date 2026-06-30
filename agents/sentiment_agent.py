"""
Sentiment Agent (LLM-powered)
-----------------------------
For each stock it: pulls recent news -> (optionally) re-ranks to the most
relevant top-k with TF-IDF -> asks Gemini for a single 0-10 score (5 = neutral).

Design choices that matter:
  * RAG re-rank: trimming to the k most relevant articles before the LLM call
    cuts tokens/latency/quota and removes off-topic noise.
  * Caching: every (ticker, date) score is cached to disk, so re-runs and
    backtests don't burn through the free Gemini quota.
  * Graceful fallback: with no GEMINI_API_KEY (or on error) it falls back to a
    tiny keyword lexicon, so the whole project still runs with zero keys.
"""

import json
import numpy as np
import config
from llm import gemini

# --- fallback lexicon (used only when Gemini isn't available) ---------------
POSITIVE = {"record", "profit", "strong", "growth", "upgrade", "robust", "bullish",
            "beat", "rally", "raises", "wins", "expand", "surge", "gains", "outperform"}
NEGATIVE = {"plunge", "miss", "weak", "downgrade", "slowing", "loss", "lawsuit",
            "bearish", "layoffs", "decline", "cuts", "fraud", "slump", "warns", "falls"}

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False


def _lexicon_score(texts):
    if not texts:
        return 5.0
    scores = []
    for t in texts:
        words = t.lower().split()
        p = sum(w in POSITIVE for w in words)
        n = sum(w in NEGATIVE for w in words)
        scores.append(5.0 if p + n == 0 else 5.0 + 5.0 * (p - n) / (p + n))
    return float(np.mean(scores))


def _rerank(texts, query, k):
    if not _HAVE_SK or len(texts) <= k:
        return texts[:k]
    vec = TfidfVectorizer(stop_words="english")
    try:
        m = vec.fit_transform(texts + [query])
    except ValueError:
        return texts[:k]
    sims = cosine_similarity(m[-1], m[:-1]).ravel()
    order = np.argsort(-sims)[:k]
    return [texts[i] for i in order]


class SentimentAgent:
    def __init__(self, news_fn, stocks, companies, k=6, use_llm=True):
        # news_fn(symbol, company, after) -> list of {date, text}
        self.news_fn = news_fn
        self.stocks = stocks
        self.companies = companies
        self.k = k
        self.use_llm = use_llm and gemini.available()
        self.cache_file = config.CACHE / "sentiment_scores.json"
        self.scores = json.loads(self.cache_file.read_text()) if self.cache_file.exists() else {}
        self.llm_calls = 0

    def _cached(self, ticker, as_of):
        return self.scores.get(f"{ticker}|{as_of}")

    def _store(self, ticker, as_of, val):
        self.scores[f"{ticker}|{as_of}"] = val
        self.cache_file.write_text(json.dumps(self.scores))

    def score(self, as_of, lookback_days=14):
        as_of = str(as_of)[:10]
        after = (np.datetime64(as_of) - np.timedelta64(lookback_days, "D")).astype(str)
        out = {}
        for s in self.stocks:
            cached = self._cached(s, as_of)
            if cached is not None:
                out[s] = cached
                continue
            items = self.news_fn(s, self.companies[s], after)
            texts = [it["text"] for it in items if it.get("text")]
            if not texts:
                out[s] = 5.0
            elif self.use_llm:
                top = _rerank(texts, f"{self.companies[s]} stock outlook earnings", self.k)
                try:
                    out[s] = gemini.score_headlines(self.companies[s], top)
                    self.llm_calls += 1
                except Exception as e:
                    print(f"  [sentiment] Gemini failed for {s} ({e}); using lexicon")
                    out[s] = _lexicon_score(top)
            else:
                out[s] = _lexicon_score(texts)
            self._store(s, as_of, out[s])
        return out
