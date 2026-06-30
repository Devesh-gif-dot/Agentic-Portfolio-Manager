"""
Offline tests -- run without any API keys or network.

    python tests/test_offline.py     (or: pytest tests/)

They use a synthetic price fixture and a fake news function to exercise the full
pipeline (agents -> orchestrator -> backtest) via the lexicon fallback, and
mock Gemini's HTTP response to check the score parsing.
"""

import sys, types
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from orchestrator import Orchestrator
from backtest import run_backtest
from llm import gemini


def _fixture():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2022-01-01", periods=400)
    tickers = list(config.STOCKS) + [config.GOLD]
    rets = pd.DataFrame(rng.normal(0.0004, 0.012, (len(dates), len(tickers))),
                        index=dates, columns=tickers)
    prices = 100 * (1 + rets).cumprod()
    return prices, prices.pct_change().dropna()


def _fake_news(symbol, company, after):
    return [{"date": after, "text": f"{company} reports strong growth and record profit"}]


def test_pipeline_offline():
    prices, returns = _fixture()
    orch = Orchestrator(prices, returns, _fake_news, list(config.STOCKS),
                        config.STOCKS, config.GOLD, use_llm=False)  # lexicon fallback
    w, _ = orch.build_portfolio(prices.index[200])
    assert abs(sum(w.values()) - 1.0) < 1e-6, "weights must sum to 1"
    assert all(v >= -1e-9 for v in w.values()), "long-only"
    curve, m, _ = run_backtest(orch, prices, returns)
    assert len(curve) > 0 and "Sharpe" in m
    print("  pipeline_offline OK  | Sharpe=%.2f  weights sum=%.4f"
          % (m["Sharpe"], sum(w.values())))


def test_gemini_parsing(monkeypatch=None):
    # mock requests.post so no network is needed
    class FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "7.5"}]}}]}
    gemini.requests.post = lambda *a, **k: FakeResp()
    gemini.config.GEMINI_API_KEY = "test-key"
    score = gemini.score_headlines("Reliance", ["great quarter, profit up"])
    assert score == 7.5, f"expected 7.5, got {score}"
    print("  gemini_parsing OK    | parsed score=%.1f" % score)


if __name__ == "__main__":
    test_pipeline_offline()
    test_gemini_parsing()
    print("\nAll offline tests passed.")
