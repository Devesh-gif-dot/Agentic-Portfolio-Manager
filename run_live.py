"""
run_live.py
-----------
The showcase. Fetches real recent prices (yfinance) and real news (news API),
scores sentiment with Gemini, and prints today's recommended Indian-equity
portfolio with a gold hedge.

    python run_live.py

Needs GEMINI_API_KEY and NEWS_API_KEY in .env (without them it still runs, using
neutral/lexicon sentiment, so you can check the plumbing).
"""

import config
from data_sources import market_data, news_client
from orchestrator import Orchestrator


def main():
    print("Fetching prices (yfinance)...")
    prices = market_data.get_prices(start="2024-01-01", end=None)
    returns = market_data.to_returns(prices)

    orch = Orchestrator(
        prices, returns, news_client.get_news,
        stocks=list(config.STOCKS), companies=config.STOCKS, gold=config.GOLD,
        use_risk=True, use_sentiment=True, use_hedge=True, use_llm=True,
    )

    as_of = prices.index[-1]
    print(f"\nGemini sentiment: {'ON' if orch.sentiment_agent.use_llm else 'OFF (no key -> lexicon)'}")
    print(f"News API: {'ON' if config.NEWS_API_KEY else 'OFF (no key -> neutral)'}\n")

    _, explanation = orch.explain(as_of)
    print("=" * 56)
    print(explanation)
    print("=" * 56)
    print(f"Gemini calls made: {orch.sentiment_agent.llm_calls}")
    print("\nDemo on real data for an Indian-equity portfolio. Not investment advice.")


if __name__ == "__main__":
    main()
