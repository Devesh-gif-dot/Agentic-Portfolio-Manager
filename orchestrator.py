"""
Orchestrator: coordinates the agents for a given date and returns final weights
plus an explanation. Same structure as before -- the agents changed data
sources, but the wiring did not.
"""

import pandas as pd
from agents import MarketDataAgent, RiskAgent, SentimentAgent, AllocatorAgent


class Orchestrator:
    def __init__(self, prices, returns, news_fn, stocks, companies, gold, *,
                 use_risk=True, use_sentiment=True, use_hedge=True, use_llm=True):
        self.stocks = stocks
        self.gold = gold
        self.market = MarketDataAgent(prices, returns)
        self.risk = RiskAgent(stocks)
        self.sentiment_agent = (
            SentimentAgent(news_fn, stocks, companies, use_llm=use_llm)
            if use_sentiment else None
        )
        self.allocator = AllocatorAgent(stocks, gold, use_risk=use_risk,
                                        use_sentiment=use_sentiment, use_hedge=use_hedge)

    def build_portfolio(self, as_of):
        trailing = self.market.returns_until(as_of)
        risk_view = self.risk.assess(trailing)
        sentiment = self.sentiment_agent.score(as_of) if self.sentiment_agent else None
        weights = self.allocator.allocate(risk_view, sentiment)
        return weights, {"risk": risk_view, "sentiment": sentiment}

    def explain(self, as_of):
        w, info = self.build_portfolio(as_of)
        lines = [f"Recommended portfolio as of {pd.Timestamp(as_of).date()}:"]
        for t, v in sorted(w.items(), key=lambda x: -x[1]):
            tag = ""
            if info["sentiment"] and t in info["sentiment"]:
                tag = f"   sentiment {info['sentiment'][t]:.1f}/10"
            lines.append(f"  {t:<13} {v*100:5.1f}%{tag}")
        lines.append(f"  (market stress signal: {info['risk']['stress']:.2f})")
        return w, "\n".join(lines)
