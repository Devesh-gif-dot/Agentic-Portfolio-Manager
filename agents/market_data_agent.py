"""Market Data Agent: serves price/return history as-of a date (no look-ahead)."""

import pandas as pd


class MarketDataAgent:
    def __init__(self, prices: pd.DataFrame, returns: pd.DataFrame):
        self.prices = prices
        self.returns = returns

    def returns_until(self, as_of, lookback_days=126):
        hist = self.returns[self.returns.index < pd.Timestamp(as_of)]
        return hist.tail(lookback_days)

    def prices_until(self, as_of):
        return self.prices[self.prices.index < pd.Timestamp(as_of)]
