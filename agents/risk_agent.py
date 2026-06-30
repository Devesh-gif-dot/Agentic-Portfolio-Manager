"""Risk Agent: volatility, inverse-vol base weights, and a 0-1 stress signal."""

import numpy as np


class RiskAgent:
    def __init__(self, stocks):
        self.stocks = stocks

    def assess(self, trailing_returns):
        r = trailing_returns[self.stocks].dropna()
        vol = r.std() * np.sqrt(252)
        inv = 1.0 / vol
        base_weights = (inv / inv.sum()).to_dict()

        market = r.mean(axis=1)
        recent = market.tail(10).std()
        baseline = market.std()
        stress = float(np.clip((recent / baseline - 1.0), 0.0, 1.0)) if baseline else 0.0

        return {"vol": vol.to_dict(), "base_weights": base_weights, "stress": stress}
