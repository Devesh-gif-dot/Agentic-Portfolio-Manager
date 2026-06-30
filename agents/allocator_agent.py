"""
Allocator Agent: combines the agents' outputs into final weights (sum = 100%).
Toggles let the ablation isolate each capability.
  use_risk      -> inverse-vol base weights vs equal weight
  use_sentiment -> tilt each stock by (sentiment - 5)
  use_hedge     -> shift part of the book into the gold ETF, sized by stress
"""

import numpy as np


class AllocatorAgent:
    def __init__(self, stocks, gold, use_risk=True, use_sentiment=True,
                 use_hedge=True, tilt_strength=0.6, max_hedge=0.40, max_weight=0.30):
        self.stocks = stocks
        self.gold = gold
        self.use_risk = use_risk
        self.use_sentiment = use_sentiment
        self.use_hedge = use_hedge
        self.tilt_strength = tilt_strength
        self.max_hedge = max_hedge
        self.max_weight = max_weight

    def allocate(self, risk, sentiment):
        if self.use_risk:
            w = dict(risk["base_weights"])
        else:
            w = {s: 1.0 / len(self.stocks) for s in self.stocks}

        if self.use_sentiment and sentiment:
            for s in self.stocks:
                factor = 1.0 + self.tilt_strength * (sentiment.get(s, 5.0) - 5.0) / 5.0
                w[s] *= max(factor, 0.0)
            tot = sum(w.values()) or 1.0
            w = {s: v / tot for s, v in w.items()}

        w = {s: min(v, self.max_weight) for s, v in w.items()}
        tot = sum(w.values()) or 1.0
        w = {s: v / tot for s, v in w.items()}

        gold_w = 0.0
        if self.use_hedge:
            gold_w = float(np.clip(self.max_hedge * risk["stress"], 0.0, self.max_hedge))
            w = {s: v * (1.0 - gold_w) for s, v in w.items()}
        w[self.gold] = gold_w
        return w
