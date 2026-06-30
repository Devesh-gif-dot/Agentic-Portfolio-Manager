"""
backtest.py
-----------
Walk-forward, out-of-sample backtest. At each monthly rebalance we ask the
orchestrator for weights using only past data, hold to the next rebalance, and
chain daily returns into an equity curve. Reports CAGR, vol, Sharpe, max DD.
"""

import numpy as np
import pandas as pd


def _metrics(equity):
    rets = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = equity.iloc[-1] ** (1 / years) - 1
    vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() * 252) / vol if vol else 0.0
    mdd = (equity / equity.cummax() - 1).min()
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe,
            "MaxDrawdown": mdd, "FinalValue": equity.iloc[-1]}


def run_backtest(orchestrator, prices, returns, warmup=126):
    rebal = pd.Series(prices.index).groupby(
        [prices.index.year, prices.index.month]).first().tolist()
    rebal = [pd.Timestamp(d) for d in rebal if d > prices.index[warmup]]

    equity, idx, wlog = [1.0], [rebal[0]], {}
    for i, start in enumerate(rebal):
        end = rebal[i + 1] if i + 1 < len(rebal) else prices.index[-1]
        weights, _ = orchestrator.build_portfolio(start)
        wlog[start] = weights
        period = returns[(returns.index >= start) & (returns.index < end)]
        w = pd.Series(weights).reindex(returns.columns).fillna(0.0)
        for d, r in (period[returns.columns] @ w).items():
            equity.append(equity[-1] * (1 + r)); idx.append(d)

    curve = pd.Series(equity[1:], index=idx[1:])
    return curve, _metrics(curve), wlog
