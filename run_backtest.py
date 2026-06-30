"""
run_backtest.py
---------------
Walk-forward backtest + agent value-add study on REAL yfinance price history.

Study A (price-driven, fully reproducible from free data): build the system up
one agent at a time -- Baseline -> +Risk -> +Hedge -- and compare CAGR, Sharpe,
and max drawdown.

Sentiment in the long backtest: free news APIs only return recent news, so a
multi-year sentiment ablation needs historical news (paid, or a cache you build
up). Set RUN_SENTIMENT=1 to also score sentiment over whatever news is cached;
otherwise Study A runs price-only and is fully reproducible today.

    python run_backtest.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from data_sources import market_data, news_client
from orchestrator import Orchestrator
from backtest import run_backtest


def main():
    prices = market_data.get_prices()
    returns = market_data.to_returns(prices)
    stocks, companies, gold = list(config.STOCKS), config.STOCKS, config.GOLD

    run_sentiment = os.getenv("RUN_SENTIMENT") == "1"
    configs = {
        "Baseline (equal weight)": dict(use_risk=False, use_sentiment=False, use_hedge=False),
        "+ Risk agent":            dict(use_risk=True,  use_sentiment=False, use_hedge=False),
        "+ Hedge agent":           dict(use_risk=True,  use_sentiment=False, use_hedge=True),
    }
    if run_sentiment:
        configs["+ Sentiment (full)"] = dict(use_risk=True, use_sentiment=True, use_hedge=True)

    rows, curves = [], {}
    for name, flags in configs.items():
        orch = Orchestrator(prices, returns, news_client.get_news,
                            stocks, companies, gold, use_llm=run_sentiment, **flags)
        curve, m, _ = run_backtest(orch, prices, returns)
        curves[name] = curve
        rows.append({"Configuration": name,
                     "CAGR %": round(m["CAGR"] * 100, 2),
                     "Sharpe": round(m["Sharpe"], 2),
                     "MaxDD %": round(m["MaxDrawdown"] * 100, 2),
                     "Rs1 ->": round(m["FinalValue"], 2)})
    table = pd.DataFrame(rows)
    table.to_csv(config.OUTPUTS / "ablation_results.csv", index=False)

    print("=" * 70)
    print(f"AGENT VALUE-ADD  (NSE large-caps, {config.START}..{config.END}, monthly)")
    print("=" * 70)
    print(table.to_string(index=False))

    plt.figure(figsize=(9, 5))
    for name, c in curves.items():
        plt.plot(c.index, c.values, label=name, linewidth=1.6)
    plt.title("Equity curves by configuration (Rs 1 start)")
    plt.ylabel("Portfolio value"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(config.OUTPUTS / "equity_curves.png", dpi=110)
    print(f"\nSaved outputs/ablation_results.csv and outputs/equity_curves.png")
    if not run_sentiment:
        print("\n(Sentiment ablation skipped: set RUN_SENTIMENT=1 with news cached "
              "to include it. See README on the historical-news limitation.)")


if __name__ == "__main__":
    main()
