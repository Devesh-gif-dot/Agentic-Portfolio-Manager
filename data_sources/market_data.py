"""
market_data.py
--------------
Fetches daily prices from Yahoo Finance (yfinance) for the configured tickers
and caches them to disk, so the backtest is reproducible and you don't refetch
on every run. yfinance needs no API key.

Returns a tidy price DataFrame: index = dates, columns = tickers.
"""

import pandas as pd
import config


def get_prices(tickers=None, start=None, end=None, refresh=False) -> pd.DataFrame:
    tickers = tickers or config.TICKERS
    start = start or config.START
    end = end or config.END
    cache_file = config.CACHE / f"prices_{start}_{end}.csv"

    if cache_file.exists() and not refresh:
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    import yfinance as yf  # imported here so the module loads without yfinance
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False)
    # yf returns a multi-index column frame; take adjusted Close
    prices = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    prices = prices.dropna(how="all").ffill().dropna()
    prices.to_csv(cache_file)
    return prices


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()
