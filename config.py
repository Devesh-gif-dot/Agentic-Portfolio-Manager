"""
config.py
---------
One place for everything you might want to change: the stock universe, the gold
hedge, which LLM/news provider to use, and where API keys come from.

Keys are read from environment variables (load a .env file via python-dotenv).
Never hard-code keys. Copy .env.example to .env and fill it in.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads .env if present

# ---- API keys (set these in .env) ------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# ---- model / provider choices ----------------------------------------------
# gemini-2.5-flash-lite has the most generous free quota (1,000 req/day).
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
NEWS_PROVIDER = os.getenv("NEWS_PROVIDER", "marketaux")  # marketaux | newsdata

# ---- the portfolio universe (NSE large-caps across sectors) ----------------
# yfinance uses ".NS" for NSE and ".BO" for BSE. company name is used for news.
STOCKS = {
    "RELIANCE.NS":  "Reliance Industries",
    "TCS.NS":       "Tata Consultancy Services",
    "INFY.NS":      "Infosys",
    "HDFCBANK.NS":  "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "HINDUNILVR.NS":"Hindustan Unilever",
    "ITC.NS":       "ITC",
    "LT.NS":        "Larsen & Toubro",
    "SUNPHARMA.NS": "Sun Pharma",
    "MARUTI.NS":    "Maruti Suzuki",
}
GOLD = "GOLDBEES.NS"        # Nippon India Gold ETF (the hedge)
BENCHMARK = "^NSEI"          # Nifty 50, for reference

TICKERS = list(STOCKS) + [GOLD]

# ---- paths -----------------------------------------------------------------
ROOT = Path(__file__).parent
CACHE = ROOT / "cache"
OUTPUTS = ROOT / "outputs"
CACHE.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

# ---- backtest window -------------------------------------------------------
START = os.getenv("BACKTEST_START", "2019-01-01")
END = os.getenv("BACKTEST_END", "2024-12-31")
