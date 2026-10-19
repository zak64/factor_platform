"""
factor_catalog.py
=================
Free Factor Catalog — 30+ liquid Yahoo Finance tickers spanning:
  - Market / Broad Beta
  - Style (Value, Growth, Momentum, Quality, Low-Vol, Size)
  - Sector (11 GICS sectors via SPDR ETFs)
  - Thematic (AI/Tech, Clean Energy, Semiconductors, Biotech, Fintech, Cybersecurity)
  - Macro (Rates, Credit, Commodities, FX)
  - Alternatives (Real Estate, Infrastructure, Volatility)

Design principles
-----------------
* All tickers are free via yfinance (no paid data required).
* The catalog is structured as a list of dicts so the UI can filter by theme.
* `fetch_catalog_returns()` downloads adjusted-close prices and returns a
  clean DataFrame of daily log-returns, aligned on a common date index.
* Missing tickers are silently dropped with a warning so the engine degrades
  gracefully when a ticker is delisted or temporarily unavailable.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    warnings.warn("yfinance not installed; catalog data unavailable.")


# ---------------------------------------------------------------------------
# Catalog definition
# ---------------------------------------------------------------------------

FACTOR_CATALOG: List[Dict] = [
    # ── Market / Broad Beta ──────────────────────────────────────────────
    {"ticker": "SPY",  "name": "S&P 500",              "theme": "Market"},
    {"ticker": "QQQ",  "name": "Nasdaq-100",            "theme": "Market"},
    {"ticker": "IWM",  "name": "Russell 2000 (Small)",  "theme": "Market"},
    {"ticker": "VT",   "name": "Global All-World",      "theme": "Market"},

    # ── Style ────────────────────────────────────────────────────────────
    {"ticker": "IVW",  "name": "S&P 500 Growth",        "theme": "Style"},
    {"ticker": "IVE",  "name": "S&P 500 Value",         "theme": "Style"},
    {"ticker": "MTUM", "name": "MSCI USA Momentum",     "theme": "Style"},
    {"ticker": "QUAL", "name": "MSCI USA Quality",      "theme": "Style"},
    {"ticker": "USMV", "name": "MSCI USA Min-Vol",      "theme": "Style"},
    {"ticker": "SIZE", "name": "MSCI USA Size Factor",  "theme": "Style"},

    # ── Sector (SPDR) ────────────────────────────────────────────────────
    {"ticker": "XLK",  "name": "Technology",            "theme": "Sector"},
    {"ticker": "XLF",  "name": "Financials",            "theme": "Sector"},
    {"ticker": "XLE",  "name": "Energy",                "theme": "Sector"},
    {"ticker": "XLV",  "name": "Health Care",           "theme": "Sector"},
    {"ticker": "XLI",  "name": "Industrials",           "theme": "Sector"},
    {"ticker": "XLY",  "name": "Consumer Discretionary","theme": "Sector"},
    {"ticker": "XLP",  "name": "Consumer Staples",      "theme": "Sector"},
    {"ticker": "XLU",  "name": "Utilities",             "theme": "Sector"},
    {"ticker": "XLRE", "name": "Real Estate",           "theme": "Sector"},
    {"ticker": "XLB",  "name": "Materials",             "theme": "Sector"},
    {"ticker": "XLC",  "name": "Communication Services","theme": "Sector"},

    # ── Thematic ─────────────────────────────────────────────────────────
    {"ticker": "SMH",  "name": "Semiconductors",        "theme": "Thematic"},
    {"ticker": "SOXX", "name": "Philadelphia Semi",     "theme": "Thematic"},
    {"ticker": "ARKK", "name": "ARK Innovation",        "theme": "Thematic"},
    {"ticker": "ICLN", "name": "Clean Energy",          "theme": "Thematic"},
    {"ticker": "LIT",  "name": "Lithium & Battery",     "theme": "Thematic"},
    {"ticker": "IBB",  "name": "Biotech",               "theme": "Thematic"},
    {"ticker": "FINX", "name": "Fintech",               "theme": "Thematic"},
    {"ticker": "CIBR", "name": "Cybersecurity",         "theme": "Thematic"},

    # ── Macro ────────────────────────────────────────────────────────────
    {"ticker": "TLT",  "name": "20Y Treasury",          "theme": "Macro"},
    {"ticker": "IEF",  "name": "7-10Y Treasury",        "theme": "Macro"},
    {"ticker": "HYG",  "name": "High Yield Credit",     "theme": "Macro"},
    {"ticker": "GLD",  "name": "Gold",                  "theme": "Macro"},
    {"ticker": "USO",  "name": "Crude Oil",             "theme": "Macro"},
    {"ticker": "UUP",  "name": "USD Index",             "theme": "Macro"},
    {"ticker": "DBC",  "name": "Broad Commodities",     "theme": "Macro"},

    # ── Alternatives ─────────────────────────────────────────────────────
    {"ticker": "VNQ",  "name": "US REITs",              "theme": "Alternatives"},
    {"ticker": "IFRA", "name": "US Infrastructure",     "theme": "Alternatives"},
    {"ticker": "VIXY", "name": "VIX Short-Term Futures","theme": "Alternatives"},
]

# Ordered list of all tickers for convenience
ALL_TICKERS: List[str] = [f["ticker"] for f in FACTOR_CATALOG]

# Theme → ticker mapping
THEMES: Dict[str, List[str]] = {}
for _f in FACTOR_CATALOG:
    THEMES.setdefault(_f["theme"], []).append(_f["ticker"])

# Ticker → metadata mapping
TICKER_META: Dict[str, Dict] = {f["ticker"]: f for f in FACTOR_CATALOG}


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_catalog_returns(
    tickers: Optional[List[str]] = None,
    lookback_years: float = 5.0,
    min_history_frac: float = 0.75,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Download adjusted-close prices for *tickers* and return daily log-returns.

    Parameters
    ----------
    tickers           : list of tickers to fetch (defaults to ALL_TICKERS)
    lookback_years    : how many years of history to download
    min_history_frac  : drop tickers with fewer than this fraction of valid rows

    Returns
    -------
    (returns_df, dropped_tickers)
    returns_df : pd.DataFrame  shape (T, N)  — daily log-returns, NaN-filled
    dropped_tickers : list of tickers silently dropped due to insufficient data
    """
    if not YF_AVAILABLE:
        raise RuntimeError("yfinance is required for catalog data ingestion.")

    if tickers is None:
        tickers = ALL_TICKERS

    end   = pd.Timestamp.today()
    start = end - pd.DateOffset(years=lookback_years)

    # Batch download — faster than one-by-one
    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # Extract Close prices; handle both single and multi-ticker DataFrames
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers[:1]

    # Compute log-returns
    log_ret = np.log(prices / prices.shift(1)).iloc[1:]

    # Drop tickers with insufficient history
    T = len(log_ret)
    min_obs = int(T * min_history_frac)
    valid_counts = log_ret.notna().sum()
    good = valid_counts[valid_counts >= min_obs].index.tolist()
    dropped = [t for t in tickers if t not in good]

    if dropped:
        warnings.warn(
            f"Dropped {len(dropped)} tickers due to insufficient history: {dropped}"
        )

    returns_df = log_ret[good].copy()
    # Forward-fill then back-fill minor gaps (e.g. ETF inception date)
    returns_df = returns_df.fillna(method="ffill", limit=3).fillna(method="bfill", limit=3)
    # Any remaining NaN → 0 (non-trading day)
    returns_df = returns_df.fillna(0.0)

    return returns_df, dropped


def get_catalog_metadata(tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """Return a DataFrame of ticker metadata for display in the UI."""
    if tickers is None:
        tickers = ALL_TICKERS
    rows = [TICKER_META.get(t, {"ticker": t, "name": t, "theme": "Custom"})
            for t in tickers]
    return pd.DataFrame(rows).set_index("ticker")


def get_theme_colors() -> Dict[str, str]:
    """Return a consistent colour palette for each theme (for Plotly charts)."""
    return {
        "Market":       "#1f77b4",
        "Style":        "#ff7f0e",
        "Sector":       "#2ca02c",
        "Thematic":     "#d62728",
        "Macro":        "#9467bd",
        "Alternatives": "#8c564b",
        "Custom":       "#7f7f7f",
    }
