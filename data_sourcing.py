"""
data_sourcing.py
================
Handles all data ingestion for the Financial Factor Exposure Platform.

Responsibilities:
  - Fetch historical adjusted-close prices via yfinance
  - Compute daily log-returns
  - Provide a hardcoded factor-model dictionary (free Yahoo Finance proxies)
  - Optionally query arXiv for recent factor-model papers
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Factor Model Registry
# ---------------------------------------------------------------------------
# Maps human-readable model names to lists of (ticker, label) pairs.
# All tickers are freely available on Yahoo Finance.

FACTOR_MODELS: Dict[str, List[Tuple[str, str]]] = {
    "Market Only (CAPM)": [
        ("SPY", "Market (SPY)"),
    ],
    "Fama-French 3-Factor Proxy": [
        ("SPY",  "Market (SPY)"),
        ("IWM",  "Size – Small Cap (IWM)"),
        ("IWD",  "Value (IWD)"),
    ],
    "Fama-French 5-Factor Proxy": [
        ("SPY",  "Market (SPY)"),
        ("IWM",  "Size – Small Cap (IWM)"),
        ("IWD",  "Value (IWD)"),
        ("MTUM", "Profitability/Momentum (MTUM)"),
        ("QUAL", "Investment Quality (QUAL)"),
    ],
    "Macro Model (Oil / Gold / Rates)": [
        ("SPY",  "Market (SPY)"),
        ("USO",  "Oil (USO)"),
        ("GLD",  "Gold (GLD)"),
        ("TLT",  "Long-Duration Rates (TLT)"),
    ],
    "Thematic AI / Semiconductor Model": [
        ("SPY",  "Market (SPY)"),
        ("SMH",  "Semiconductors / AI (SMH)"),
        ("SOXX", "Broad Semis (SOXX)"),
        ("ARKK", "Disruptive Innovation (ARKK)"),
    ],
    "Energy Transition Model": [
        ("SPY",  "Market (SPY)"),
        ("XLE",  "Energy (XLE)"),
        ("ICLN", "Clean Energy (ICLN)"),
        ("LIT",  "Lithium / Battery (LIT)"),
    ],
    "Multi-Asset Macro Model": [
        ("SPY",  "Market (SPY)"),
        ("GLD",  "Gold (GLD)"),
        ("USO",  "Oil (USO)"),
        ("TLT",  "Long-Duration Rates (TLT)"),
        ("UUP",  "USD Index (UUP)"),
        ("VNQ",  "Real Estate (VNQ)"),
    ],
}


# ---------------------------------------------------------------------------
# Price / Return Fetching
# ---------------------------------------------------------------------------

def fetch_returns(
    tickers: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    lookback_years: int = 5,
) -> pd.DataFrame:
    """
    Download adjusted-close prices for *tickers* and return daily log-returns.

    Parameters
    ----------
    tickers : list of str
        Yahoo Finance ticker symbols.
    start : str, optional
        ISO date string 'YYYY-MM-DD'.  Defaults to *lookback_years* ago.
    end : str, optional
        ISO date string 'YYYY-MM-DD'.  Defaults to today.
    lookback_years : int
        Used only when *start* is None.

    Returns
    -------
    pd.DataFrame
        Daily log-returns indexed by date, one column per ticker.
        NaN rows (e.g. first row) are dropped.
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.today() - timedelta(days=365 * lookback_years)).strftime(
            "%Y-%m-%d"
        )

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yfinance returns MultiIndex columns when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Ensure all requested tickers are present (fill missing with NaN)
    for t in tickers:
        if t not in prices.columns:
            prices[t] = np.nan
            logger.warning("Ticker %s not found; filled with NaN.", t)

    prices = prices[tickers]
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns


def fetch_factor_data(
    model_name: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    lookback_years: int = 5,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Fetch returns for all factors in a named model.

    Returns
    -------
    (returns_df, factor_labels)
        returns_df : DataFrame with one column per factor ticker
        factor_labels : human-readable names matching the columns
    """
    factor_spec = FACTOR_MODELS[model_name]
    tickers = [t for t, _ in factor_spec]
    labels  = [l for _, l in factor_spec]
    df = fetch_returns(tickers, start=start, end=end, lookback_years=lookback_years)
    return df, labels


def align_returns(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inner-join asset and factor returns on the date index so they share
    exactly the same trading days.
    """
    common = asset_returns.index.intersection(factor_returns.index)
    return asset_returns.loc[common], factor_returns.loc[common]


# ---------------------------------------------------------------------------
# Portfolio Persistence
# ---------------------------------------------------------------------------

PORTFOLIO_DIR = Path(__file__).parent / "portfolios"
PORTFOLIO_DIR.mkdir(exist_ok=True)


def save_portfolio(name: str, tickers: List[str], weights: List[float]) -> Path:
    """Persist a portfolio to JSON."""
    path = PORTFOLIO_DIR / f"{name}.json"
    payload = {
        "name": name,
        "tickers": tickers,
        "weights": weights,
        "saved_at": datetime.utcnow().isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_portfolio(name: str) -> Dict:
    """Load a previously saved portfolio from JSON."""
    path = PORTFOLIO_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Portfolio '{name}' not found at {path}")
    return json.loads(path.read_text())


def list_portfolios() -> List[str]:
    """Return names of all saved portfolios."""
    return [p.stem for p in sorted(PORTFOLIO_DIR.glob("*.json"))]


# ---------------------------------------------------------------------------
# arXiv Factor-Model Paper Search Agent
# ---------------------------------------------------------------------------

ARXIV_API = "https://export.arxiv.org/api/query"


def search_arxiv_papers(
    query: str = "factor model portfolio construction thematic",
    max_results: int = 5,
) -> List[Dict]:
    """
    Query the arXiv public API for recent papers related to factor models.

    Returns
    -------
    list of dict with keys: title, authors, summary, link, published
    """
    params = {
        "search_query": f"all:{query}",
        "sortBy": "relevance",
        "sortOrder": "descending",
        "max_results": max_results,
    }
    try:
        resp = requests.get(ARXIV_API, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("arXiv query failed: %s", exc)
        return []

    import xml.etree.ElementTree as ET
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    papers = []
    for entry in root.findall("atom:entry", ns):
        title   = entry.findtext("atom:title",   default="", namespaces=ns).strip()
        summary = entry.findtext("atom:summary", default="", namespaces=ns).strip()
        link    = entry.findtext("atom:id",      default="", namespaces=ns).strip()
        pub     = entry.findtext("atom:published", default="", namespaces=ns).strip()
        authors = [
            a.findtext("atom:name", default="", namespaces=ns)
            for a in entry.findall("atom:author", ns)
        ]
        papers.append(
            {
                "title":     title,
                "authors":   authors,
                "summary":   summary[:400] + ("…" if len(summary) > 400 else ""),
                "link":      link,
                "published": pub[:10],
            }
        )
    return papers
