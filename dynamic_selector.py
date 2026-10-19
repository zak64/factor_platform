"""
dynamic_selector.py
===================
High-level integration module for the Dynamic Factor Selector (Section 5).

This module is the single entry point called by the Streamlit UI.  It
orchestrates:
  1. Factor catalog ingestion (factor_catalog.py)
  2. D-Guarded Lasso Path (lasso_guard.py)
  3. Recursive State EWMA + MILP Knapsack (knapsack_optimizer.py)
  4. Produces all DataFrames needed by the UI charts

Public API
----------
run_dynamic_selection(asset_ticker, params) → DynamicSelectionResult
build_factor_radar_df(selection_df, meta_df) → pd.DataFrame
build_lasso_path_df(lasso_result)           → pd.DataFrame
"""

from __future__ import annotations

import sys
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure the platform directory is on the path when called from Streamlit
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from factor_catalog import (
    fetch_catalog_returns,
    get_catalog_metadata,
    get_theme_colors,
    FACTOR_CATALOG,
    ALL_TICKERS,
    THEMES,
)
from lasso_guard import dguarded_lasso_path, rolling_dguarded_lasso
from knapsack_optimizer import (
    RecursiveStateTracker,
    knapsack_milp,
    compute_rolling_tstats,
    run_knapsack_selection,
)


# ---------------------------------------------------------------------------
# Parameters dataclass
# ---------------------------------------------------------------------------

@dataclass
class DynamicSelectionParams:
    """All hyper-parameters exposed in the Streamlit sidebar."""
    lookback_years:  float = 3.0
    tau:             int   = 60       # rolling window for t-stats
    delta_max:       float = 2.0      # D-guard multicollinearity threshold
    gamma:           float = 1.0      # turnover penalty
    k_max:           int   = 5        # max active factors
    ewma_halflife:   int   = 21       # EWMA decay half-life (trading days)
    rebal_step:      int   = 5        # rebalancing frequency (trading days)
    candidate_themes: List[str] = field(
        default_factory=lambda: list(THEMES.keys())
    )
    custom_tickers:  List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DynamicSelectionResult:
    """All outputs produced by run_dynamic_selection."""
    # Core selection outputs
    selection_df:    pd.DataFrame   # (n_rebal, P) binary matrix
    state_df:        pd.DataFrame   # final EWMA state per factor
    tstat_df:        pd.DataFrame   # (T, P) rolling t-statistics

    # Lasso path outputs (last window only, for the path chart)
    lasso_result:    Dict

    # Factor Radar data
    radar_df:        pd.DataFrame   # (n_rebal, theme) — active factor count per theme
    inclusion_df:    pd.DataFrame   # (n_rebal, P) — cumulative inclusion fraction

    # Metadata
    factor_meta:     pd.DataFrame   # ticker → name, theme
    theme_colors:    Dict[str, str]
    factors:         List[str]
    params:          DynamicSelectionParams

    # Timing
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_dynamic_selection(
    asset_ticker: str,
    asset_returns: pd.Series,
    params: DynamicSelectionParams,
) -> DynamicSelectionResult:
    """
    Run the full Dynamic Factor Selection pipeline.

    Parameters
    ----------
    asset_ticker  : str            — label for the target asset
    asset_returns : pd.Series      — daily log-returns of the target asset
    params        : DynamicSelectionParams

    Returns
    -------
    DynamicSelectionResult
    """
    import time
    t0 = time.perf_counter()

    # ── 1. Build candidate ticker list ──────────────────────────────────
    candidate_tickers: List[str] = []
    for theme in params.candidate_themes:
        candidate_tickers.extend(THEMES.get(theme, []))
    # Add custom tickers
    for t in params.custom_tickers:
        t = t.strip().upper()
        if t and t not in candidate_tickers:
            candidate_tickers.append(t)
    # Remove duplicates, preserve order
    seen: set = set()
    candidate_tickers = [
        t for t in candidate_tickers
        if not (t in seen or seen.add(t))
    ]

    # ── 2. Fetch factor returns ──────────────────────────────────────────
    factor_ret, dropped = fetch_catalog_returns(
        tickers=candidate_tickers,
        lookback_years=params.lookback_years,
    )
    if dropped:
        warnings.warn(f"Dropped tickers: {dropped}")

    # Align asset returns to factor return dates
    common_idx  = asset_returns.index.intersection(factor_ret.index)
    y           = asset_returns.loc[common_idx]
    X           = factor_ret.loc[common_idx]
    factors     = X.columns.tolist()

    # ── 3. Rolling t-statistics ──────────────────────────────────────────
    tstat_df = compute_rolling_tstats(X, y, tau=params.tau)

    # ── 4. Full knapsack selection loop ──────────────────────────────────
    selection_df, state_df, tracker = run_knapsack_selection(
        X              = X,
        y              = y,
        delta_max      = params.delta_max,
        gamma          = params.gamma,
        k_max          = params.k_max,
        tau            = params.tau,
        ewma_halflife  = params.ewma_halflife,
        rebal_step     = params.rebal_step,
    )

    # ── 5. Lasso path on the last window (for the path chart) ────────────
    T = len(X)
    last_start = max(0, T - params.tau)
    try:
        lasso_result = dguarded_lasso_path(
            X.iloc[last_start:],
            y.iloc[last_start:],
            delta_max=params.delta_max,
        )
    except Exception as e:
        lasso_result = {
            "selected_factors": [],
            "selected_alpha":   np.nan,
            "selected_delta":   np.nan,
            "path_df":          pd.DataFrame(),
            "coef_path":        np.array([]),
            "alphas":           np.array([]),
            "factor_names":     factors,
        }

    # ── 6. Build Factor Radar DataFrame ──────────────────────────────────
    meta_df     = get_catalog_metadata(factors)
    radar_df    = build_factor_radar_df(selection_df, meta_df)
    inclusion_df = selection_df.expanding().mean()  # cumulative inclusion rate

    elapsed = time.perf_counter() - t0

    return DynamicSelectionResult(
        selection_df    = selection_df,
        state_df        = state_df,
        tstat_df        = tstat_df,
        lasso_result    = lasso_result,
        radar_df        = radar_df,
        inclusion_df    = inclusion_df,
        factor_meta     = meta_df,
        theme_colors    = get_theme_colors(),
        factors         = factors,
        params          = params,
        elapsed_seconds = elapsed,
    )


# ---------------------------------------------------------------------------
# Chart data builders
# ---------------------------------------------------------------------------

def build_factor_radar_df(
    selection_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate the binary selection matrix into a per-theme active-factor
    count time-series, suitable for a stacked area chart (Factor Radar).

    Returns
    -------
    pd.DataFrame  index=date, columns=themes, values=count of active factors
    """
    themes_present = meta_df["theme"].unique().tolist()
    records = []
    for date, row in selection_df.iterrows():
        rec = {"date": date}
        for theme in themes_present:
            theme_tickers = meta_df[meta_df["theme"] == theme].index.tolist()
            active_in_theme = sum(
                int(row.get(t, 0)) for t in theme_tickers if t in row.index
            )
            rec[theme] = active_in_theme
        records.append(rec)
    return pd.DataFrame(records).set_index("date")


def build_lasso_path_chart_df(lasso_result: Dict) -> pd.DataFrame:
    """
    Return the Lasso path diagnostics in a form ready for Plotly.

    Columns: log_alpha, n_active, delta, guard_pass
    """
    path_df = lasso_result.get("path_df", pd.DataFrame())
    if path_df.empty:
        return path_df
    df = path_df.copy()
    df["log_alpha"] = np.log10(df["alpha"].clip(lower=1e-10))
    return df[["log_alpha", "n_active", "delta", "guard_pass", "active_set"]]


def build_coef_heatmap_df(lasso_result: Dict) -> pd.DataFrame:
    """
    Return a tidy DataFrame of Lasso coefficients along the path for a
    coefficient-path heatmap.

    Columns: log_alpha, factor, coefficient
    """
    coefs   = lasso_result.get("coef_path", np.array([]))
    alphas  = lasso_result.get("alphas",    np.array([]))
    factors = lasso_result.get("factor_names", [])

    if coefs.size == 0:
        return pd.DataFrame()

    records = []
    for i, alpha in enumerate(alphas):
        log_a = float(np.log10(max(alpha, 1e-10)))
        for j, f in enumerate(factors):
            records.append({
                "log_alpha":   log_a,
                "factor":      f,
                "coefficient": float(coefs[j, i]),
            })
    return pd.DataFrame(records)
