"""
engine_sklearn.py
=================
Engine A – Scikit-Learn Huber Regression Baseline (CPU loop).

Methodology
-----------
For each asset and each rolling window of length *tau* we fit a
`sklearn.linear_model.HuberRegressor` (which internally uses IRLS with the
Huber loss).  This is the "traditional slow approach" used as a benchmark.

Outputs
-------
For every (asset, date) pair we record:
  - beta coefficients for each factor
  - t-statistics and p-values (derived from OLS covariance approximation)
  - residual standard error
"""

from __future__ import annotations

import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_sklearn_rolling(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    tau: int = 60,
    epsilon: float = 1.35,
    max_iter: int = 200,
) -> Tuple[pd.DataFrame, float]:
    """
    Run rolling Huber regression for every asset using Scikit-Learn.

    Parameters
    ----------
    asset_returns : pd.DataFrame
        Shape (T, N_assets).  Daily log-returns.
    factor_returns : pd.DataFrame
        Shape (T, K_factors).  Daily log-returns for each factor.
    tau : int
        Rolling window length in trading days.
    epsilon : float
        Huber loss threshold (default 1.35 corresponds to ~95 % efficiency).
    max_iter : int
        Maximum IRLS iterations inside HuberRegressor.

    Returns
    -------
    (results_df, elapsed_seconds)
        results_df : long-format DataFrame with columns
            [asset, date, factor, beta, t_stat, p_value, r_squared, method]
        elapsed_seconds : wall-clock time for the entire computation
    """
    assets  = asset_returns.columns.tolist()
    factors = factor_returns.columns.tolist()
    dates   = asset_returns.index
    T       = len(dates)

    records: List[Dict] = []
    t0 = time.perf_counter()

    for asset in assets:
        y_all = asset_returns[asset].values.astype(np.float64)
        X_all = factor_returns[factors].values.astype(np.float64)

        for t_end in range(tau, T + 1):
            t_start = t_end - tau
            y_win = y_all[t_start:t_end]
            X_win = X_all[t_start:t_end]
            date  = dates[t_end - 1]

            # Skip windows with too many NaNs
            mask = np.isfinite(y_win) & np.all(np.isfinite(X_win), axis=1)
            if mask.sum() < max(tau // 2, len(factors) + 2):
                continue

            y_clean = y_win[mask]
            X_clean = X_win[mask]

            try:
                model = HuberRegressor(
                    epsilon=epsilon,
                    max_iter=max_iter,
                    fit_intercept=True,
                    alpha=0.0,          # no regularisation
                )
                model.fit(X_clean, y_clean)
                betas = model.coef_            # shape (K,)

                # ---- Approximate t-stats via WLS covariance ----------------
                # Use Huber sample weights to form a WLS covariance estimate
                residuals = y_clean - model.predict(X_clean)
                abs_res   = np.abs(residuals)
                sigma_hat = np.median(abs_res) / 0.6745  # MAD estimator
                huber_w   = np.where(
                    abs_res <= epsilon * sigma_hat,
                    1.0,
                    epsilon * sigma_hat / (abs_res + 1e-12),
                )
                W   = np.diag(huber_w)
                XtW = X_clean.T @ W
                try:
                    cov_beta = np.linalg.inv(XtW @ X_clean) * (sigma_hat ** 2)
                    se_beta  = np.sqrt(np.diag(cov_beta))
                except np.linalg.LinAlgError:
                    se_beta = np.full(len(betas), np.nan)

                t_stats = betas / (se_beta + 1e-12)
                dof     = max(mask.sum() - len(factors) - 1, 1)
                p_vals  = 2 * stats.t.sf(np.abs(t_stats), df=dof)

                # R-squared
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((y_clean - y_clean.mean()) ** 2)
                r2     = 1.0 - ss_res / (ss_tot + 1e-12)

                for k, factor in enumerate(factors):
                    records.append(
                        {
                            "asset":     asset,
                            "date":      date,
                            "factor":    factor,
                            "beta":      float(betas[k]),
                            "t_stat":    float(t_stats[k]),
                            "p_value":   float(p_vals[k]),
                            "r_squared": float(r2),
                            "method":    "Scikit-Learn (CPU)",
                        }
                    )

            except Exception as exc:
                # Gracefully skip windows that fail to converge
                continue

    elapsed = time.perf_counter() - t0
    results_df = pd.DataFrame(records)
    return results_df, elapsed


def pivot_betas(results_df: pd.DataFrame, factor: str) -> pd.DataFrame:
    """
    Pivot results to a wide DataFrame: rows = dates, columns = assets,
    values = beta for a given factor.
    """
    sub = results_df[results_df["factor"] == factor]
    return sub.pivot(index="date", columns="asset", values="beta")
