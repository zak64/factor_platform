"""
lasso_guard.py
==============
D-Guarded Lasso Path Algorithm
--------------------------------
Implements the multicollinearity-guarded regularisation path for factor
selection, as described in Section 5.2 of the specification.

Algorithm
---------
1. Standardise the candidate factor return matrix X (T × P) and the asset
   return vector y (T,).
2. Compute the full Lasso regularisation path using
   ``sklearn.linear_model.lasso_path``.  This produces a sequence of
   coefficient matrices indexed by decreasing λ.
3. At each step λ_i, identify the **active set** A_i — the columns of X
   with non-zero coefficients.
4. Compute the **D-guard statistic**:

       δ_λ = −ln det ρ(X_{A_λ})

   where ρ(·) is the Pearson correlation matrix of the active columns.
   When |A_λ| = 0 or 1, δ_λ is defined as 0 (no multicollinearity possible).

5. **Filter**: retain only steps where δ_λ ≤ δ_max.  Among the surviving
   steps, select the one with the most active factors (i.e., the most
   parsimonious model that still satisfies the guard).

6. Return the selected active-set together with the full path diagnostics
   for visualisation.

Notes on numerical stability
-----------------------------
* ``lasso_path`` is called with ``method='lars'`` (LARS-Lasso) which is
  exact and numerically stable.
* The log-det is computed via ``numpy.linalg.slogdet`` to avoid overflow.
* A small ridge regularisation (ε = 1e-6) is added to the correlation
  matrix before computing the log-det to handle near-singular cases.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import lasso_path
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def compute_log_det(X_active: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute δ = −ln det ρ(X_active) for an active factor sub-matrix.

    Parameters
    ----------
    X_active : np.ndarray  shape (T, k)  — raw or standardised returns
    eps      : ridge added to diagonal for numerical stability

    Returns
    -------
    delta : float  (0.0 when k ≤ 1)
    """
    k = X_active.shape[1]
    if k <= 1:
        return 0.0

    # Pearson correlation matrix
    corr = np.corrcoef(X_active, rowvar=False)
    # Add ridge to diagonal
    corr = corr + eps * np.eye(k)

    sign, logabsdet = np.linalg.slogdet(corr)
    if sign <= 0:
        # Numerically degenerate — treat as maximally collinear
        return np.inf
    return float(-logabsdet)


def dguarded_lasso_path(
    X: pd.DataFrame,
    y: pd.Series,
    delta_max: float = 2.0,
    n_alphas: int = 100,
    eps_path: float = 1e-3,
) -> Dict:
    """
    Run the D-Guarded Lasso Path and return the selected factor set.

    Parameters
    ----------
    X         : pd.DataFrame  shape (T, P)  — candidate factor returns
    y         : pd.Series     shape (T,)    — asset returns
    delta_max : float  — maximum allowed log-det (multicollinearity guard)
    n_alphas  : int    — number of regularisation steps
    eps_path  : float  — ratio of min/max alpha for the path

    Returns
    -------
    result dict with keys:
      "selected_factors"  : List[str]  — tickers of the chosen active set
      "selected_alpha"    : float      — λ value at the chosen step
      "selected_delta"    : float      — δ value at the chosen step
      "path_df"           : pd.DataFrame — full path diagnostics
          columns: alpha, n_active, delta, guard_pass, active_set (str)
      "coef_path"         : np.ndarray  shape (P, n_alphas)
      "alphas"            : np.ndarray  shape (n_alphas,)
    """
    factors = X.columns.tolist()
    P = len(factors)

    # Standardise
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    Xs = scaler_X.fit_transform(X.values)
    ys = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

    # Lasso path — coordinate descent (sklearn 1.9+)
    # Note: method='lars' and fit_intercept are not accepted by lasso_path;
    # the function always uses coordinate descent and centres internally.
    alphas, coefs, _ = lasso_path(
        Xs, ys,
        alphas=n_alphas,   # integer: number of alphas on the path
        eps=eps_path,
    )
    # coefs shape: (P, n_alphas)  — each column is the coefficient vector at one α

    # ── Build path diagnostics ──────────────────────────────────────────
    path_records = []
    for i, alpha in enumerate(alphas):
        coef_i   = coefs[:, i]
        active   = [factors[j] for j in range(P) if coef_i[j] != 0.0]
        n_active = len(active)

        if n_active == 0:
            delta = 0.0
        else:
            X_active = Xs[:, [factors.index(f) for f in active]]
            delta    = compute_log_det(X_active)

        guard_pass = (delta <= delta_max)
        path_records.append(
            {
                "alpha":      float(alpha),
                "n_active":   n_active,
                "delta":      float(delta) if np.isfinite(delta) else 999.0,
                "guard_pass": guard_pass,
                "active_set": ",".join(active),
            }
        )

    path_df = pd.DataFrame(path_records)

    # ── Select the best surviving step ──────────────────────────────────
    # Among guard-passing steps, pick the one with the most active factors
    # (i.e., least regularised model that still satisfies the D-guard).
    passing = path_df[path_df["guard_pass"]]

    if passing.empty:
        # Fallback: pick the step with the smallest delta regardless
        best_idx = path_df["delta"].idxmin()
    else:
        best_idx = passing["n_active"].idxmax()

    best_row          = path_df.loc[best_idx]
    selected_factors  = best_row["active_set"].split(",") if best_row["active_set"] else []
    selected_factors  = [f for f in selected_factors if f]  # remove empty strings

    return {
        "selected_factors": selected_factors,
        "selected_alpha":   float(best_row["alpha"]),
        "selected_delta":   float(best_row["delta"]),
        "path_df":          path_df,
        "coef_path":        coefs,
        "alphas":           alphas,
        "factor_names":     factors,
    }


# ---------------------------------------------------------------------------
# Rolling / windowed variant
# ---------------------------------------------------------------------------

def rolling_dguarded_lasso(
    X: pd.DataFrame,
    y: pd.Series,
    tau: int = 120,
    delta_max: float = 2.0,
    n_alphas: int = 80,
    step: int = 5,
) -> pd.DataFrame:
    """
    Apply D-Guarded Lasso Path on a rolling window of length *tau*,
    advancing by *step* days at a time.

    Returns
    -------
    pd.DataFrame with columns:
      date, selected_factors (comma-sep str), n_active, delta, alpha
    """
    dates   = X.index
    T       = len(dates)
    records = []

    for end_idx in range(tau - 1, T, step):
        start_idx = end_idx - tau + 1
        X_win     = X.iloc[start_idx:end_idx + 1]
        y_win     = y.iloc[start_idx:end_idx + 1]

        try:
            res = dguarded_lasso_path(X_win, y_win, delta_max=delta_max,
                                      n_alphas=n_alphas)  # n_alphas forwarded correctly
            records.append(
                {
                    "date":             dates[end_idx],
                    "selected_factors": ",".join(res["selected_factors"]),
                    "n_active":         len(res["selected_factors"]),
                    "delta":            res["selected_delta"],
                    "alpha":            res["selected_alpha"],
                }
            )
        except Exception as exc:
            records.append(
                {
                    "date":             dates[end_idx],
                    "selected_factors": "",
                    "n_active":         0,
                    "delta":            np.nan,
                    "alpha":            np.nan,
                }
            )

    return pd.DataFrame(records).set_index("date")
