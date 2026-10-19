"""
knapsack_optimizer.py
=====================
Recursive State EWMA Tracker + MILP Knapsack Factor Optimizer
--------------------------------------------------------------
Implements the dynamic factor selection routine described in Section 5.2
of the specification.

Components
----------
1. **RecursiveStateTracker**
   Maintains a daily state vector for each candidate factor:
   - ``ewma_tstat[j]``  : EWMA of the factor's rolling t-statistic
   - ``duration[j]``    : consecutive days the factor has been included (>0)
                          or excluded (<0)
   Updated via ``update(t_stats, selected)`` at each rebalancing date.

2. **knapsack_milp**
   Solves a Mixed-Integer Linear Program to choose the best k ≤ k_max
   factors at each date, maximising cumulative EWMA t-stat value while
   penalising turnover against the previous day's selection.

   Objective (maximise):
       Σ_j  ewma_tstat[j] * s[j]  −  γ * Σ_j |s[j] − s_prev[j]|

   subject to:
       Σ_j s[j] ≤ k_max
       s[j] ∈ {0, 1}

   The absolute-value turnover term is linearised as:
       |s[j] − s_prev[j]| = flip_pos[j] + flip_neg[j]
       flip_pos[j] − flip_neg[j] = s[j] − s_prev[j]
       flip_pos[j], flip_neg[j] ≥ 0

   Implementation uses ``scipy.optimize.milp`` (HiGHS solver, no external
   licence required).

3. **run_knapsack_selection**
   Orchestrates the full daily loop:
   a. Compute rolling OLS t-statistics for every factor in the catalog.
   b. Update the EWMA state.
   c. Solve the MILP to get today's selection.
   d. Record the selection for the Factor Radar visualisation.

Mathematical notation
---------------------
* P     : number of candidate factors
* k_max : maximum active factors
* γ     : turnover / stability penalty (≥ 0)
* λ_ewma: EWMA decay factor = 2 / (halflife + 1)
* s*    : previous day's binary selection vector (P,)
* v     : EWMA t-stat value vector (P,)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.stats import t as t_dist


# ---------------------------------------------------------------------------
# 1. Recursive State Tracker
# ---------------------------------------------------------------------------

class RecursiveStateTracker:
    """
    Maintains a daily EWMA of per-factor t-statistics and an
    inclusion/exclusion duration counter.

    Parameters
    ----------
    factor_names : list of factor tickers
    halflife     : EWMA half-life in trading days (default 21 ≈ 1 month)
    """

    def __init__(self, factor_names: List[str], halflife: int = 21):
        self.factor_names = factor_names
        self.P            = len(factor_names)
        self._idx         = {f: i for i, f in enumerate(factor_names)}
        self.decay        = 2.0 / (halflife + 1.0)

        # State vectors
        self.ewma_tstat   = np.zeros(self.P, dtype=np.float64)
        self.duration     = np.zeros(self.P, dtype=np.int32)   # + included, - excluded
        self.prev_selected = np.zeros(self.P, dtype=np.int32)

    def update(
        self,
        t_stats: Dict[str, float],
        selected: List[str],
    ) -> None:
        """
        Update state after observing today's t-statistics and selection.

        Parameters
        ----------
        t_stats  : dict  ticker → rolling t-statistic (NaN if unavailable)
        selected : list of tickers currently selected by the MILP
        """
        sel_set = set(selected)
        for fname in self.factor_names:
            j   = self._idx[fname]
            ts  = t_stats.get(fname, np.nan)
            if np.isfinite(ts):
                self.ewma_tstat[j] = (
                    self.decay * ts + (1.0 - self.decay) * self.ewma_tstat[j]
                )
            # Duration counter
            if fname in sel_set:
                self.duration[j] = max(self.duration[j], 0) + 1
            else:
                self.duration[j] = min(self.duration[j], 0) - 1

        self.prev_selected = np.array(
            [1 if f in sel_set else 0 for f in self.factor_names],
            dtype=np.int32,
        )

    def get_state_df(self) -> pd.DataFrame:
        """Return the current state as a DataFrame for display."""
        return pd.DataFrame(
            {
                "factor":      self.factor_names,
                "ewma_tstat":  self.ewma_tstat,
                "duration":    self.duration,
                "selected":    self.prev_selected,
            }
        ).set_index("factor")


# ---------------------------------------------------------------------------
# 2. MILP Knapsack Optimizer
# ---------------------------------------------------------------------------

def knapsack_milp(
    values: np.ndarray,
    s_prev: np.ndarray,
    k_max: int,
    gamma: float = 1.0,
    guard_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Solve the factor-selection MILP via scipy.optimize.milp (HiGHS).

    Maximise:   Σ_j v[j]*s[j]  −  γ * Σ_j (flip_pos[j] + flip_neg[j])
    subject to: Σ_j s[j] ≤ k_max
                flip_pos[j] − flip_neg[j] = s[j] − s_prev[j]   ∀j
                s[j] ∈ {0,1},  flip_pos[j], flip_neg[j] ≥ 0

    Decision variables layout: [s (P,), flip_pos (P,), flip_neg (P,)]
    Total variables: 3P.

    Parameters
    ----------
    values     : np.ndarray (P,)  — EWMA t-stat values (objective coefficients)
    s_prev     : np.ndarray (P,)  — previous selection (0/1)
    k_max      : int              — maximum active factors
    gamma      : float            — turnover penalty weight
    guard_mask : np.ndarray (P,)  — boolean; False = factor is D-guard blocked

    Returns
    -------
    s_opt : np.ndarray (P,) int  — optimal binary selection vector
    """
    P = len(values)
    if guard_mask is None:
        guard_mask = np.ones(P, dtype=bool)

    # Force-zero any guard-blocked factors
    v = values.copy().astype(float)
    v[~guard_mask] = -1e9   # make them unattractive

    # ── Objective (milp minimises, so negate) ──────────────────────────
    # Variables: [s(P), fp(P), fn(P)]
    # cost = -v[j]*s[j] + γ*(fp[j] + fn[j])
    c = np.concatenate([-v, gamma * np.ones(P), gamma * np.ones(P)])

    # ── Integrality: s[j] are integer (0/1), flip vars are continuous ──
    integrality = np.concatenate([
        np.ones(P),   # s: integer
        np.zeros(P),  # fp: continuous
        np.zeros(P),  # fn: continuous
    ])

    # ── Bounds ──────────────────────────────────────────────────────────
    lb = np.zeros(3 * P)
    ub = np.concatenate([np.ones(P), np.ones(P), np.ones(P)])
    # Guard-blocked factors: fix s[j] = 0
    ub[:P][~guard_mask] = 0.0
    bounds = Bounds(lb=lb, ub=ub)

    # ── Constraints ─────────────────────────────────────────────────────
    # 1. Cardinality: Σ s[j] ≤ k_max
    A_card = np.zeros((1, 3 * P))
    A_card[0, :P] = 1.0

    # 2. Linearisation: fp[j] - fn[j] = s[j] - s_prev[j]  for each j
    #    → fp[j] - fn[j] - s[j] = -s_prev[j]
    A_lin = np.zeros((P, 3 * P))
    for j in range(P):
        A_lin[j, j]       = -1.0   # -s[j]
        A_lin[j, P + j]   =  1.0   # +fp[j]
        A_lin[j, 2*P + j] = -1.0   # -fn[j]

    A_full = np.vstack([A_card, A_lin])
    rhs_card = np.array([float(k_max)])
    rhs_lin  = -s_prev.astype(float)
    b_full   = np.concatenate([rhs_card, rhs_lin])

    constraints = LinearConstraint(A_full, lb=b_full, ub=b_full)
    # Cardinality is ≤ k_max, not equality; fix upper bound
    constraints_list = [
        LinearConstraint(A_card, lb=-np.inf, ub=float(k_max)),
        LinearConstraint(A_lin,  lb=rhs_lin, ub=rhs_lin),
    ]

    result = milp(
        c=c,
        constraints=constraints_list,
        integrality=integrality,
        bounds=bounds,
        options={"disp": False, "time_limit": 10.0},
    )

    if result.success:
        s_opt = np.round(result.x[:P]).astype(int)
        s_opt = np.clip(s_opt, 0, 1)
    else:
        # Fallback: greedy top-k by value (respecting guard)
        masked_v = v.copy()
        masked_v[~guard_mask] = -np.inf
        top_k    = np.argsort(masked_v)[::-1][:k_max]
        s_opt    = np.zeros(P, dtype=int)
        s_opt[top_k] = 1

    return s_opt


# ---------------------------------------------------------------------------
# 3. Rolling t-statistic computation (pure NumPy, no sklearn dependency)
# ---------------------------------------------------------------------------

def compute_rolling_tstats(
    X: pd.DataFrame,
    y: pd.Series,
    tau: int = 60,
) -> pd.DataFrame:
    """
    Compute rolling OLS t-statistics for each factor in X against y.

    Uses a simple univariate OLS (one factor at a time) for speed.

    Returns
    -------
    pd.DataFrame  shape (T, P)  — t-statistics, NaN for t < tau
    """
    factors = X.columns.tolist()
    T       = len(y)
    t_stats = np.full((T, len(factors)), np.nan)

    y_np = y.values.astype(float)
    X_np = X.values.astype(float)

    for j, _ in enumerate(factors):
        x_j = X_np[:, j]
        for t in range(tau - 1, T):
            t0    = t - tau + 1
            y_w   = y_np[t0:t + 1]
            x_w   = x_j[t0:t + 1]
            xbar  = x_w.mean()
            ybar  = y_w.mean()
            sxx   = np.sum((x_w - xbar) ** 2)
            if sxx < 1e-12:
                continue
            beta  = np.sum((x_w - xbar) * (y_w - ybar)) / sxx
            alpha = ybar - beta * xbar
            resid = y_w - (alpha + beta * x_w)
            s2    = np.sum(resid ** 2) / max(tau - 2, 1)
            se    = np.sqrt(s2 / (sxx + 1e-12))
            t_stats[t, j] = beta / (se + 1e-12)

    return pd.DataFrame(t_stats, index=y.index, columns=factors)


# ---------------------------------------------------------------------------
# 4. Full orchestration
# ---------------------------------------------------------------------------

def run_knapsack_selection(
    X: pd.DataFrame,
    y: pd.Series,
    delta_max: float = 2.0,
    gamma: float = 1.0,
    k_max: int = 5,
    tau: int = 60,
    ewma_halflife: int = 21,
    rebal_step: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, RecursiveStateTracker]:
    """
    Full dynamic factor selection pipeline.

    Steps
    -----
    1. Compute rolling t-statistics for all factors.
    2. At each rebalancing date:
       a. Update EWMA state.
       b. Compute D-guard mask from rolling Lasso path.
       c. Solve MILP Knapsack.
       d. Record selection.

    Returns
    -------
    selection_df : pd.DataFrame  shape (n_rebal, P)  — binary selection matrix
                   rows = rebalancing dates, columns = factor tickers
    state_df     : pd.DataFrame  — final EWMA state
    tracker      : RecursiveStateTracker  — final state object
    """
    from lasso_guard import dguarded_lasso_path

    factors = X.columns.tolist()
    P       = len(factors)
    dates   = X.index
    T       = len(dates)

    tracker = RecursiveStateTracker(factors, halflife=ewma_halflife)

    # Step 1: Rolling t-stats (vectorised per factor)
    tstat_df = compute_rolling_tstats(X, y, tau=tau)

    selection_records = []

    for end_idx in range(tau - 1, T, rebal_step):
        date    = dates[end_idx]
        t_start = max(0, end_idx - tau + 1)

        # Current t-stats (scalar per factor)
        t_stats_today = {
            f: float(tstat_df.iloc[end_idx][f])
            for f in factors
        }

        # D-guard: run Lasso path on this window
        X_win = X.iloc[t_start:end_idx + 1]
        y_win = y.iloc[t_start:end_idx + 1]
        try:
            lasso_res  = dguarded_lasso_path(X_win, y_win, delta_max=delta_max,
                                             n_alphas=60)
            guard_set  = set(lasso_res["selected_factors"])
            guard_mask = np.array([f in guard_set for f in factors], dtype=bool)
            # If guard_set is empty (all blocked), fall back to all allowed
            if not any(guard_mask):
                guard_mask = np.ones(P, dtype=bool)
        except Exception:
            guard_mask = np.ones(P, dtype=bool)

        # EWMA values
        values = tracker.ewma_tstat.copy()
        # Replace NaN/0 with raw t-stat if EWMA not yet warmed up
        for j, f in enumerate(factors):
            if values[j] == 0.0 and np.isfinite(t_stats_today.get(f, np.nan)):
                values[j] = t_stats_today[f]

        # MILP solve
        s_opt = knapsack_milp(
            values    = values,
            s_prev    = tracker.prev_selected,
            k_max     = k_max,
            gamma     = gamma,
            guard_mask= guard_mask,
        )

        selected = [factors[j] for j in range(P) if s_opt[j] == 1]

        # Update state
        tracker.update(t_stats_today, selected)

        row = {"date": date}
        row.update({f: int(s_opt[j]) for j, f in enumerate(factors)})
        selection_records.append(row)

    selection_df = pd.DataFrame(selection_records).set_index("date")
    state_df     = tracker.get_state_df()

    return selection_df, state_df, tracker
