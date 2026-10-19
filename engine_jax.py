"""
engine_jax.py
=============
Engine B – Bayesline Accelerated Windowed IRLS Huber Regression (JAX).

Methodology (Bayesline-style)
------------------------------
1. **Global outlier weights** are computed once per asset over the full
   time-series using the Huber threshold M (typically 1.35).
2. **Rolling OLS** is expressed as 1-D convolutions (boxcar kernel of length
   τ) applied to the weighted cross-products, eliminating all Python loops
   over time.  The closed-form single-factor update is:

       β = [Σ_w(xy) - (1/τ) Σ_w(x) Σ_w(y)] /
           [Σ_w(x²) - (1/τ) Σ_w(x)²]

   where Σ_w(·) denotes the rolling weighted sum implemented via conv.

3. The IRLS loop iterates: update weights → update betas → repeat until
   convergence.  The entire inner loop is `jax.jit`-compiled.

4. For multi-factor models we solve the full K×K normal equations inside
   the rolling window using JAX's `jnp.linalg.solve`.

Key JAX correctness notes
--------------------------
* `jnp.where` does NOT short-circuit — both branches are always evaluated.
  We must guard denominators with a safe fallback before dividing.
* `jnp.nanmedian` is not supported inside `jit`; we use a sort-based
  approximate median instead.
* `jax.lax.scan` requires the `length` argument to be a *static* Python int,
  so `n_iter` must be declared as a static arg in `@partial(jit, ...)`.
"""

from __future__ import annotations

import time
import warnings
from functools import partial
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# ---- JAX imports (lazy to allow graceful fallback) -------------------------
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    warnings.warn("JAX not available; Engine B will fall back to NumPy.")


# ---------------------------------------------------------------------------
# Low-level JAX primitives
# ---------------------------------------------------------------------------

if JAX_AVAILABLE:

    def _boxcar_conv(signal: jnp.ndarray, tau: int) -> jnp.ndarray:
        """
        Compute rolling sum of *signal* with a boxcar kernel of length *tau*.
        Uses jax.lax.conv_general_dilated for maximum efficiency.

        Returns array of shape (T,) where the first tau-1 entries are NaN
        (insufficient history).
        """
        T = signal.shape[0]
        kernel = jnp.ones((tau,), dtype=signal.dtype)

        # lax.conv expects (N, C, H) layout
        x = signal.reshape(1, 1, T)    # (N=1, C=1, H=T)
        k = kernel.reshape(1, 1, tau)  # (O=1, I=1, H=tau)

        # 'VALID' padding: output length = T - tau + 1
        out = jax.lax.conv_general_dilated(
            x, k,
            window_strides=(1,),
            padding="VALID",
            dimension_numbers=("NCH", "OIH", "NCH"),
        )
        out = out.reshape(-1)                           # shape (T - tau + 1,)

        # Prepend NaNs so the output aligns with the original time axis
        pad = jnp.full((tau - 1,), jnp.nan, dtype=signal.dtype)
        return jnp.concatenate([pad, out])


    def _approx_median(x: jnp.ndarray) -> jnp.ndarray:
        """
        Approximate median via sorting (works inside jit).
        Replaces jnp.nanmedian which is not jit-compatible.
        """
        # Replace NaN with 0 for sorting purposes
        x_clean = jnp.where(jnp.isnan(x), 0.0, x)
        sorted_x = jnp.sort(jnp.abs(x_clean))
        n = sorted_x.shape[0]
        return sorted_x[n // 2]


    @partial(jit, static_argnums=(1, 2, 3))
    def _irls_single_factor(
        yx: jnp.ndarray,   # shape (T, 2)  columns: [y, x]
        tau: int,
        M: float = 1.35,
        n_iter: int = 10,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        JIT-compiled single-factor Windowed IRLS.

        Returns
        -------
        betas : shape (T,)  – rolling beta for each date (NaN for t < tau)
        alphas: shape (T,)  – rolling intercept
        """
        y = yx[:, 0]
        x = yx[:, 1]
        T = y.shape[0]

        # Initialise weights uniformly
        w = jnp.ones(T, dtype=jnp.float32)

        def irls_step(w, _):
            # Weighted cross-products
            wy  = w * y
            wx  = w * x
            wxy = w * x * y
            wx2 = w * x * x
            ww  = w                         # sum of weights

            # Rolling sums via boxcar convolution
            S_wy  = _boxcar_conv(wy,  tau)
            S_wx  = _boxcar_conv(wx,  tau)
            S_wxy = _boxcar_conv(wxy, tau)
            S_wx2 = _boxcar_conv(wx2, tau)
            S_ww  = _boxcar_conv(ww,  tau)

            # OLS normal equations (Bayesline closed form)
            # Guard denominator to avoid NaN from NaN/NaN in padded region
            denom     = S_ww * S_wx2 - S_wx ** 2
            denom_safe = jnp.where(jnp.abs(denom) > 1e-12, denom, 1.0)
            numer_b   = S_ww * S_wxy - S_wx * S_wy
            beta      = jnp.where(
                jnp.abs(denom) > 1e-12,
                numer_b / denom_safe,
                jnp.nan,
            )

            S_ww_safe = jnp.where(S_ww > 1e-12, S_ww, 1.0)
            alpha     = jnp.where(
                S_ww > 1e-12,
                (S_wy - beta * S_wx) / S_ww_safe,
                jnp.nan,
            )

            # Residuals & Huber weight update
            resid    = y - (alpha + beta * x)
            # Replace NaN residuals (padded region) with 0 before computing weights
            resid_safe = jnp.where(jnp.isnan(resid), 0.0, resid)
            sigma    = _approx_median(resid_safe) / 0.6745
            sigma    = jnp.where(sigma < 1e-8, 1e-8, sigma)
            abs_r    = jnp.abs(resid_safe)
            new_w    = jnp.where(abs_r <= M * sigma, 1.0, M * sigma / (abs_r + 1e-12))
            new_w    = jnp.clip(new_w, 0.0, 1.0)

            return new_w, (beta, alpha)

        # Run IRLS iterations using lax.scan for efficiency
        final_w, (betas_trace, alphas_trace) = jax.lax.scan(
            irls_step, w, None, length=n_iter
        )
        # Return the last iteration's estimates
        return betas_trace[-1], alphas_trace[-1]


    @partial(jit, static_argnums=(1, 2, 3))
    def _irls_multi_factor(
        data: jnp.ndarray,   # shape (T, 1+K)  columns: [y, x1, ..., xK]
        tau: int,
        M: float = 1.35,
        n_iter: int = 10,
    ) -> jnp.ndarray:
        """
        JIT-compiled multi-factor Windowed IRLS.

        Solves the K×K normal equations at each rolling window via
        jnp.linalg.solve applied over the time axis.

        Returns
        -------
        betas : shape (T, K)  – rolling betas (NaN for t < tau)
        """
        T, KK = data.shape
        K = KK - 1
        y = data[:, 0]
        X = data[:, 1:]          # (T, K)

        w = jnp.ones(T, dtype=jnp.float32)

        def irls_step(w, _):
            # Weighted signals
            wy  = w * y                          # (T,)
            wX  = w[:, None] * X                 # (T, K)

            # Rolling sums of weighted signals
            S_wy  = _boxcar_conv(wy, tau)                            # (T,)
            S_wX  = jnp.stack([_boxcar_conv(wX[:, k], tau)
                                for k in range(K)], axis=1)          # (T, K)
            S_ww  = _boxcar_conv(w, tau)                              # (T,)

            # Rolling sums of weighted cross-products X^T X
            S_wXX = jnp.stack(
                [jnp.stack([_boxcar_conv(wX[:, i] * X[:, j], tau)
                             for j in range(K)], axis=1)
                 for i in range(K)], axis=1)                          # (T, K, K)

            # Rolling sums of weighted X^T y
            S_wXy = jnp.stack(
                [_boxcar_conv(wX[:, k] * y, tau) for k in range(K)], axis=1
            )                                                          # (T, K)

            # Demean within window to handle intercept
            S_ww_safe = jnp.where(S_ww > 1e-12, S_ww, 1.0)
            mean_y = S_wy / S_ww_safe                                 # (T,)
            mean_X = S_wX / S_ww_safe[:, None]                        # (T, K)

            # Centred cross-products
            S_wXy_dm = (
                S_wXy
                - mean_X * S_wy[:, None]
                - mean_y[:, None] * S_wX
                + mean_X * mean_y[:, None] * S_ww[:, None]
            )                                                          # (T, K)

            S_wXX_dm = (
                S_wXX
                - mean_X[:, :, None] * S_wX[:, None, :]
                - mean_X[:, None, :] * S_wX[:, :, None]
                + mean_X[:, :, None] * mean_X[:, None, :] * S_ww[:, None, None]
            )                                                          # (T, K, K)

            # Solve K×K system for each t via vmap
            # Add Tikhonov regularisation to avoid singular matrices
            reg = jnp.eye(K) * 1e-8

            def solve_one(A, b):
                return jnp.linalg.solve(A + reg, b)

            betas = vmap(solve_one)(S_wXX_dm, S_wXy_dm)              # (T, K)

            # Mask out the padded (NaN) region
            valid = jnp.isfinite(S_ww)
            betas = jnp.where(valid[:, None], betas, jnp.nan)

            # Intercepts
            alphas = mean_y - jnp.sum(betas * mean_X, axis=1)        # (T,)

            # Residuals & Huber weight update
            resid  = y - alphas - jnp.sum(betas * X, axis=1)
            # Replace NaN residuals (padded region) with 0 before computing weights
            resid_safe = jnp.where(jnp.isnan(resid), 0.0, resid)
            sigma  = _approx_median(resid_safe) / 0.6745
            sigma  = jnp.where(sigma < 1e-8, 1e-8, sigma)
            abs_r  = jnp.abs(resid_safe)
            new_w  = jnp.where(abs_r <= M * sigma, 1.0, M * sigma / (abs_r + 1e-12))
            new_w  = jnp.clip(new_w, 0.0, 1.0)

            return new_w, betas

        final_w, betas_trace = jax.lax.scan(irls_step, w, None, length=n_iter)
        return betas_trace[-1]   # (T, K)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_jax_rolling(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    tau: int = 60,
    M: float = 1.35,
    n_iter: int = 10,
) -> Tuple[pd.DataFrame, float]:
    """
    Run JAX-accelerated Windowed IRLS for every asset.

    Parameters
    ----------
    asset_returns : pd.DataFrame  shape (T, N_assets)
    factor_returns: pd.DataFrame  shape (T, K_factors)
    tau   : rolling window length
    M     : Huber threshold
    n_iter: IRLS iterations

    Returns
    -------
    (results_df, elapsed_seconds)
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is not installed.  Cannot run Engine B.")

    assets  = asset_returns.columns.tolist()
    factors = factor_returns.columns.tolist()
    K       = len(factors)
    dates   = asset_returns.index
    T       = len(dates)

    records: List[Dict] = []
    t0 = time.perf_counter()

    for asset in assets:
        y_np = asset_returns[asset].values.astype(np.float32)
        X_np = factor_returns[factors].values.astype(np.float32)

        # Replace NaN with 0 (masked out by weights anyway)
        nan_mask = ~(np.isfinite(y_np) & np.all(np.isfinite(X_np), axis=1))
        y_np[nan_mask] = 0.0
        X_np[nan_mask] = 0.0

        data = np.concatenate([y_np[:, None], X_np], axis=1)  # (T, 1+K)
        data_jax = jnp.array(data)

        if K == 1:
            yx = data_jax[:, :2]
            betas_jax, _ = _irls_single_factor(yx, tau, M, n_iter)
            betas_np = np.array(betas_jax).reshape(T, 1)
        else:
            betas_jax = _irls_multi_factor(data_jax, tau, M, n_iter)
            betas_np  = np.array(betas_jax)          # (T, K)

        # ---- Approximate t-stats (post-hoc, NumPy) -------------------------
        for t_idx in range(tau - 1, T):
            t_start = t_idx - tau + 1
            y_win = y_np[t_start:t_idx + 1]
            X_win = X_np[t_start:t_idx + 1]
            betas_t = betas_np[t_idx]                # (K,)

            if not np.all(np.isfinite(betas_t)):
                continue

            # Huber weights at this window
            alpha_t = np.mean(y_win) - np.dot(betas_t, np.mean(X_win, axis=0))
            resid   = y_win - alpha_t - X_win @ betas_t
            sigma   = np.median(np.abs(resid)) / 0.6745
            if sigma < 1e-8:
                sigma = 1e-8
            abs_r   = np.abs(resid)
            w_t     = np.where(abs_r <= M * sigma, 1.0, M * sigma / (abs_r + 1e-12))

            W   = np.diag(w_t)
            XtW = X_win.T @ W
            try:
                cov_b  = np.linalg.inv(XtW @ X_win + np.eye(K) * 1e-8) * (sigma ** 2)
                se_b   = np.sqrt(np.diag(cov_b))
            except np.linalg.LinAlgError:
                se_b = np.full(K, np.nan)

            t_stats = betas_t / (se_b + 1e-12)
            dof     = max(tau - K - 1, 1)
            p_vals  = 2 * stats.t.sf(np.abs(t_stats), df=dof)

            ss_res = np.sum(resid ** 2)
            ss_tot = np.sum((y_win - y_win.mean()) ** 2)
            r2     = 1.0 - ss_res / (ss_tot + 1e-12)

            date = dates[t_idx]
            for k, factor in enumerate(factors):
                records.append(
                    {
                        "asset":     asset,
                        "date":      date,
                        "factor":    factor,
                        "beta":      float(betas_t[k]),
                        "t_stat":    float(t_stats[k]),
                        "p_value":   float(p_vals[k]),
                        "r_squared": float(r2),
                        "method":    "Bayesline JAX",
                    }
                )

    elapsed = time.perf_counter() - t0
    results_df = pd.DataFrame(records)
    return results_df, elapsed


def pivot_betas(results_df: pd.DataFrame, factor: str) -> pd.DataFrame:
    """Pivot to wide format: rows=dates, columns=assets, values=beta."""
    sub = results_df[results_df["factor"] == factor]
    return sub.pivot(index="date", columns="asset", values="beta")
