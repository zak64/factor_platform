"""
test_backend.py
===============
Quick integration test for the data sourcing and both computation engines.
Uses a small synthetic dataset so the test runs offline and fast.
"""

import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ubuntu/factor_platform")

# ---- Synthetic data --------------------------------------------------------
np.random.seed(42)
T = 300
dates = pd.date_range("2020-01-01", periods=T, freq="B")

# Two factors
factor_ret = pd.DataFrame(
    {
        "SPY": np.random.normal(0.0005, 0.01, T),
        "GLD": np.random.normal(0.0002, 0.008, T),
    },
    index=dates,
)

# Three assets: y = beta1*SPY + beta2*GLD + noise
betas_true = {"AAPL": (1.2, 0.1), "MSFT": (0.9, -0.05), "NVDA": (1.5, 0.2)}
asset_ret = pd.DataFrame(
    {
        ticker: b1 * factor_ret["SPY"] + b2 * factor_ret["GLD"]
                + np.random.normal(0, 0.005, T)
        for ticker, (b1, b2) in betas_true.items()
    },
    index=dates,
)

TAU = 60

# ---- Test Engine A ---------------------------------------------------------
print("=" * 60)
print("Testing Engine A: Scikit-Learn Baseline")
from engine_sklearn import run_sklearn_rolling

res_skl, t_skl = run_sklearn_rolling(asset_ret, factor_ret, tau=TAU)
assert not res_skl.empty, "Engine A returned empty results"
print(f"  Rows: {len(res_skl):,}  |  Time: {t_skl*1000:.1f} ms")
print(f"  Columns: {list(res_skl.columns)}")
print(f"  Sample beta (AAPL ~ SPY, last date):")
sample = res_skl[
    (res_skl["asset"] == "AAPL") & (res_skl["factor"] == "SPY")
].tail(1)
print(f"    {sample[['date','beta','t_stat','p_value']].to_string(index=False)}")
print("  Engine A: PASSED ✓")

# ---- Test Engine B ---------------------------------------------------------
print()
print("=" * 60)
print("Testing Engine B: JAX Windowed IRLS")
from engine_jax import run_jax_rolling

res_jax, t_jax = run_jax_rolling(asset_ret, factor_ret, tau=TAU, M=1.35, n_iter=10)
assert not res_jax.empty, "Engine B returned empty results"
print(f"  Rows: {len(res_jax):,}  |  Time: {t_jax*1000:.1f} ms")
print(f"  Sample beta (AAPL ~ SPY, last date):")
sample_j = res_jax[
    (res_jax["asset"] == "AAPL") & (res_jax["factor"] == "SPY")
].tail(1)
print(f"    {sample_j[['date','beta','t_stat','p_value']].to_string(index=False)}")
print("  Engine B: PASSED ✓")

# ---- Convergence check -----------------------------------------------------
print()
print("=" * 60)
print("Convergence check (JAX vs Scikit-Learn betas)")

for ticker in ["AAPL", "MSFT", "NVDA"]:
    for factor in ["SPY", "GLD"]:
        b_skl = res_skl[
            (res_skl["asset"] == ticker) & (res_skl["factor"] == factor)
        ]["beta"].dropna().values
        b_jax = res_jax[
            (res_jax["asset"] == ticker) & (res_jax["factor"] == factor)
        ]["beta"].dropna().values

        # Align lengths
        n = min(len(b_skl), len(b_jax))
        if n == 0:
            continue
        mae = np.mean(np.abs(b_skl[-n:] - b_jax[-n:]))
        print(f"  {ticker} ~ {factor}: MAE = {mae:.6f}")
        assert mae < 0.5, f"Convergence failure for {ticker} ~ {factor}: MAE={mae:.4f}"

print("  Convergence: PASSED ✓")

# ---- Speed comparison ------------------------------------------------------
if t_jax > 0 and t_skl > 0:
    print()
    print("=" * 60)
    print(f"Speed comparison:")
    print(f"  Scikit-Learn: {t_skl*1000:.1f} ms")
    print(f"  JAX:          {t_jax*1000:.1f} ms")
    speedup = t_skl / t_jax
    print(f"  Speed-up:     {speedup:.2f}×")

# ---- Data sourcing module --------------------------------------------------
print()
print("=" * 60)
print("Testing data_sourcing module (factor model registry)")
from data_sourcing import FACTOR_MODELS, list_portfolios, save_portfolio, load_portfolio

assert len(FACTOR_MODELS) >= 5, "Factor model registry too small"
print(f"  Factor models registered: {len(FACTOR_MODELS)}")
for name, spec in FACTOR_MODELS.items():
    print(f"    {name}: {[t for t,_ in spec]}")

# Test portfolio save/load
save_portfolio("test_port", ["AAPL", "MSFT"], [0.6, 0.4])
p = load_portfolio("test_port")
assert p["tickers"] == ["AAPL", "MSFT"]
assert abs(sum(p["weights"]) - 1.0) < 1e-6
print("  Portfolio save/load: PASSED ✓")

print()
print("=" * 60)
print("ALL TESTS PASSED ✓")
