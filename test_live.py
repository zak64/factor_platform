"""
test_live.py
============
End-to-end test using real yfinance data.
Downloads 2 years of data for AAPL, MSFT vs SPY (Market factor).
"""
import sys
sys.path.insert(0, "/home/ubuntu/factor_platform")

from data_sourcing import fetch_returns, fetch_factor_data, align_returns
from engine_sklearn import run_sklearn_rolling
from engine_jax import run_jax_rolling

print("Fetching real market data...")
asset_ret = fetch_returns(["AAPL", "MSFT"], lookback_years=2)
factor_ret, labels = fetch_factor_data("Market Only (CAPM)", lookback_years=2)
asset_ret, factor_ret = align_returns(asset_ret, factor_ret)

print(f"  Asset returns shape: {asset_ret.shape}")
print(f"  Factor returns shape: {factor_ret.shape}")
print(f"  Date range: {asset_ret.index[0].date()} to {asset_ret.index[-1].date()}")

TAU = 60

print("\nRunning Scikit-Learn engine...")
res_skl, t_skl = run_sklearn_rolling(asset_ret, factor_ret, tau=TAU)
print(f"  Rows: {len(res_skl):,}  |  Time: {t_skl*1000:.1f} ms")
print(f"  Latest AAPL beta (SPY):")
s = res_skl[(res_skl["asset"]=="AAPL") & (res_skl["factor"]=="SPY")].tail(1)
print(f"    {s[['date','beta','t_stat','p_value','r_squared']].to_string(index=False)}")

print("\nRunning JAX engine...")
res_jax, t_jax = run_jax_rolling(asset_ret, factor_ret, tau=TAU)
print(f"  Rows: {len(res_jax):,}  |  Time: {t_jax*1000:.1f} ms")
print(f"  Latest AAPL beta (SPY):")
s = res_jax[(res_jax["asset"]=="AAPL") & (res_jax["factor"]=="SPY")].tail(1)
print(f"    {s[['date','beta','t_stat','p_value','r_squared']].to_string(index=False)}")

print(f"\nSpeed-up: {t_skl/t_jax:.2f}×")
print("\nLIVE TEST PASSED ✓")
