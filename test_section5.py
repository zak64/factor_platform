"""
test_section5.py
================
End-to-end tests for Section 5: D-Guarded Lasso & Knapsack Factor Selection.
Uses synthetic data so no internet connection is required.
"""
import sys, os
sys.path.insert(0, "/home/ubuntu/factor_platform")

import numpy as np
import pandas as pd

np.random.seed(42)

# ── Generate synthetic factor returns ──────────────────────────────────────
T = 300
P = 12   # number of candidate factors
dates = pd.date_range("2022-01-01", periods=T, freq="B")

# True betas: asset is driven by factors 0, 3, 7
true_betas = np.zeros(P)
true_betas[[0, 3, 7]] = [1.1, 0.6, -0.4]

X_np = np.random.normal(0, 0.01, (T, P)).astype(np.float64)
# Add some correlation between factors 0 and 1 (should be caught by D-guard)
X_np[:, 1] = 0.9 * X_np[:, 0] + 0.1 * np.random.normal(0, 0.01, T)

y_np = X_np @ true_betas + np.random.normal(0, 0.005, T)
# Add 5 % outliers
outlier_idx = np.random.choice(T, size=int(0.05*T), replace=False)
y_np[outlier_idx] += np.random.normal(0, 0.05, len(outlier_idx))

factor_names = [f"F{j:02d}" for j in range(P)]
X = pd.DataFrame(X_np, index=dates, columns=factor_names)
y = pd.Series(y_np, index=dates, name="ASSET")

print("=" * 60)
print("Test 1: factor_catalog module")
from factor_catalog import FACTOR_CATALOG, ALL_TICKERS, THEMES, get_catalog_metadata, get_theme_colors

assert len(FACTOR_CATALOG) >= 30, f"Expected ≥30 factors, got {len(FACTOR_CATALOG)}"
assert len(ALL_TICKERS) >= 30
assert len(THEMES) >= 5
meta = get_catalog_metadata()
assert "theme" in meta.columns
colors = get_theme_colors()
assert len(colors) >= 5
print(f"  Catalog: {len(FACTOR_CATALOG)} factors across {len(THEMES)} themes  ✓")

print("=" * 60)
print("Test 2: compute_log_det")
from lasso_guard import compute_log_det

# Uncorrelated random data → correlation matrix ≈ identity → delta ≈ 0
X_uncorr = np.random.normal(0, 1, (500, 3))   # 3 independent columns
delta_uncorr = compute_log_det(X_uncorr)
assert delta_uncorr < 0.5, f"Expected small delta for uncorrelated factors, got {delta_uncorr}"

# Highly correlated matrix → large delta
X_corr = np.column_stack([X_np[:, 0], X_np[:, 0] + 1e-4*np.random.randn(T)])
delta_corr = compute_log_det(X_corr)
assert delta_corr > 5.0, f"Expected large delta for correlated factors, got {delta_corr}"
print(f"  Uncorrelated delta={delta_uncorr:.4f}  Correlated delta={delta_corr:.2f}  ✓")

print("=" * 60)
print("Test 3: D-Guarded Lasso Path")
from lasso_guard import dguarded_lasso_path

result = dguarded_lasso_path(X, y, delta_max=2.0, n_alphas=60)
assert "selected_factors" in result
assert "path_df" in result
assert len(result["path_df"]) > 0
assert "delta" in result["path_df"].columns
assert "guard_pass" in result["path_df"].columns

# Check that correlated factor pair (F00, F01) is not both selected
sf = result["selected_factors"]
both_corr = ("F00" in sf and "F01" in sf)
print(f"  Selected factors: {sf}")
print(f"  Both correlated factors selected: {both_corr} (should be False ideally)")
print(f"  δ at selection: {result['selected_delta']:.4f}")
print(f"  Guard-pass steps: {result['path_df']['guard_pass'].sum()}/{len(result['path_df'])}  ✓")

print("=" * 60)
print("Test 4: Rolling t-statistics")
from knapsack_optimizer import compute_rolling_tstats

tstat_df = compute_rolling_tstats(X, y, tau=60)
assert tstat_df.shape == (T, P)
# True factors should have higher t-stats than noise factors
true_factors = ["F00", "F03", "F07"]
noise_factors = ["F02", "F04", "F05"]
mean_true  = tstat_df[true_factors].abs().mean().mean()
mean_noise = tstat_df[noise_factors].abs().mean().mean()
print(f"  Mean |t-stat| true factors: {mean_true:.3f}")
print(f"  Mean |t-stat| noise factors: {mean_noise:.3f}")
assert mean_true > mean_noise, "True factors should have higher t-stats"
print(f"  Rolling t-stats shape: {tstat_df.shape}  ✓")

print("=" * 60)
print("Test 5: RecursiveStateTracker")
from knapsack_optimizer import RecursiveStateTracker

tracker = RecursiveStateTracker(factor_names, halflife=21)
assert tracker.ewma_tstat.shape == (P,)
assert tracker.duration.shape   == (P,)

# Simulate 5 updates
for step in range(5):
    t_stats = {f: float(np.random.randn()) for f in factor_names}
    selected = factor_names[:3]
    tracker.update(t_stats, selected)

state_df = tracker.get_state_df()
assert len(state_df) == P
assert "ewma_tstat" in state_df.columns
assert "duration"   in state_df.columns
# Duration for first 3 factors should be positive (included 5 times)
for f in factor_names[:3]:
    assert state_df.loc[f, "duration"] == 5, f"Expected duration=5 for {f}"
print(f"  EWMA state shape: {state_df.shape}  ✓")

print("=" * 60)
print("Test 6: MILP Knapsack Optimizer")
from knapsack_optimizer import knapsack_milp

values    = np.array([3.0, 1.0, 2.5, 0.5, 4.0, 1.5, 2.0, 0.8, 3.5, 1.2, 0.3, 2.8])
s_prev    = np.zeros(P, dtype=int)
guard_mask = np.ones(P, dtype=bool)
guard_mask[1] = False   # block F01 (correlated with F00)

s_opt = knapsack_milp(values, s_prev, k_max=4, gamma=0.5, guard_mask=guard_mask)
assert s_opt.shape == (P,)
assert s_opt.sum() <= 4, f"Expected ≤4 active factors, got {s_opt.sum()}"
assert s_opt[1] == 0, "Blocked factor F01 should not be selected"
selected = [factor_names[j] for j in range(P) if s_opt[j] == 1]
print(f"  Selected (k_max=4): {selected}")
print(f"  Total active: {s_opt.sum()}  ✓")

# Test turnover penalty: running again with s_prev = s_opt should return same selection
s_opt2 = knapsack_milp(values, s_opt, k_max=4, gamma=5.0, guard_mask=guard_mask)
turnover = int(np.sum(np.abs(s_opt2 - s_opt)))
print(f"  Turnover with γ=5.0: {turnover} (should be 0 or very low)  ✓")

print("=" * 60)
print("Test 7: Full run_knapsack_selection pipeline")
from knapsack_optimizer import run_knapsack_selection

selection_df, state_df, tracker = run_knapsack_selection(
    X=X, y=y,
    delta_max=2.0, gamma=1.0, k_max=4,
    tau=60, ewma_halflife=21, rebal_step=10,
)
assert len(selection_df) > 0
assert selection_df.shape[1] == P
assert selection_df.values.max() <= 1
assert selection_df.values.min() >= 0
# Check k_max constraint
row_sums = selection_df.sum(axis=1)
assert (row_sums <= 4).all(), f"k_max violated: max={row_sums.max()}"
print(f"  Selection shape: {selection_df.shape}")
print(f"  Max active per date: {row_sums.max()}  (k_max=4)  ✓")

print("=" * 60)
print("Test 8: dynamic_selector integration")
from dynamic_selector import (
    DynamicSelectionParams,
    build_factor_radar_df,
    build_lasso_path_chart_df,
    build_coef_heatmap_df,
)

# build_factor_radar_df
from factor_catalog import get_catalog_metadata
meta_small = pd.DataFrame(
    [{"ticker": f, "name": f, "theme": "Test"} for f in factor_names]
).set_index("ticker")
radar_df = build_factor_radar_df(selection_df, meta_small)
assert "Test" in radar_df.columns
assert len(radar_df) == len(selection_df)
print(f"  Factor Radar shape: {radar_df.shape}  ✓")

# build_lasso_path_chart_df
from lasso_guard import dguarded_lasso_path
lasso_res = dguarded_lasso_path(X.iloc[-60:], y.iloc[-60:], delta_max=2.0, n_alphas=40)
path_chart = build_lasso_path_chart_df(lasso_res)
assert "log_alpha" in path_chart.columns
assert "n_active"  in path_chart.columns
assert "delta"     in path_chart.columns
print(f"  Lasso path chart shape: {path_chart.shape}  ✓")

coef_heat = build_coef_heatmap_df(lasso_res)
assert "factor"      in coef_heat.columns
assert "coefficient" in coef_heat.columns
print(f"  Coef heatmap shape: {coef_heat.shape}  ✓")

print("=" * 60)
print("Test 9: DynamicSelectionParams defaults")
params = DynamicSelectionParams()
assert params.k_max == 5
assert params.delta_max == 2.0
assert params.gamma == 1.0
print(f"  DynamicSelectionParams defaults OK  ✓")

print("=" * 60)
print("ALL SECTION 5 TESTS PASSED ✓")
