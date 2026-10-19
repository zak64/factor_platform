"""
app.py
======
Financial Factor Exposure Platform – Streamlit UI

Panels
------
1. Portfolio Manager        – input tickers/weights, save/reload portfolios
2. Factor Selection         – choose factor model, rolling window τ, custom tickers
3. Computation Engine       – select JAX vs Scikit-Learn, run analysis
4. Results Dashboard        – timing, output table, beta chart
5. Convergence              – rolling beta overlay + heatmap
6. Raw Data                 – return tables + correlation matrix
7. arXiv Papers             – live academic paper search
8. Methodology              – full mathematical derivation
9. Dynamic Factor Selector  – D-Guarded Lasso + MILP Knapsack (Section 5)
"""

from __future__ import annotations

import json
import sys
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure the platform directory is on the path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---- Project modules -------------------------------------------------------
from data_sourcing import (
    FACTOR_MODELS,
    align_returns,
    fetch_factor_data,
    fetch_returns,
    list_portfolios,
    load_portfolio,
    save_portfolio,
    search_arxiv_papers,
)
from engine_sklearn import run_sklearn_rolling
from engine_jax import run_jax_rolling
from factor_catalog import (
    FACTOR_CATALOG,
    ALL_TICKERS,
    THEMES,
    get_catalog_metadata,
    get_theme_colors,
)
from dynamic_selector import (
    DynamicSelectionParams,
    run_dynamic_selection,
    build_lasso_path_chart_df,
    build_coef_heatmap_df,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Factor Exposure Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        background: linear-gradient(90deg, #1a73e8, #0d47a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #555;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #1a73e8;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1a73e8;
        border-bottom: 2px solid #e8f0fe;
        padding-bottom: 4px;
        margin-top: 1rem;
        margin-bottom: 0.6rem;
    }
    .badge-jax    { background:#e8f5e9; color:#2e7d32; padding:2px 8px; border-radius:12px; font-size:0.8rem; }
    .badge-sklearn{ background:#fff3e0; color:#e65100; padding:2px 8px; border-radius:12px; font-size:0.8rem; }
    .badge-dyn    { background:#f3e5f5; color:#6a1b9a; padding:2px 8px; border-radius:12px; font-size:0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="main-title">📈 Financial Factor Exposure Platform</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Windowed IRLS Huber Regression · Bayesline JAX Engine · D-Guarded Lasso & Knapsack Factor Selection</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    # ── Portfolio Manager ────────────────────────────────────────────────
    st.markdown("## 🗂️ Portfolio Manager")
    saved = list_portfolios()
    if saved:
        with st.expander("Load Saved Portfolio", expanded=False):
            sel_port = st.selectbox("Select portfolio", saved, key="sel_port")
            if st.button("Load", key="btn_load"):
                p = load_portfolio(sel_port)
                st.session_state["tickers_raw"] = ", ".join(p["tickers"])
                st.session_state["weights_raw"] = ", ".join(str(w) for w in p["weights"])
                st.success(f"Loaded '{sel_port}'")

    st.markdown("### Asset Tickers")
    tickers_raw = st.text_area(
        "Enter tickers (comma-separated)",
        value=st.session_state.get("tickers_raw", "AAPL, MSFT, NVDA, GOOGL, AMZN"),
        height=90,
        key="tickers_input",
    )
    st.markdown("### Weights (optional)")
    weights_raw = st.text_area(
        "Equal weights if blank (comma-separated, must sum ≤ 1)",
        value=st.session_state.get("weights_raw", ""),
        height=60,
        key="weights_input",
    )
    port_name = st.text_input("Portfolio name", value="my_portfolio", key="port_name")
    if st.button("💾 Save", key="btn_save"):
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
        if weights_raw.strip():
            weights = [float(w.strip()) for w in weights_raw.split(",") if w.strip()]
        else:
            weights = [round(1 / len(tickers), 4)] * len(tickers)
        if len(tickers) != len(weights):
            st.error("Ticker/weight count mismatch.")
        else:
            save_portfolio(port_name, tickers, weights)
            st.success(f"Saved as '{port_name}'")

    st.markdown("---")

    # ── Factor Model (for Engines A/B) ──────────────────────────────────
    st.markdown("## ⚙️ Factor Model")
    model_choice = st.selectbox("Factor Model", list(FACTOR_MODELS.keys()), index=0, key="model_choice")
    use_custom_factor = st.checkbox("Add custom factor ticker", value=False)
    custom_factor_ticker = ""
    custom_factor_label  = ""
    if use_custom_factor:
        custom_factor_ticker = st.text_input("Custom factor ticker (e.g. BTC-USD)", key="cft")
        custom_factor_label  = st.text_input("Label for this factor", value="Custom", key="cfl")
    tau = st.slider("Rolling window τ (days)", min_value=20, max_value=500, value=60, step=10)

    st.markdown("---")

    # ── Computation Engine ───────────────────────────────────────────────
    st.markdown("## 🚀 Computation Engine")
    engine_choice = st.radio(
        "Select engine",
        ["Bayesline (JAX – Accelerated)", "Scikit-Learn (CPU Loop)"],
        index=0,
        key="engine_choice",
    )
    run_both = st.checkbox("Run both engines (convergence comparison)", value=False)
    lookback = st.slider("Data lookback (years)", 1, 10, 5)
    st.markdown("---")
    run_btn = st.button("▶ Run Analysis", type="primary", use_container_width=True)

    st.markdown("---")

    # ── Dynamic Factor Selector (Section 5) ─────────────────────────────
    st.markdown("## 🧬 Dynamic Factor Selector")
    st.caption("D-Guarded Lasso Path + MILP Knapsack Optimizer")

    dyn_asset = st.text_input(
        "Target asset ticker",
        value="AAPL",
        key="dyn_asset",
        help="Single asset whose returns are regressed against the factor catalog.",
    )

    st.markdown("**Candidate Factor Themes**")
    all_themes = list(THEMES.keys())
    dyn_themes = st.multiselect(
        "Include themes",
        options=all_themes,
        default=all_themes,
        key="dyn_themes",
    )

    st.markdown("**Hyper-parameters**")
    delta_max = st.slider(
        "δ_max — Multicollinearity Guardrail",
        min_value=0.5, max_value=10.0, value=2.0, step=0.25,
        help="Maximum allowed −ln det ρ(X_λ). Lower = stricter collinearity guard.",
        key="delta_max",
    )
    gamma = st.slider(
        "γ — Turnover / Stability Penalty",
        min_value=0.0, max_value=10.0, value=1.0, step=0.25,
        help="Penalises factor turnover in the MILP objective.",
        key="gamma",
    )
    k_max = st.slider(
        "k_max — Max Active Factors",
        min_value=1, max_value=15, value=5, step=1,
        help="Hard constraint on the number of simultaneously active factors.",
        key="k_max",
    )

    with st.expander("Advanced settings", expanded=False):
        dyn_lookback   = st.slider("Lookback (years)", 1, 8, 3, key="dyn_lookback")
        dyn_tau        = st.slider("Rolling τ for t-stats", 20, 252, 60, step=10, key="dyn_tau")
        ewma_halflife  = st.slider("EWMA half-life (days)", 5, 63, 21, key="ewma_hl")
        rebal_step     = st.slider("Rebalancing step (days)", 1, 21, 5, key="rebal_step")
        dyn_custom     = st.text_input("Extra custom tickers (comma-sep)", value="", key="dyn_custom")

    dyn_btn = st.button("🧬 Run Dynamic Selection", type="secondary", use_container_width=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def parse_tickers(raw: str) -> List[str]:
    return [t.strip().upper() for t in raw.split(",") if t.strip()]

def parse_weights(raw: str, n: int) -> List[float]:
    if not raw.strip():
        return [round(1 / n, 4)] * n
    w = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(w) != n:
        st.warning(f"Weight count ({len(w)}) ≠ ticker count ({n}).  Using equal weights.")
        return [round(1 / n, 4)] * n
    return w


# ---------------------------------------------------------------------------
# Main content tabs
# ---------------------------------------------------------------------------
(
    tab_results,
    tab_convergence,
    tab_data,
    tab_arxiv,
    tab_methodology,
    tab_dynamic,
) = st.tabs([
    "📊 Results",
    "🔄 Convergence",
    "📋 Raw Data",
    "📚 arXiv Papers",
    "📐 Methodology",
    "🧬 Dynamic Factor Selector",
])

# ---------------------------------------------------------------------------
# Run Engines A / B on button click
# ---------------------------------------------------------------------------
if run_btn:
    tickers = parse_tickers(tickers_raw)
    weights = parse_weights(weights_raw, len(tickers))
    if not tickers:
        st.error("Please enter at least one ticker.")
        st.stop()

    with st.spinner("Fetching market data…"):
        try:
            asset_ret = fetch_returns(tickers, lookback_years=lookback)
        except Exception as e:
            st.error(f"Failed to fetch asset data: {e}"); st.stop()
        try:
            factor_ret, factor_labels = fetch_factor_data(model_choice, lookback_years=lookback)
        except Exception as e:
            st.error(f"Failed to fetch factor data: {e}"); st.stop()
        if use_custom_factor and custom_factor_ticker.strip():
            try:
                cf_ret = fetch_returns([custom_factor_ticker.strip().upper()], lookback_years=lookback)
                factor_ret = factor_ret.join(cf_ret, how="inner")
                factor_labels.append(custom_factor_label or custom_factor_ticker)
            except Exception as e:
                st.warning(f"Custom factor fetch failed: {e}")
        asset_ret, factor_ret = align_returns(asset_ret, factor_ret)

    st.session_state.update({
        "asset_ret": asset_ret, "factor_ret": factor_ret,
        "factor_labels": factor_labels, "tickers": tickers,
        "weights": weights, "tau": tau,
    })

    results_jax = results_skl = time_jax = time_skl = None

    if engine_choice.startswith("Bayesline") or run_both:
        with st.spinner("Running Bayesline JAX engine…"):
            try:
                results_jax, time_jax = run_jax_rolling(asset_ret, factor_ret, tau=tau, M=1.35, n_iter=10)
            except Exception as e:
                st.error(f"JAX engine error: {e}")

    if engine_choice.startswith("Scikit-Learn") or run_both:
        with st.spinner("Running Scikit-Learn CPU engine…"):
            try:
                results_skl, time_skl = run_sklearn_rolling(asset_ret, factor_ret, tau=tau, epsilon=1.35)
            except Exception as e:
                st.error(f"Scikit-Learn engine error: {e}")

    st.session_state.update({
        "results_jax": results_jax, "results_skl": results_skl,
        "time_jax": time_jax, "time_skl": time_skl, "run_both": run_both,
    })

# ---------------------------------------------------------------------------
# Run Dynamic Factor Selector on button click
# ---------------------------------------------------------------------------
if dyn_btn:
    dyn_asset_ticker = dyn_asset.strip().upper()
    if not dyn_asset_ticker:
        st.error("Please enter a target asset ticker.")
    else:
        with st.spinner(f"Fetching returns for {dyn_asset_ticker}…"):
            try:
                asset_series_df = fetch_returns([dyn_asset_ticker], lookback_years=dyn_lookback)
                asset_series    = asset_series_df.iloc[:, 0].rename(dyn_asset_ticker)
            except Exception as e:
                st.error(f"Failed to fetch {dyn_asset_ticker}: {e}")
                st.stop()

        custom_extra = [t.strip().upper() for t in dyn_custom.split(",") if t.strip()]
        params = DynamicSelectionParams(
            lookback_years   = dyn_lookback,
            tau              = dyn_tau,
            delta_max        = delta_max,
            gamma            = gamma,
            k_max            = k_max,
            ewma_halflife    = ewma_halflife,
            rebal_step       = rebal_step,
            candidate_themes = dyn_themes,
            custom_tickers   = custom_extra,
        )

        with st.spinner("Running D-Guarded Lasso + MILP Knapsack selection…"):
            try:
                dyn_result = run_dynamic_selection(dyn_asset_ticker, asset_series, params)
                st.session_state["dyn_result"] = dyn_result
                st.success(
                    f"Dynamic selection complete in {dyn_result.elapsed_seconds:.1f}s — "
                    f"{len(dyn_result.factors)} candidate factors, "
                    f"{len(dyn_result.selection_df)} rebalancing dates."
                )
            except Exception as e:
                st.error(f"Dynamic selection failed: {e}")
                import traceback; st.code(traceback.format_exc())


# ===========================================================================
# Tab 1 – Results
# ===========================================================================
with tab_results:
    results_jax = st.session_state.get("results_jax")
    results_skl = st.session_state.get("results_skl")
    time_jax    = st.session_state.get("time_jax")
    time_skl    = st.session_state.get("time_skl")
    run_both_   = st.session_state.get("run_both", False)

    if results_jax is None and results_skl is None:
        st.info("Configure your portfolio and factor model in the sidebar, then click **▶ Run Analysis**.")
    else:
        st.markdown('<p class="section-header">⏱ Performance Metrics</p>', unsafe_allow_html=True)
        mcols = st.columns(4)
        active_results = results_jax if results_jax is not None else results_skl
        active_time    = time_jax    if time_jax    is not None else time_skl
        active_label   = "Bayesline JAX" if results_jax is not None else "Scikit-Learn"
        with mcols[0]: st.metric("Engine", active_label)
        with mcols[1]: st.metric("Execution Time", f"{active_time*1000:.1f} ms" if active_time else "—")
        with mcols[2]: st.metric("Assets", len(active_results["asset"].unique()))
        with mcols[3]: st.metric("Observations", f"{len(active_results):,}")

        if run_both_ and time_jax and time_skl:
            st.markdown('<p class="section-header">🏎 Speed Comparison</p>', unsafe_allow_html=True)
            speedup = time_skl / time_jax if time_jax > 0 else float("inf")
            bc1, bc2, bc3 = st.columns(3)
            with bc1: st.metric("JAX Time",          f"{time_jax*1000:.1f} ms")
            with bc2: st.metric("Scikit-Learn Time", f"{time_skl*1000:.1f} ms")
            with bc3: st.metric("Speed-up",          f"{speedup:.1f}×")
            speed_df = pd.DataFrame({
                "Engine":    ["Bayesline JAX", "Scikit-Learn CPU"],
                "Time (ms)": [time_jax*1000,   time_skl*1000],
            })
            fig_speed = px.bar(speed_df, x="Engine", y="Time (ms)", color="Engine",
                color_discrete_map={"Bayesline JAX":"#1a73e8","Scikit-Learn CPU":"#e65100"},
                title="Execution Time Comparison", text_auto=".1f")
            fig_speed.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_speed, use_container_width=True)

        st.markdown('<p class="section-header">📋 Factor Exposure Table</p>', unsafe_allow_html=True)
        display_df = active_results.copy()
        display_df["date"]      = pd.to_datetime(display_df["date"]).dt.date
        display_df["beta"]      = display_df["beta"].round(4)
        display_df["t_stat"]    = display_df["t_stat"].round(3)
        display_df["p_value"]   = display_df["p_value"].round(4)
        display_df["r_squared"] = display_df["r_squared"].round(4)

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            asset_filter = st.multiselect("Filter by asset",
                sorted(display_df["asset"].unique()),
                default=sorted(display_df["asset"].unique())[:3], key="asset_filter")
        with fc2:
            factor_filter = st.multiselect("Filter by factor",
                sorted(display_df["factor"].unique()),
                default=sorted(display_df["factor"].unique())[:1], key="factor_filter")
        with fc3:
            sig_only = st.checkbox("Significant only (p < 0.05)", value=False, key="sig_only")

        mask = (display_df["asset"].isin(asset_filter) & display_df["factor"].isin(factor_filter))
        if sig_only:
            mask &= display_df["p_value"] < 0.05
        filtered = display_df[mask]

        st.dataframe(filtered.head(500), use_container_width=True, height=400,
            column_config={
                "beta":      st.column_config.NumberColumn("β (Beta)",    format="%.4f"),
                "t_stat":    st.column_config.NumberColumn("t-Stat",      format="%.3f"),
                "p_value":   st.column_config.NumberColumn("p-Value",     format="%.4f"),
                "r_squared": st.column_config.NumberColumn("R²",          format="%.4f"),
            })
        st.download_button("⬇ Download CSV", data=filtered.to_csv(index=False),
            file_name="factor_exposures.csv", mime="text/csv")


# ===========================================================================
# Tab 2 – Convergence
# ===========================================================================
with tab_convergence:
    results_jax = st.session_state.get("results_jax")
    results_skl = st.session_state.get("results_skl")
    if results_jax is None and results_skl is None:
        st.info("Run the analysis first to see convergence charts.")
    else:
        st.markdown('<p class="section-header">🔄 Rolling Beta Time-Series</p>', unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)
        all_assets  = set()
        all_factors = set()
        for r in [results_jax, results_skl]:
            if r is not None:
                all_assets  |= set(r["asset"].unique())
                all_factors |= set(r["factor"].unique())
        with cc1: conv_asset  = st.selectbox("Asset",  sorted(all_assets),  key="conv_asset")
        with cc2: conv_factor = st.selectbox("Factor", sorted(all_factors), key="conv_factor")

        fig = go.Figure()
        for res, label, color, dash in [
            (results_jax, "Bayesline JAX",   "#1a73e8", "solid"),
            (results_skl, "Scikit-Learn CPU","#e65100", "dot"),
        ]:
            if res is not None:
                sub = res[(res["asset"]==conv_asset)&(res["factor"]==conv_factor)].sort_values("date")
                fig.add_trace(go.Scatter(x=sub["date"], y=sub["beta"], name=label,
                    line=dict(color=color, width=2, dash=dash), mode="lines"))
        fig.update_layout(title=f"Rolling β: {conv_asset} ~ {conv_factor}",
            xaxis_title="Date", yaxis_title="Beta Exposure", hovermode="x unified",
            height=450, plot_bgcolor="#fafafa",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<p class="section-header">🗺 Latest Beta Heatmap</p>', unsafe_allow_html=True)
        active = results_jax if results_jax is not None else results_skl
        latest_date = active["date"].max()
        latest = active[active["date"]==latest_date].pivot(index="asset", columns="factor", values="beta")
        fig_heat = px.imshow(latest, color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
            title=f"Factor Beta Heatmap — {latest_date}", aspect="auto", text_auto=".3f")
        fig_heat.update_layout(height=350)
        st.plotly_chart(fig_heat, use_container_width=True)


# ===========================================================================
# Tab 3 – Raw Data
# ===========================================================================
with tab_data:
    asset_ret  = st.session_state.get("asset_ret")
    factor_ret = st.session_state.get("factor_ret")
    if asset_ret is None:
        st.info("Run the analysis first to inspect raw data.")
    else:
        st.markdown('<p class="section-header">Asset Returns</p>', unsafe_allow_html=True)
        st.dataframe(asset_ret.tail(100).round(6), use_container_width=True, height=300)
        st.markdown('<p class="section-header">Factor Returns</p>', unsafe_allow_html=True)
        st.dataframe(factor_ret.tail(100).round(6), use_container_width=True, height=300)
        st.markdown('<p class="section-header">Return Correlation Matrix</p>', unsafe_allow_html=True)
        combined = pd.concat([asset_ret, factor_ret], axis=1).dropna()
        corr = combined.corr()
        fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
            title="Correlation Matrix (Assets + Factors)", text_auto=".2f", aspect="auto")
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)


# ===========================================================================
# Tab 4 – arXiv Papers
# ===========================================================================
with tab_arxiv:
    st.markdown('<p class="section-header">📚 Recent Factor Model Research (arXiv)</p>', unsafe_allow_html=True)
    query_input = st.text_input("Search query",
        value="factor model portfolio construction thematic", key="arxiv_query")
    if st.button("🔍 Search arXiv", key="arxiv_btn"):
        with st.spinner("Querying arXiv…"):
            papers = search_arxiv_papers(query=query_input, max_results=8)
        if papers:
            for p in papers:
                with st.expander(f"📄 {p['title']} ({p['published']})"):
                    st.markdown(f"**Authors:** {', '.join(p['authors'][:4])}")
                    st.markdown(f"**Abstract:** {p['summary']}")
                    st.markdown(f"[View on arXiv]({p['link']})")
        else:
            st.warning("No results returned.  Check your internet connection.")


# ===========================================================================
# Tab 5 – Methodology
# ===========================================================================
with tab_methodology:
    st.markdown(
        r"""
## Mathematical Methodology

### Engine A – Scikit-Learn Baseline (CPU Loop)

The baseline engine wraps `sklearn.linear_model.HuberRegressor` in a nested
Python loop over assets and rolling windows of length $\tau$.  For each window
the Huber loss minimisation problem

$$\min_{\beta,\alpha} \sum_{t=1}^{\tau} \rho_M\!\left(y_t - \alpha - \mathbf{x}_t^\top \boldsymbol{\beta}\right)$$

is solved via the built-in IRLS solver, where

$$\rho_M(r) = \begin{cases} \tfrac{1}{2}r^2 & |r| \le M \\ M|r| - \tfrac{1}{2}M^2 & |r| > M \end{cases}$$

with $M = 1.35$ (95 % asymptotic efficiency relative to OLS under Gaussianity).

---

### Engine B – Bayesline Accelerated Windowed IRLS (JAX)

The key insight is that the rolling weighted OLS update at every time step can
be expressed as **1-D convolutions** with a boxcar kernel
$K = [\underbrace{1,\dots,1}_{\tau}]$, eliminating all Python loops over time.

#### Single-Factor Closed Form

$$\hat{\beta}_t = \frac{\bigl(\mathbf{w}\cdot\mathbf{x}\cdot\mathbf{y}\bigr)*K \;-\; \tfrac{1}{\tau}\bigl(\mathbf{w}\cdot\mathbf{x}*K\bigr)\bigl(\mathbf{w}\cdot\mathbf{y}*K\bigr)}{\bigl(\mathbf{w}\cdot\mathbf{x}^2\bigr)*K \;-\; \tfrac{1}{\tau}\bigl(\mathbf{w}\cdot\mathbf{x}*K\bigr)^2}$$

where $*$ denotes the convolution operator and $\mathbf{w}$ are the Huber
outlier weights (global, computed once per full time-series).

#### Multi-Factor Extension

For $K$ factors the normal equations at each window are

$$\hat{\boldsymbol{\beta}}_t = \left(\mathbf{X}_{t}^\top \mathbf{W}_t \mathbf{X}_{t}\right)^{-1} \mathbf{X}_{t}^\top \mathbf{W}_t \mathbf{y}_t$$

implemented via `jnp.linalg.solve` mapped over the time axis with `vmap`.

---

### Section 5 – D-Guarded Lasso & Knapsack Factor Selection

#### D-Guarded Lasso Path

The regularisation path is generated by `sklearn.linear_model.lasso_path`.
At each step $\lambda_i$ the **multicollinearity guard statistic** is:

$$\delta_\lambda = -\ln \det \rho\!\left(X_{A_\lambda}\right)$$

where $\rho(\cdot)$ is the Pearson correlation matrix of the active columns
$A_\lambda$.  Steps with $\delta_\lambda > \delta_{\max}$ are filtered out.
The selected active set is the largest surviving set.

#### Recursive State EWMA

For each factor $j$ the EWMA t-statistic state is updated daily:

$$\tilde{v}_j^{(t)} = \lambda \cdot \hat{t}_j^{(t)} + (1-\lambda)\,\tilde{v}_j^{(t-1)}, \qquad \lambda = \frac{2}{\text{halflife}+1}$$

alongside a duration counter $d_j^{(t)}$ tracking consecutive inclusion or
exclusion days.

#### MILP Knapsack Optimizer

At each rebalancing date the selection problem is:

$$\max_{s \in \{0,1\}^P} \;\sum_j \tilde{v}_j\,s_j \;-\; \gamma \sum_j |s_j - s_j^*|$$

$$\text{subject to} \quad \sum_j s_j \le k_{\max}$$

The absolute-value turnover term is linearised via auxiliary variables
$\phi_j^+, \phi_j^- \ge 0$ with $\phi_j^+ - \phi_j^- = s_j - s_j^*$,
and the MILP is solved by the HiGHS solver via `scipy.optimize.milp`.
        """
    )


# ===========================================================================
# Tab 6 – Dynamic Factor Selector
# ===========================================================================
with tab_dynamic:
    dyn_result = st.session_state.get("dyn_result")

    if dyn_result is None:
        st.info(
            "Configure the **Dynamic Factor Selector** in the sidebar and click "
            "**🧬 Run Dynamic Selection** to start the D-Guarded Lasso + MILP Knapsack pipeline."
        )
        # Show catalog table as a preview
        st.markdown('<p class="section-header">📚 Factor Catalog Preview</p>', unsafe_allow_html=True)
        meta_all = get_catalog_metadata()
        meta_all = meta_all.reset_index()
        st.dataframe(meta_all, use_container_width=True, height=400,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker"),
                "name":   st.column_config.TextColumn("Name"),
                "theme":  st.column_config.TextColumn("Theme"),
            })
    else:
        # ── Summary metrics ──────────────────────────────────────────────
        st.markdown('<p class="section-header">⚙️ Run Summary</p>', unsafe_allow_html=True)
        p = dyn_result.params
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1: st.metric("Candidate Factors", len(dyn_result.factors))
        with m2: st.metric("Rebalancing Dates", len(dyn_result.selection_df))
        with m3: st.metric("δ_max",  f"{p.delta_max:.2f}")
        with m4: st.metric("γ",      f"{p.gamma:.2f}")
        with m5: st.metric("k_max",  p.k_max)

        st.metric("Elapsed", f"{dyn_result.elapsed_seconds:.1f} s")

        # ── Factor Radar — Stacked Area Chart ────────────────────────────
        st.markdown('<p class="section-header">🎯 Factor Radar — Theme Rotation Over Time</p>',
                    unsafe_allow_html=True)
        st.markdown(
            "Each coloured band shows how many factors from a given theme are "
            "active at each rebalancing date.  Themes that fade out have been "
            "phased out by the D-guard (multicollinearity) or by fading EWMA "
            "t-statistics.  Themes that grow are being rotated in by the MILP."
        )

        radar_df = dyn_result.radar_df.reset_index()
        theme_colors = dyn_result.theme_colors
        themes_in_result = [c for c in radar_df.columns if c != "date"]

        fig_radar = go.Figure()
        for theme in themes_in_result:
            if theme in radar_df.columns and radar_df[theme].sum() > 0:
                fig_radar.add_trace(go.Scatter(
                    x=radar_df["date"],
                    y=radar_df[theme],
                    name=theme,
                    mode="lines",
                    stackgroup="one",
                    line=dict(width=0.5),
                    fillcolor=theme_colors.get(theme, "#999"),
                    line_color=theme_colors.get(theme, "#999"),
                    hovertemplate=f"<b>{theme}</b><br>Date: %{{x}}<br>Active: %{{y}}<extra></extra>",
                ))
        fig_radar.update_layout(
            title="Active Factors by Theme (Stacked Area — Factor Radar)",
            xaxis_title="Rebalancing Date",
            yaxis_title="Number of Active Factors",
            hovermode="x unified",
            height=450,
            plot_bgcolor="#fafafa",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── Selection Heatmap ────────────────────────────────────────────
        st.markdown('<p class="section-header">🗺 Factor Selection Heatmap</p>',
                    unsafe_allow_html=True)
        st.markdown(
            "Each cell is 1 (active) or 0 (inactive).  Rows are rebalancing dates; "
            "columns are candidate factors grouped by theme."
        )

        sel_df = dyn_result.selection_df.copy()
        # Sort columns by theme for readability
        meta_df = dyn_result.factor_meta
        theme_order = ["Market", "Style", "Sector", "Thematic", "Macro", "Alternatives", "Custom"]
        ordered_cols = []
        for theme in theme_order:
            cols_in_theme = [c for c in sel_df.columns
                             if meta_df.loc[c, "theme"] == theme] if theme in theme_order else []
            ordered_cols.extend(cols_in_theme)
        # Append any remaining
        ordered_cols += [c for c in sel_df.columns if c not in ordered_cols]
        sel_df = sel_df[[c for c in ordered_cols if c in sel_df.columns]]

        # Downsample rows if too many
        max_rows = 80
        if len(sel_df) > max_rows:
            step = len(sel_df) // max_rows
            sel_df = sel_df.iloc[::step]

        fig_sel = px.imshow(
            sel_df.T,
            color_continuous_scale=[[0, "#f5f5f5"], [1, "#1a73e8"]],
            aspect="auto",
            title="Factor Inclusion Matrix (blue = active)",
            labels={"x": "Rebalancing Date", "y": "Factor", "color": "Active"},
        )
        fig_sel.update_layout(height=max(300, len(sel_df.columns) * 14 + 100),
                              coloraxis_showscale=False)
        st.plotly_chart(fig_sel, use_container_width=True)

        # ── EWMA State Table ─────────────────────────────────────────────
        st.markdown('<p class="section-header">📊 Final EWMA State</p>', unsafe_allow_html=True)
        state_df = dyn_result.state_df.copy().reset_index()
        state_df = state_df.merge(
            meta_df.reset_index()[["ticker","name","theme"]],
            left_on="factor", right_on="ticker", how="left"
        ).drop(columns=["ticker"])
        state_df["ewma_tstat"] = state_df["ewma_tstat"].round(3)
        state_df = state_df.sort_values("ewma_tstat", ascending=False)
        st.dataframe(
            state_df,
            use_container_width=True,
            height=350,
            column_config={
                "factor":     st.column_config.TextColumn("Ticker"),
                "name":       st.column_config.TextColumn("Name"),
                "theme":      st.column_config.TextColumn("Theme"),
                "ewma_tstat": st.column_config.NumberColumn("EWMA t-Stat", format="%.3f"),
                "duration":   st.column_config.NumberColumn("Duration (days)"),
                "selected":   st.column_config.CheckboxColumn("Currently Active"),
            },
        )

        # ── Lasso Path Chart ─────────────────────────────────────────────
        st.markdown('<p class="section-header">🔬 D-Guarded Lasso Path (Last Window)</p>',
                    unsafe_allow_html=True)
        st.markdown(
            "The regularisation path shows how many factors enter the model as λ decreases "
            "(left → right).  The **red dashed line** marks δ_max; steps to the right of "
            "where the δ curve crosses this threshold are blocked by the D-guard."
        )

        lasso_path_df = build_lasso_path_chart_df(dyn_result.lasso_result)
        if not lasso_path_df.empty:
            fig_lasso = go.Figure()
            # Number of active factors
            fig_lasso.add_trace(go.Scatter(
                x=lasso_path_df["log_alpha"], y=lasso_path_df["n_active"],
                name="Active Factors", mode="lines+markers",
                line=dict(color="#1a73e8", width=2),
                yaxis="y1",
            ))
            # Delta (log-det)
            fig_lasso.add_trace(go.Scatter(
                x=lasso_path_df["log_alpha"], y=lasso_path_df["delta"],
                name="δ (−ln det ρ)", mode="lines",
                line=dict(color="#e65100", width=2, dash="dot"),
                yaxis="y2",
            ))
            # D-guard threshold line
            fig_lasso.add_hline(
                y=p.delta_max, line_dash="dash", line_color="red",
                annotation_text=f"δ_max = {p.delta_max}", yref="y2",
            )
            fig_lasso.update_layout(
                title="Lasso Regularisation Path with D-Guard",
                xaxis_title="log₁₀(λ)  ← less regularised",
                yaxis=dict(title="Active Factors", side="left"),
                yaxis2=dict(title="δ = −ln det ρ", overlaying="y", side="right"),
                hovermode="x unified",
                height=400,
                plot_bgcolor="#fafafa",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_lasso, use_container_width=True)

            # Guard-pass summary
            n_pass = lasso_path_df["guard_pass"].sum()
            n_total = len(lasso_path_df)
            st.caption(
                f"{n_pass}/{n_total} path steps pass the D-guard (δ ≤ {p.delta_max:.2f}).  "
                f"Selected active set: **{', '.join(dyn_result.lasso_result.get('selected_factors', []))}**"
            )

        # ── Coefficient Path Heatmap ─────────────────────────────────────
        st.markdown('<p class="section-header">🌡 Lasso Coefficient Path Heatmap</p>',
                    unsafe_allow_html=True)
        coef_df = build_coef_heatmap_df(dyn_result.lasso_result)
        if not coef_df.empty:
            coef_pivot = coef_df.pivot(index="factor", columns="log_alpha", values="coefficient")
            # Round column labels for display
            coef_pivot.columns = [f"{c:.2f}" for c in coef_pivot.columns]
            fig_coef = px.imshow(
                coef_pivot,
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                aspect="auto",
                title="Lasso Coefficient Path (rows=factors, columns=log₁₀λ)",
                labels={"x": "log₁₀(λ)", "y": "Factor", "color": "Coef"},
            )
            fig_coef.update_layout(height=max(300, len(coef_pivot) * 18 + 100))
            st.plotly_chart(fig_coef, use_container_width=True)

        # ── Download ─────────────────────────────────────────────────────
        st.markdown('<p class="section-header">⬇ Export</p>', unsafe_allow_html=True)
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "Download Selection Matrix (CSV)",
                data=dyn_result.selection_df.to_csv(),
                file_name="factor_selection.csv",
                mime="text/csv",
            )
        with dl2:
            st.download_button(
                "Download EWMA State (CSV)",
                data=dyn_result.state_df.to_csv(),
                file_name="ewma_state.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<small>Financial Factor Exposure Platform · "
    "Powered by JAX, Scikit-Learn, yfinance, PuLP & Streamlit</small>",
    unsafe_allow_html=True,
)
