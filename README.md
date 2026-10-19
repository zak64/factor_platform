# Financial Factor Exposure Platform

A high-performance quantitative finance application implementing the **Bayesline Windowed IRLS Huber Regression** methodology, benchmarked against a Scikit-Learn CPU baseline, with a fully interactive Streamlit dashboard.

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Architecture

```
factor_platform/
├── app.py              # Streamlit UI (all 5 panels)
├── engine_jax.py       # Engine B: JAX-accelerated Windowed IRLS
├── engine_sklearn.py   # Engine A: Scikit-Learn Huber baseline
├── data_sourcing.py    # yfinance ingestion + factor model registry + arXiv agent
├── requirements.txt    # Python dependencies
├── portfolios/         # Saved portfolio JSON files
└── data/               # (optional) cached data
```

---

## Engines

### Engine A — Scikit-Learn Baseline (CPU Loop)

- Wraps `sklearn.linear_model.HuberRegressor` in a Python loop over assets and rolling windows
- Provides the ground-truth reference for convergence validation
- Typical speed: **3–5 seconds** for 5 assets × 500 days × τ=60

### Engine B — Bayesline JAX Accelerated

- Implements **Windowed IRLS** via 1-D boxcar convolutions (`jax.lax.conv_general_dilated`)
- Entire IRLS loop compiled end-to-end with `jax.jit` + `jax.lax.scan`
- Multi-factor normal equations solved with `jnp.linalg.solve` mapped via `vmap`
- Typical speed: **< 1 second** on CPU; significantly faster on GPU
- **Speed-up: 4–10× on CPU; 50–200× on GPU** (depending on hardware)

---

## Mathematical Methodology

### Huber Loss

$$\rho_M(r) = \begin{cases} \tfrac{1}{2}r^2 & |r| \le M \\ M|r| - \tfrac{1}{2}M^2 & |r| > M \end{cases}$$

### Windowed IRLS Closed Form (Single Factor)

$$\hat{\beta}_t = \frac{(\mathbf{w}\cdot\mathbf{x}\cdot\mathbf{y})*K - \tfrac{1}{\tau}(\mathbf{w}\cdot\mathbf{x}*K)(\mathbf{w}\cdot\mathbf{y}*K)}{(\mathbf{w}\cdot\mathbf{x}^2)*K - \tfrac{1}{\tau}(\mathbf{w}\cdot\mathbf{x}*K)^2}$$

where $K = [1,\dots,1]$ is a boxcar kernel of length $\tau$ and $*$ denotes convolution.

---

## Factor Model Registry

| Model | Factors |
|---|---|
| Market Only (CAPM) | SPY |
| Fama-French 3-Factor Proxy | SPY, IWM, IWD |
| Fama-French 5-Factor Proxy | SPY, IWM, IWD, MTUM, QUAL |
| Macro Model | SPY, USO, GLD, TLT |
| Thematic AI / Semiconductor | SPY, SMH, SOXX, ARKK |
| Energy Transition | SPY, XLE, ICLN, LIT |
| Multi-Asset Macro | SPY, GLD, USO, TLT, UUP, VNQ |

---

## UI Panels

| Panel | Description |
|---|---|
| Portfolio Manager | Input tickers/weights, save/reload portfolios as JSON |
| Factor Selection | Choose factor model, rolling window τ, optional custom ticker |
| Computation Engine | Select JAX or Scikit-Learn, optionally run both for comparison |
| Results Dashboard | Timing metrics, factor exposure table, CSV download |
| Convergence Chart | Rolling beta time-series overlay + beta heatmap |
| Raw Data | Asset/factor return tables + correlation matrix |
| arXiv Papers | Live search of recent factor model research papers |
| Methodology | Full mathematical derivation with LaTeX equations |

---

## Dependencies

| Package | Purpose |
|---|---|
| `jax` / `jaxlib` | JIT-compiled IRLS engine |
| `scikit-learn` | Huber regression baseline |
| `yfinance` | Market data ingestion |
| `streamlit` | Interactive web UI |
| `plotly` | Interactive charts |
| `scipy` | t-statistics, p-values |
| `pandas` / `numpy` | Data manipulation |
| `requests` | arXiv API queries |
