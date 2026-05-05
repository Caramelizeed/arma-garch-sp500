# ARMA–GJR-GARCH Risk Modeling on S&P 500 Returns

## Overview

This project implements a complete econometric and risk modeling pipeline on S&P 500 daily returns using an AR–GJR-GARCH framework.

The system goes beyond volatility modeling and includes:

- conditional mean modeling (AR)
- conditional volatility modeling (GJR-GARCH)
- rolling volatility forecasting
- Value-at-Risk (VaR) estimation
- Expected Shortfall (ES)
- statistical backtesting (Kupiec & Christoffersen tests)

---

## Objectives

- Model time-varying volatility in financial returns
- Quantify market risk using VaR and ES
- Validate models using rigorous statistical tests
- Build a research-grade quantitative risk pipeline

---

## Data

- **Source:** Yahoo Finance (`^GSPC`)
- **Frequency:** Daily
- **Period:** ~2010–present

Stored in:

- `data/raw/` → raw OHLCV data
- `data/processed/` → cleaned log returns

---

## Methodology

### 1. Preprocessing

Log returns computed as:

```
r_t = log(P_t / P_{t-1})
```

- Missing values removed
- Chronological ordering enforced

---

### 2. Stationarity (ADF Test)

- Null: non-stationary
- Result: rejected → returns are stationary

---

### 3. ARCH Effect Detection

- Strong ARCH effects detected
- Confirms volatility clustering
- Justifies GARCH-family models

---

### 4. Model Specification

**Mean Model:** AR(1)

**Volatility Model:** GJR-GARCH(1,1)

**Equations:**

```
r_t = μ + φ r_{t-1} + ε_t
σ_t² = ω + α ε_{t-1}² + γ I(ε_{t-1}<0) ε_{t-1}² + β σ_{t-1}²
```

---

### 5. Key Properties

- High persistence: α + β ≈ 0.97
- Significant asymmetry (γ > 0)
- Negative shocks increase volatility more than positive shocks

---

## Diagnostics

### Residual Tests

| Test | Result |
|------|--------|
| Ljung-Box (residuals) | ✔ no autocorrelation |
| Ljung-Box (squared residuals) | ✔ volatility captured |
| ARCH test (residuals) | ⚠ minor remaining effects |

---

## Forecasting

Rolling volatility forecasts implemented.

**Evaluation Metrics:**

| Metric | Value |
|--------|-------|
| MSE    | ~0.63 |
| MAE    | ~0.56 |
| QLIKE  | ~0.71 |

---

## Risk Modeling

### Rolling VaR (5%)

- Observed violation rate: **5.55%**
- Expected: **5.00%**

> Slight underestimation of risk — within acceptable tolerance.

---

### Kupiec Test (Unconditional Coverage)

- LR = 1.91 < 3.84 → **ACCEPT**

> Correct violation frequency confirmed.

---

### Christoffersen Test (Independence)

- LR = 0.036 < 3.84 → **ACCEPT**

> No clustering of violations detected.

---

### Expected Shortfall (ES)

| Metric       | Value  |
|--------------|--------|
| Avg Tail Loss | -2.187 |
| ES Estimate  | -2.108 |
| Difference   | +0.079 |

> Mild underestimation (~3–4%), consistent with VaR behavior.

---

## Final Model Assessment

The model satisfies:

- ✔ Correct violation frequency
- ✔ Independent violations
- ✔ Reasonable tail estimation
- ✔ Stable volatility forecasts

**Conclusion:**

> Statistically validated and suitable as a baseline risk model.

---

## Project Structure

```
arma-garch-sp500/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data/
│   ├── diagnostics/
│   ├── models/
│   └── utils/
│
├── main.py
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
python main.py
```

**Pipeline includes:**

1. Data ingestion
2. Preprocessing
3. Diagnostics
4. Model fitting
5. Rolling forecasts
6. VaR & ES computation
7. Statistical validation

---

## Dependencies

```
numpy
pandas
statsmodels
arch
yfinance
matplotlib
scipy
```

Install via:

```bash
pip install -r requirements.txt
```

---

## Limitations

- Residual ARCH effects remain in standardized residuals
- Normal distribution assumption underestimates extreme tails slightly
- Single-asset model — no portfolio extension

---

## Future Work

- EGARCH / advanced asymmetric volatility models
- Student-t / skewed-t error distributions
- Portfolio-level VaR & ES
- Extreme Value Theory (EVT)
- Volatility-based trading strategies

---

## Notes

- Focus is on **risk modeling**, not return prediction
- Designed as a **research-grade baseline** for quantitative finance workflows