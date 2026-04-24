# ARMA–GARCH Modeling on S&P 500 Returns

## Overview

This project implements a full econometric pipeline to model financial time series using an ARMA–GARCH framework on S&P 500 daily returns.

The objective is to:

* model the conditional mean (AR component)
* model time-varying volatility (GARCH component)
* validate statistical assumptions rigorously before fitting

---

## Data

* Source: Yahoo Finance (`^GSPC`)
* Frequency: Daily
* Period: ~2010–present
* Stored in:

  * `data/raw/` → original prices
  * `data/processed/` → log returns

---

## Pipeline

### 1. Data Ingestion

* Fetch OHLCV data
* Handle API inconsistencies (MultiIndex normalization)
* Store locally for reproducibility

### 2. Preprocessing

* Compute log returns:

  r_t = log(P_t / P_{t-1})

* Remove NaNs

* Ensure time ordering

### 3. Stationarity Test (ADF)

* Null: non-stationary
* Result: rejected → returns are stationary

### 4. ACF / PACF Analysis

* Minimal autocorrelation observed
* Weak AR(1) structure at most

### 5. ARCH Test

* Strong ARCH effects detected
* Confirms volatility clustering

### 6. ARMA–GARCH Model

* Mean: AR(1)
* Volatility: GARCH(1,1)

Model form:

r_t = μ + φ r_{t-1} + ε_t
σ_t² = ω + α ε_{t-1}² + β σ_{t-1}²

---

## Key Results

* Returns are stationary
* Mean predictability is weak
* Strong volatility clustering observed
* GARCH parameters show high persistence:

  α + β ≈ 0.97

Interpretation:

* market direction is largely unpredictable
* volatility (risk) is predictable

---

## Project Structure

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

---

## How to Run

```bash
python main.py
```

Pipeline executed:

* data fetch
* preprocessing
* ADF test
* ACF/PACF analysis
* ARCH test
* ARMA–GARCH fit

---

## Dependencies

* numpy
* pandas
* statsmodels
* arch
* yfinance
* matplotlib

---

## Next Steps

* Residual diagnostics
* Student-t GARCH (fat tails)
* Volatility forecasting
* Backtesting strategies based on volatility

---

## Notes

* This project focuses on volatility modeling, not return prediction
* Designed as a research-grade baseline for quantitative finance workflows
