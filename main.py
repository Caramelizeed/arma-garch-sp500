from src.data.fetch import fetch_sp500
from src.data.preprocess import preprocess_sp500

from src.diagnostics.stationarity import test_sp500_returns
from src.diagnostics.acf_pacf import run_acf_pacf_analysis
from src.diagnostics.arch_test import test_arch_effects
from src.diagnostics.residuals import run_residual_diagnostics

from src.models.forecast import forecast_volatility, plot_volatility
from src.models.rolling_forecast import (
    plot_rolling_forecast,
    rolling_var_backtest
)
from src.models.evaluate import evaluate_forecasts

from src.models.kupiec import kupiec_test


if __name__ == "__main__":
    # =========================
    # 1. DATA PIPELINE
    # =========================
    fetch_sp500()
    df = preprocess_sp500()

    # =========================
    # 2. DIAGNOSTICS (run once)
    # =========================
    test_sp500_returns()
    run_acf_pacf_analysis()
    test_arch_effects()
    run_residual_diagnostics()

    # =========================
    # 3. VOLATILITY MODEL
    # =========================
    plot_volatility()
    forecast_volatility(horizon=5)

    # =========================
    # 4. ROLLING FORECAST
    # =========================
    plot_rolling_forecast()

    # =========================
    # 5. FORECAST EVALUATION
    # =========================
    evaluate_forecasts()

    # =========================
    # 6. FINAL RISK MODEL (IMPORTANT)
    # =========================
    rolling_var_backtest(alpha=0.05, dist="normal")

    kupiec_test(alpha=0.05, dist="normal", window=1000)

    print(df.head())