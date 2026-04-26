from src.data.fetch import fetch_sp500
from src.data.preprocess import preprocess_sp500
from src.diagnostics.stationarity import test_sp500_returns
from src.diagnostics.acf_pacf import run_acf_pacf_analysis
from src.diagnostics.arch_test import test_arch_effects
from src.models.arma_garch import fit_arma_garch
from src.diagnostics.residuals import run_residual_diagnostics
from src.models.forecast import forecast_volatility, plot_volatility

if __name__ == "__main__":
    fetch_sp500()
    df = preprocess_sp500()
    test_sp500_returns()
    run_acf_pacf_analysis()
    test_arch_effects()
    fit_arma_garch()
    run_residual_diagnostics()
    plot_volatility()
    forecast_volatility(horizon=5)
    print(df.head())