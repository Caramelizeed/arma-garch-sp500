from src.data.fetch import fetch_sp500
from src.data.preprocess import preprocess_sp500
from src.diagnostics.stationarity import test_sp500_returns
from src.diagnostics.acf_pacf import run_acf_pacf_analysis
from src.diagnostics.arch_test import test_arch_effects

if __name__ == "__main__":
    fetch_sp500()
    df = preprocess_sp500()
    test_sp500_returns()
    run_acf_pacf_analysis()
    test_arch_effects()
    print(df.head())