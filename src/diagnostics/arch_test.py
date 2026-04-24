import pandas as pd
from statsmodels.stats.diagnostic import het_arch

from src.utils.config import DATA_PROCESSED

def load_returns():
    file_path = DATA_PROCESSED / "sp500_returns.csv"

    if not file_path.exists():
        raise FileNotFoundError("Processed data not found. Run preprocess first.")

    df = pd.read_csv(file_path, parse_dates=["Date"])
    df = df.sort_values("Date")

    return df

def run_arch_test(series, lags=10):
    test = het_arch(series, nlags=lags)

    print("\n[ARCH TEST]")
    print(f"LM Statistic: {test[0]}")
    print(f"p-value: {test[1]}")
    print(f"F-Statistic: {test[2]}")
    print(f"F-test p-value: {test[3]}")

    if test[1] < 0.05:
        print("→ Result: ARCH effects present (reject H0)")
    else:        print("→ Result: No ARCH effects (fail to reject H0)")

def test_arch_effects():
    df = load_returns()
    run_arch_test(df["log_return"],lags=10)