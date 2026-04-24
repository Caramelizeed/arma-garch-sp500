import pandas as pd
from statsmodels.tsa.stattools import adfuller

from src.utils.config import DATA_PROCESSED


def load_returns():
    file_path = DATA_PROCESSED / "sp500_returns.csv"

    if not file_path.exists():
        raise FileNotFoundError("Processed data not found. Run preprocess first.")

    df = pd.read_csv(file_path, parse_dates=["Date"])
    df = df.sort_values("Date")

    return df


def run_adf_test(series, name="Series"):
    result = adfuller(series)

    print(f"\n[ADF TEST] {name}")
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print(f"Used Lags: {result[2]}")
    print(f"Number of Observations: {result[3]}")

    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value}")

    if result[1] < 0.05:
        print("→ Result: Stationary (reject H0)")
    else:
        print("→ Result: Non-stationary (fail to reject H0)")


def test_sp500_returns():
    df = load_returns()
    run_adf_test(df["log_return"], name="S&P 500 Log Returns")