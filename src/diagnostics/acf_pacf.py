import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from src.utils.config import DATA_PROCESSED

def load_returns():
    file_path = DATA_PROCESSED / "sp500_returns.csv"

    df = pd.read_csv(file_path, parse_dates=["Date"])
    df = df.sort_values("Date")

    return df 

def plot_acf_pacf(series):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    plot_acf(series, ax=axes[0], lags=40)
    axes[0].set_title("ACF of S&P 500 Log Returns")

    plot_pacf(series, ax=axes[1], lags=40, method='ywm')
    axes[1].set_title("PACF of S&P 500 Log Returns")

    plt.tight_layout()
    plt.show()

def run_acf_pacf_analysis():
    df = load_returns()
    plot_acf_pacf(df["log_return"])