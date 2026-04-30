import pandas as pd
import numpy as np
from scipy.stats import alpha, t
import matplotlib.pyplot as plt

from src.models.arma_garch import fit_arma_garch
from src.data.preprocess import preprocess_sp500


def compute_var(alpha=0.05):
    df = preprocess_sp500()
    result = fit_arma_garch()

    returns = df["log_return"] * 100
    sigma = result.conditional_volatility

    mu = 0

    if "nu" in result.params:
        # Student-t
        nu = result.params["nu"]
        q = t.ppf(alpha, df=nu) * np.sqrt((nu - 2) / nu)
    else:
        # Normal
        from scipy.stats import norm
        q = norm.ppf(alpha)

    var = mu + sigma * q

    df["VaR"] = var
    df["returns"] = returns

    return df

def plot_var(df, alpha=0.05):
    plt.figure(figsize=(12, 5))
    plt.plot(df["returns"], label="Returns", alpha=0.6)
    plt.plot(df["VaR"], label=f"VaR ({int(alpha*100)}%)")

    plt.title("Value at Risk (VaR)")
    plt.legend()
    plt.show()

def var_backtest(df, alpha=0.05):
    df = df.dropna()

    violations = df["returns"] < df["VaR"]
    violation_rate = violations.mean()

    print("\n[VaR Backtest]")
    print(f"Expected rate : {alpha}")
    print(f"Observed rate : {violation_rate:.4f}")
    print(f"Violations    : {violations.sum()} / {len(violations)}")

