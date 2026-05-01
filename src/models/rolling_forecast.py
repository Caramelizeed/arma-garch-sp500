from matplotlib.pylab import norm
import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt

from src.utils.config import DATA_PROCESSED


def load_returns():
    df = pd.read_csv(DATA_PROCESSED / "sp500_returns.csv", parse_dates=["Date"])
    df = df.sort_values("Date")
    return df


def plot_rolling_forecast():
    forecast_vol, actual_vol = rolling_volatility_forecast()

    plt.figure(figsize=(12, 5))
    plt.plot(forecast_vol, label="Forecast Volatility")
    plt.plot(actual_vol, label="Actual (|returns|)", alpha=0.6)

    plt.title("Rolling Volatility Forecast vs Actual")
    plt.legend()
    plt.show()

def rolling_volatility_forecast(window=1000, horizon=1, step=5):
    df = load_returns()
    returns = df["log_return"] * 100

    forecasts = []
    actuals = []

    for i in range(window, len(returns) - horizon, step):
        train = returns[i - window:i]

        model = arch_model(
            train,
            mean="ARX",
            lags=1,
            vol="GARCH",
            p=1,
            o=1,
            q=1,
            dist="t"
        )

        result = model.fit(disp="off")

        fcast = result.forecast(horizon=horizon)
        var = fcast.variance.iloc[-1, 0]

        forecasts.append(np.sqrt(var))
        actuals.append(abs(returns.iloc[i]))

    index = returns.index[window:len(returns)-horizon:step]

    return pd.Series(forecasts, index=index), pd.Series(actuals, index=index)

def rolling_var(window=1000, alpha=0.05, dist="normal"):
    df = load_returns()
    returns = df["log_return"] * 100

    var_series = []
    actuals = []

    for i in range(window, len(returns) - 1):
        train = returns[i - window:i]

        model = arch_model(
            train,
            mean="ARX",
            lags=1,
            vol="GARCH",
            p=1,
            o=1,
            q=1,
            dist=dist
        )

        res = model.fit(disp="off")

        # 1-step ahead forecast
        fcast = res.forecast(horizon=1)
        sigma = np.sqrt(fcast.variance.iloc[-1, 0])

        # distribution-specific quantile
        if dist == "t":
            nu = res.params["nu"]
            q = t.ppf(alpha, df=nu) * np.sqrt((nu - 2) / nu)
        else:
            q = norm.ppf(alpha)

        var = sigma * q  # mean ≈ 0

        var_series.append(var)
        actuals.append(returns.iloc[i])

    index = returns.index[window:len(returns)-1]

    return pd.Series(var_series, index=index), pd.Series(actuals, index=index)

def rolling_var_backtest(window=1000, alpha=0.05, dist="normal"):
    var_series, actuals = rolling_var(window, alpha, dist)

    violations = actuals < var_series
    rate = violations.mean()

    print("\n[Rolling VaR Backtest]")
    print(f"Distribution : {dist}")
    print(f"Expected     : {alpha}")
    print(f"Observed     : {rate:.4f}")
    print(f"Violations   : {violations.sum()} / {len(violations)}")
    return rolling_var_backtest

#gonnna start working no the parametre