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