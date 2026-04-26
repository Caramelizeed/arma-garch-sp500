import pandas as pd
import matplotlib.pyplot as plt

from src.models.arma_garch import fit_arma_garch
from src.data.preprocess import preprocess_sp500


def forecast_volatility(horizon=5):
    # Ensure latest data
    df = preprocess_sp500()

    result = fit_arma_garch()

    forecasts = result.forecast(horizon=horizon)

    # Extract variance forecasts
    variance = forecasts.variance.iloc[-1]

    # Convert to volatility
    volatility = variance ** 0.5

    print("\n[Volatility Forecast]")
    print(volatility)

    return volatility

def plot_volatility():
    df = preprocess_sp500()
    result = fit_arma_garch()

    cond_vol = result.conditional_volatility

    plt.figure(figsize=(12, 5))
    plt.plot(cond_vol)
    plt.title("Conditional Volatility (GJR-GARCH)")
    plt.xlabel("Time")
    plt.ylabel("Volatility")
    plt.show()