import numpy as np
from src.models.rolling_forecast import rolling_volatility_forecast


def evaluate_forecasts():
    forecast_vol, actual_vol = rolling_volatility_forecast()

    # Convert to numpy arrays
    f = forecast_vol.values
    a = actual_vol.values

    # MSE
    mse = np.mean((f - a) ** 2)

    # MAE
    mae = np.mean(np.abs(f - a))

    # QLIKE (use squared returns for consistency)
    qlike = np.mean(np.log(f**2) + (a**2) / (f**2))

    print("\n[Forecast Evaluation]")
    print(f"MSE   : {mse:.4f}")
    print(f"MAE   : {mae:.4f}")
    print(f"QLIKE : {qlike:.4f}")