import pandas as pd
import numpy as np

from src.utils.config import DATA_RAW, DATA_PROCESSED


def load_raw_data():
    file_path = DATA_RAW / "sp500.csv"

    if not file_path.exists():
        raise FileNotFoundError("Raw data not found. Run fetch.py first.")

    df = pd.read_csv(file_path, parse_dates=["Date"])
    df = df.sort_values("Date")

    return df


def compute_log_returns(df):
    if "Close" not in df.columns:
        raise ValueError("Close column missing in raw data.")

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna()

    return df


def save_processed_data(df):
    file_path = DATA_PROCESSED / "sp500_returns.csv"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    df.columns.name = None  # Remove column name if exists

    df.to_csv(file_path, index=False)

    print(f"[INFO] Processed data saved to {file_path}")
    print(f"[INFO] Rows: {len(df)}")


def preprocess_sp500():
    df = load_raw_data()
    df = compute_log_returns(df)
    save_processed_data(df)

    return df