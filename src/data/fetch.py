import yfinance as yf
import pandas as pd
from datetime import datetime

from src.utils.config import DATA_RAW


def fetch_sp500(start="2010-01-01", end=None, force=False):
    """
    Fetch S&P 500 data and store it in data/raw/sp500.csv

    Parameters:
    - start (str): start date in YYYY-MM-DD
    - end (str or None): end date (default = today)
    - force (bool): overwrite existing file if True
    """

    file_path = DATA_RAW / "sp500.csv"

    # Avoid redundant downloads
    if file_path.exists() and not force:
        print(f"[INFO] Using cached data at {file_path}")
        return pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

    # Set end date dynamically
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    print("[INFO] Downloading data for S&P 500 (^GSPC)...")

    try:
        data = yf.download("^GSPC", start=start, end=end, progress=False)
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")
    # Handle MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Validation checks
    if data is None or data.empty:
        raise ValueError("Downloaded data is empty. Check internet or ticker.")

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(data.columns):
        raise ValueError(f"Missing expected columns. Found: {data.columns}")

    # Clean dataset
    data = data.dropna()
    data = data.reset_index()
    

    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    data.to_csv(file_path, index=False)

    print(f"[INFO] Data saved to {file_path}")
    print(f"[INFO] Rows: {len(data)} | Columns: {list(data.columns)}")

    return data