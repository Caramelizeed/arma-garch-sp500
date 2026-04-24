from src.data.fetch import fetch_sp500
from src.data.preprocess import preprocess_sp500

if __name__ == "__main__":
    fetch_sp500()
    df = preprocess_sp500()
    print(df.head())