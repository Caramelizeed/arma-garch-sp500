from src.data.fetch import fetch_sp500
if  __name__ == "__main__":
    # Fetch and save S&P 500 data
    sp500_data = fetch_sp500()
    print(sp500_data.head())