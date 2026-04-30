import pandas as pd
from arch import arch_model

from src.utils.config import DATA_PROCESSED
#till now this model capture the rt​=μ+ϕrt−1​+ϵt​ and also this σt2​=ω+αϵt−12​+βσt−12​

def load_returns():
    file_path = DATA_PROCESSED / "sp500_returns.csv"

    df = pd.read_csv(file_path, parse_dates=["Date"])
    df = df.sort_values("Date")

    return df


def fit_arma_garch():
    df = load_returns()
    returns = df["log_return"] * 100

    model = arch_model(
        returns,
        mean="ARX",
        lags=1,
        vol="GARCH",
        p=1,
        o=1,          # 🔴 adds GJR asymmetry term
        q=1,
        dist="normal"      # keep Student-t
    )

    result = model.fit(disp="off")
    print(result.summary())
    return result