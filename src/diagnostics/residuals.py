import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from src.utils.config import DATA_PROCESSED
from src.models.arma_garch import fit_arma_garch


def load_returns():
    file_path = DATA_PROCESSED / "sp500_returns.csv"
    df = pd.read_csv(file_path, parse_dates=["Date"])
    df = df.sort_values("Date")
    return df


def extract_residuals():
    result = fit_arma_garch()

    resid = result.resid
    cond_vol = result.conditional_volatility

    std_resid = resid / cond_vol

    resid = resid.dropna()
    std_resid = std_resid.replace([float("inf"), float("-inf")], pd.NA).dropna()

    return resid, std_resid


def ljung_box_tests(std_resid, lags=20):
    print("\n[LJUNG-BOX TEST — Standardized Residuals]")
    lb_res = acorr_ljungbox(std_resid, lags=[lags], return_df=True)
    print(lb_res)

    print("\n[LJUNG-BOX TEST — Squared Standardized Residuals]")
    lb_sq = acorr_ljungbox(std_resid**2, lags=[lags], return_df=True)
    print(lb_sq)


def arch_test_residuals(resid, lags=10):
    print("\n[ARCH TEST — Residuals]")
    test = het_arch(resid, nlags=lags)

    print(f"LM Statistic: {test[0]}")
    print(f"p-value: {test[1]}")
    print(f"F-statistic: {test[2]}")
    print(f"F-test p-value: {test[3]}")

    if test[1] < 0.05:
        print("→ Remaining ARCH effect present (model insufficient)")
    else:
        print("→ No ARCH effect (model captured volatility)")


def run_residual_diagnostics():
    resid, std_resid = extract_residuals()

    ljung_box_tests(std_resid)
    arch_test_residuals(resid)