import numpy as np
from scipy.stats import chi2

from src.models.rolling_forecast import rolling_var


def kupiec_test(alpha=0.05, dist="normal", window=1000):
    var_series, actuals = rolling_var(window=window, alpha=alpha, dist=dist)

    violations = actuals < var_series

    T = len(violations)
    N = violations.sum()
    p_hat = N / T

    # Avoid log(0)
    eps = 1e-10
    p_hat = max(min(p_hat, 1 - eps), eps)

    # Likelihood ratio
    LR = -2 * (
        (T - N) * np.log((1 - alpha) / (1 - p_hat)) +
        N * np.log(alpha / p_hat)
    )

    # Critical value (95% confidence)
    crit = chi2.ppf(0.95, df=1)

    print("\n[KUPIEC TEST]")
    print(f"Expected (alpha) : {alpha}")
    print(f"Observed (p_hat) : {p_hat:.4f}")
    print(f"Violations       : {N} / {T}")
    print(f"LR statistic     : {LR:.4f}")
    print(f"Critical value   : {crit:.4f}")

    if LR < crit:
        print("→ Result: ACCEPT model (VaR is calibrated)")
    else:
        print("→ Result: REJECT model (VaR is not calibrated)")