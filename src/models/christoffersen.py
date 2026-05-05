import numpy as np
from scipy.stats import chi2

from src.models.rolling_forecast import rolling_var


def christoffersen_test(window=1000, alpha=0.05, dist="normal"):
    var_series, actuals = rolling_var(window=window, alpha=alpha, dist=dist)

    violations = (actuals < var_series).astype(int)

    # transition counts
    n00 = n01 = n10 = n11 = 0

    for i in range(1, len(violations)):
        prev = violations.iloc[i - 1]
        curr = violations.iloc[i]

        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        elif prev == 1 and curr == 1:
            n11 += 1

    # probabilities
    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi  = (n01 + n11) / (n00 + n01 + n10 + n11)

    # avoid log(0)
    eps = 1e-10
    pi0 = np.clip(pi0, eps, 1 - eps)
    pi1 = np.clip(pi1, eps, 1 - eps)
    pi  = np.clip(pi,  eps, 1 - eps)

    # likelihood ratio
    L_ind = (
        (1 - pi) ** (n00 + n10) *
        (pi) ** (n01 + n11)
    )

    L_dep = (
        (1 - pi0) ** n00 *
        (pi0) ** n01 *
        (1 - pi1) ** n10 *
        (pi1) ** n11
    )

    LR = -2 * np.log(L_ind / L_dep)

    crit = chi2.ppf(0.95, df=1)

    print("\n[CHRISTOFFERSEN TEST — INDEPENDENCE]")
    print(f"n00={n00}, n01={n01}, n10={n10}, n11={n11}")
    print(f"pi0={pi0:.4f}, pi1={pi1:.4f}, pi={pi:.4f}")
    print(f"LR statistic   : {LR:.4f}")
    print(f"Critical value : {crit:.4f}")

    if LR < crit:
        print("→ Result: ACCEPT (violations are independent)")
    else:
        print("→ Result: REJECT (violations are clustered)")