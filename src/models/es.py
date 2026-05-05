import numpy as np
from src.models.rolling_forecast import rolling_var


def rolling_es(window=1000, alpha=0.05, dist="normal"):
    var_series, actuals = rolling_var(window=window, alpha=alpha, dist=dist)

    es_values = []

    var_arr = var_series.values
    actual_arr = actuals.values

    for i in range(len(actual_arr)):
        start = max(0, i - window)

        window_actual = actual_arr[start:i+1]
        window_var = var_arr[start:i+1]

        mask = window_actual < window_var

        if np.sum(mask) > 0:
            es = window_actual[mask].mean()
        else:
            es = np.nan

        es_values.append(es)

    return np.array(es_values), var_arr, actual_arr

def es_backtest(window=1000, alpha=0.05, dist="normal"):
    es, var, actual = rolling_es(window, alpha, dist)

    # focus only where ES is defined
    valid = ~np.isnan(es)

    es = es[valid]
    actual = actual[valid]
    var = var[valid]

    # violations (same as VaR)
    violations = actual < var

    # ES condition: actual losses should be close to ES
    tail_losses = actual[violations]
    avg_tail_loss = tail_losses.mean()

    avg_es = es[violations].mean()

    print("\n[EXPECTED SHORTFALL (ES)]")
    print(f"Alpha             : {alpha}")
    print(f"Avg Tail Loss     : {avg_tail_loss:.4f}")
    print(f"Avg ES Estimate   : {avg_es:.4f}")
    print(f"Difference        : {avg_es - avg_tail_loss:.4f}")