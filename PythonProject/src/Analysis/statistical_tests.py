from statsmodels.tsa.stattools import adfuller
import numpy as np

def run_adf_test(series):
    result = adfuller(series, autolag="AIC")

    # Clean the critical values (convert np.float64 → float)
    critical_vals = {k: round(float(v), 3) for k, v in result[4].items()}

    return {
        "adf_stat": float(result[0]),
        "p_value": float(result[1]),
        "critical_values": critical_vals
    }

def hurst_exponent(ts):
    lags = range(2, 100)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

