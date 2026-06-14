import numpy as np
import pandas as pd

def compute_log_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'log_returns' column to the given DataFrame based on the 'close' prices.
    """
    data["log_returns"] = np.log(data["close"] / data["close"].shift(1))
    return data.dropna()
