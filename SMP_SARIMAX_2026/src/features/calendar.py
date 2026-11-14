from __future__ import annotations

import numpy as np
import pandas as pd


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    idx = pd.DatetimeIndex(index)
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    doy = idx.dayofyear.to_numpy()
    month = idx.month.to_numpy()
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7),
        "year_sin": np.sin(2 * np.pi * doy / 365.25), "year_cos": np.cos(2 * np.pi * doy / 365.25),
        "month": month,
    }, index=idx)
