from __future__ import annotations

from dataclasses import dataclass
import time

import pandas as pd

from src.features.calendar import calendar_features


@dataclass(frozen=True)
class Candidate:
    name: str
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]


CANDIDATES = [
    Candidate("SARIMAX_100_100_24", (1, 0, 0), (1, 0, 0, 24)),
    Candidate("SARIMAX_001_001_24", (0, 0, 1), (0, 0, 1, 24)),
]


def fit(y: pd.Series, candidate: Candidate):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    X = calendar_features(pd.DatetimeIndex(y.index))
    start = time.perf_counter()
    model = SARIMAX(y, exog=X, order=candidate.order, seasonal_order=candidate.seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False,
                    simple_differencing=True, concentrate_scale=False)
    result = model.fit(disp=False, maxiter=25, method="lbfgs")
    elapsed = time.perf_counter() - start
    return result, elapsed


def forecast_72(result, origin: pd.Timestamp) -> pd.Series:
    future = pd.date_range(origin + pd.Timedelta(hours=1), periods=72, freq="h")
    x = calendar_features(future)
    values = result.get_forecast(steps=72, exog=x).predicted_mean
    return pd.Series(values.to_numpy(), index=future, name="predicted_smp")
