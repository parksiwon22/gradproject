from __future__ import annotations

import numpy as np
import pandas as pd


def make_forecast_row(origin: pd.Timestamp, y: pd.Series, prediction: pd.Series, split: str) -> pd.DataFrame:
    targets = pd.DatetimeIndex(prediction.index)
    if len(targets) != 72:
        raise ValueError(f"Expected 72 prediction timestamps; received {len(targets)}")
    horizon = np.arange(1, 73)
    actual = y.reindex(targets).to_numpy()
    # Use only past analogues. This avoids target-minus-24 leakage on leads beyond D+1.
    prev_day_idx = [t - pd.Timedelta(hours=24 * int(np.ceil(h / 24))) for t, h in zip(targets, horizon)]
    prev_week_idx = [t - pd.Timedelta(hours=168) for t in targets]
    out = pd.DataFrame({
        "forecast_origin": origin, "target_timestamp": targets,
        "forecast_date": targets.date.astype(str), "horizon": horizon,
        "forecast_day": [f"D+{int(np.ceil(h / 24))}" for h in horizon],
        "actual_smp": actual, "predicted_smp": prediction.to_numpy(),
        "absolute_error": np.abs(actual - prediction.to_numpy()), "model_name": "SARIMAX",
        "previous_day_smp": y.reindex(prev_day_idx).to_numpy(),
        "previous_week_smp": y.reindex(prev_week_idx).to_numpy(), "split": split,
    })
    out["target_month"] = targets.to_period("M").astype(str)
    return out
