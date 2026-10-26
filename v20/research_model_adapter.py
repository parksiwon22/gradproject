"""Non-operating V20 adapter for the calibrated SMP research candidate.

The caller must pass the immutable/as-of price snapshot selected at issue time.
This module is intentionally not wired into the official publication gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'docs' / 'smp_improvement' / 'final_candidate.json'


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def predict(prices: pd.DataFrame, target_date: str, horizon_days: int,
            config: dict | None = None) -> pd.DataFrame:
    """Build a D+1/D+2/D+3 24-hour forecast from timestamp-cutoff-safe lags.

    Required input columns are ``timestamp`` (timezone-aware Asia/Seoul or
    convertible) and ``smp``. For each target hour, the newest permitted lag is
    the same hour at ``target - horizon_days``. All anchor lags precede it.
    """
    if horizon_days not in (1, 2, 3):
        raise ValueError('horizon_days must be 1, 2, or 3')
    cfg = config or load_config()
    if not {'timestamp', 'smp'}.issubset(prices.columns):
        raise ValueError('prices must contain timestamp and smp')
    frame = prices[['timestamp', 'smp']].copy()
    frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True, format='mixed').dt.tz_convert('Asia/Seoul')
    target = pd.date_range(pd.Timestamp(target_date, tz='Asia/Seoul'), periods=24, freq='h')
    origin = target[0].normalize() - pd.Timedelta(days=horizon_days) + pd.Timedelta(hours=23)
    receipt_verified = 'first_seen' in prices.columns
    if receipt_verified:
        receipts = pd.to_datetime(prices['first_seen'], utc=True, format='mixed', errors='coerce')
        frame = frame.loc[receipts <= origin.tz_convert('UTC')].copy()
    frame = frame.loc[frame.timestamp <= origin].copy()
    if frame.timestamp.duplicated().any():
        raise ValueError('duplicate SMP timestamps')
    series = frame.set_index('timestamp').smp.sort_index()
    calibration_cfg = cfg['calibration']
    end_day = origin.normalize() - pd.Timedelta(days=1)
    start_day = end_day - pd.Timedelta(days=int(calibration_cfg['window_days']) - 1)
    anchors = calibration_cfg['anchors']
    weights = [float(x) for x in calibration_cfg['weights']]

    def allocate(hourly_prices):
        need = 60 * (80 - 30) / 100 / 0.9
        charge = np.zeros(24, dtype=float)
        for index in np.argsort(hourly_prices, kind='stable'):
            amount = min(7.0, need)
            charge[index] = amount
            need -= amount
            if need <= 1e-10:
                break
        return charge

    calibration = []
    for day in pd.date_range(start_day, end_day, freq='D'):
        times = pd.date_range(day, periods=24, freq='h')
        actual = series.reindex(times).to_numpy(dtype=float)
        persistence = series.reindex(times - pd.Timedelta(days=horizon_days)).to_numpy(dtype=float)
        anchor_values = {}
        for name, lags in anchors.items():
            values = np.vstack([series.reindex(times - pd.Timedelta(days=int(lag))).to_numpy(dtype=float)
                                for lag in lags]).T
            anchor_values[name] = np.median(values, axis=1) if name.startswith('median') else np.mean(values, axis=1)
        if (np.isfinite(actual).all() and np.isfinite(persistence).all() and
                all(np.isfinite(v).all() for v in anchor_values.values())):
            calibration.append((actual, persistence, anchor_values))

    baseline_mae = float(np.mean([np.mean(np.abs(a - p)) for a, p, _ in calibration])) if calibration else np.inf
    baseline_regret = 0.0
    if calibration:
        baseline_regret = float(np.mean([
            np.dot(a, allocate(p) - allocate(a)) for a, p, _ in calibration]))
    candidates = []
    for anchor_name in anchors:
        for weight in weights:
            mae, rmse, regrets = [], [], []
            for actual, persistence, anchor_values in calibration:
                forecast = (1 - weight) * persistence + weight * anchor_values[anchor_name]
                error = forecast - actual
                mae.append(float(np.mean(np.abs(error))))
                rmse.append(float(np.sqrt(np.mean(error ** 2))))
                regrets.append(float(np.dot(actual, allocate(forecast) - allocate(actual))))
            if mae and np.mean(mae) <= baseline_mae + 1e-10 and np.mean(regrets) <= baseline_regret + 1e-10:
                candidates.append((float(np.mean(mae)), float(np.mean(regrets)),
                                   float(np.sqrt(np.mean(np.square(rmse)))), anchor_name, weight))
    if candidates:
        _, _, _, anchor, weight = min(candidates)
    else:
        anchor, weight = 'persistence', 0.0

    lag_times = target - pd.Timedelta(days=horizon_days)
    persistence = series.reindex(lag_times).to_numpy(dtype=float)
    if anchor == 'persistence':
        seasonal = persistence.copy()
    else:
        anchor_lags = anchors[anchor]
        lag_values = np.vstack([series.reindex(target - pd.Timedelta(days=int(lag))).to_numpy(dtype=float)
                                for lag in anchor_lags]).T
        if anchor.startswith('median'):
            seasonal = np.median(lag_values, axis=1)
        elif anchor == 'mean4':
            seasonal = np.mean(lag_values, axis=1)
        else:
            raise ValueError(f'unsupported seasonal anchor: {anchor}')
    sources = [lag_times] + ([target - pd.Timedelta(days=int(lag)) for lag in anchor_lags]
                              if anchor != 'persistence' else [])
    if max(pd.DatetimeIndex(source).max() for source in sources) > origin:
        raise ValueError('forecast input includes a timestamp after its 23:00 origin')
    if not (np.isfinite(persistence).all() and np.isfinite(seasonal).all()):
        raise ValueError('incomplete issue-time lag history')
    predicted = (1 - weight) * persistence + weight * seasonal
    return pd.DataFrame({'timestamp': target, 'predicted_smp': predicted,
                         'horizon_days': horizon_days, 'origin_timestamp': origin,
                         'model': cfg['algorithm'], 'anchor': anchor, 'anchor_weight': weight,
                         'calibration_days': len(calibration), 'input_receipt_vintage_present': receipt_verified})
