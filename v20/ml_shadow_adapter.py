"""Optional, non-operating adapter for horizon-specific HGB shadow bundles.

This is intentionally not called by the V20 publisher or publication gate.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from v11.features import build as build_d1
from v17.features import build as build_dx


DEFAULT_BUNDLE_DIR = Path(__file__).resolve().parents[1] / 'docs' / 'smp_improvement_ml'


def predict(prices: pd.DataFrame, target_date: str, horizon_days: int,
            bundle_path: str | Path | None = None) -> pd.DataFrame:
    """Return a 24-hour shadow forecast using only issue-time-available prices.

    Input needs timestamp (timezone-aware or UTC-convertible) and smp columns.
    If first_seen receipt metadata is present, it is enforced at the nominal
    23:00 KST origin. Model bundles are trusted local joblib artifacts.
    """
    if horizon_days not in (1, 2, 3):
        raise ValueError('horizon_days must be 1, 2, or 3')
    if not {'timestamp', 'smp'}.issubset(prices.columns):
        raise ValueError('prices must contain timestamp and smp columns')
    path = Path(bundle_path) if bundle_path else DEFAULT_BUNDLE_DIR / f'd{horizon_days}_recommended_shadow.joblib'
    bundle = joblib.load(path)
    if int(bundle['horizon_days']) != horizon_days:
        raise ValueError('bundle horizon does not match request')
    frame = prices.copy()
    frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True, format='mixed').dt.tz_convert('Asia/Seoul')
    target = pd.date_range(pd.Timestamp(target_date, tz='Asia/Seoul'), periods=24, freq='h')
    origin = target[0].normalize() - pd.Timedelta(days=horizon_days) + pd.Timedelta(hours=23)
    receipt_verified = 'first_seen' in frame.columns
    if receipt_verified:
        received = pd.to_datetime(frame.first_seen, utc=True, format='mixed', errors='coerce')
        frame = frame.loc[received <= origin.tz_convert('UTC')].copy()
    frame = frame.loc[frame.timestamp <= origin].copy()
    frame = frame[['timestamp', 'smp']].drop_duplicates('timestamp').sort_values('timestamp')
    features = (build_d1(frame, targets=target, days_ahead=1, include_labels=False)
                if horizon_days == 1 else build_dx(frame, targets=target, days_ahead=horizon_days, include_labels=False))
    if len(features) != 24:
        raise ValueError('insufficient issue-time SMP history to build all 24 hours')
    latest_base = target - pd.Timedelta(days=horizon_days)
    if latest_base.max() > origin:
        raise ValueError('forecast feature timestamp exceeds nominal issue origin')
    x = features[bundle['features']].to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError('non-finite input features')
    ml = features.daily.to_numpy(float) + bundle['model'].predict(x)
    weekly = np.vstack([frame.set_index('timestamp').smp.reindex(target - pd.Timedelta(days=7*k)).to_numpy(float)
                        for k in (1,2,3,4)])
    anchor = np.median(weekly, axis=0)
    if not np.isfinite(anchor).all():
        raise ValueError('insufficient weekly seasonal history')
    weight = float(bundle['anchor_weight'])
    prediction = (1-weight)*ml + weight*anchor
    return pd.DataFrame(dict(timestamp=target, predicted_smp=prediction, ml_smp=ml,
        seasonal_anchor_smp=anchor, horizon_days=horizon_days,
        model=f'hgb_{bundle["mode"]}_seasonal_blend_shadow', origin_timestamp=origin,
        anchor_weight=weight, trained_through=bundle['trained_through'],
        input_first_seen_vintage_enforced=receipt_verified,
        operating_model_replacement=False))
