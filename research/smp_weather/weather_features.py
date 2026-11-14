"""Vintage-safe weather feature construction for the isolated HGB candidate.

The module accepts only KMA short-forecast records. ASOS observations are
explicitly rejected; callers must provide the original issue and receive times.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__import__("os").environ.get("SMP_SOURCE", ROOT.parent / "gradproject")).resolve()
import sys
sys.path.insert(0, str(SOURCE))
from v17.features import FEATURES as SMP_FEATURES, build as build_smp_features

WEATHER_VARIABLES = ("TMP", "REH", "WSD", "POP")
REGIONS = ("Gangneung", "Seoul", "Daejeon", "Daegu", "Gwangju", "Busan")
WEATHER_FEATURES = tuple(f"weather_{region}_{variable}" for region in REGIONS for variable in WEATHER_VARIABLES)
WEATHER_MODEL_FEATURES = tuple(SMP_FEATURES) + WEATHER_FEATURES
KST = "Asia/Seoul"


def load_vintages(path: Path) -> list[dict]:
    """Read append-only JSONL; missing files return no records, malformed rows fail."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid forecast vintage JSONL at line {line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"forecast vintage row {line_no} is not an object")
            rows.append(row)
    return rows


def index_vintages_by_target(records: list[dict]) -> dict[str, list[dict]]:
    """Index immutable vintage rows by valid calendar date for fold replay."""
    indexed: dict[str, list[dict]] = {}
    for row in records:
        if row.get("record_type") != "forecast":
            continue
        day = row.get("target_date")
        if not day and row.get("target_at_kst"):
            day = str(row["target_at_kst"])[:10]
        if day:
            indexed.setdefault(str(day), []).append(row)
    return indexed


def _time(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("weather vintage timestamps must include a timezone")
    return stamp.tz_convert(KST)


def target_weather_features(records: list[dict] | dict[str, list[dict]], target_hours: pd.DatetimeIndex,
                            forecast_origin_kst: str) -> pd.DataFrame:
    """Select the newest complete eligible forecast vintage for each target.

    A record is eligible only when it is explicitly a forecast, its issue time
    and measured first receipt are no later than the SMP origin, and the valid
    time matches the target hour. The latest eligible value wins per region,
    variable and target hour. Actual observations are never fallback inputs.
    """
    target_hours = pd.DatetimeIndex(target_hours)
    if len(target_hours) != 24 or target_hours.tz is None:
        raise ValueError("weather target must be exactly 24 timezone-aware hourly timestamps")
    target_hours = target_hours.tz_convert(KST)
    if target_hours.hour.tolist() != list(range(24)) or target_hours.normalize().nunique() != 1:
        raise ValueError("weather target hours must be one complete local calendar day")
    origin = _time(forecast_origin_kst)
    by_key: dict[tuple[str, str, pd.Timestamp], tuple[pd.Timestamp, float]] = {}
    target_set = set(target_hours)
    if isinstance(records, dict):
        relevant_records = records.get(target_hours[0].strftime("%Y-%m-%d"), [])
    else:
        relevant_records = records
    for row in relevant_records:
        if row.get("record_type") != "forecast":
            continue
        if "KMA API Hub" not in str(row.get("source", "")):
            continue
        region = row.get("location")
        variable = str(row.get("category", ""))
        if region not in REGIONS or variable not in WEATHER_VARIABLES:
            continue
        try:
            issue = _time(row["issued_at_kst"])
            received = _time(row["received_at_utc"])
            valid = _time(row["target_at_kst"])
            value = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if issue > origin or received > origin or valid not in target_set or not np.isfinite(value):
            continue
        key = (region, variable, valid)
        previous = by_key.get(key)
        if previous is None or issue > previous[0]:
            by_key[key] = (issue, value)

    out = pd.DataFrame({"timestamp": target_hours})
    for region in REGIONS:
        for variable in WEATHER_VARIABLES:
            feature = f"weather_{region}_{variable}"
            out[feature] = [by_key.get((region, variable, ts), (None, np.nan))[1]
                            for ts in target_hours]
    if out.loc[:, WEATHER_FEATURES].isna().any().any():
        missing = int(out.loc[:, WEATHER_FEATURES].isna().sum().sum())
        raise ValueError(f"eligible KMA forecast vintage incomplete: {missing} weather values missing")
    if not np.isfinite(out.loc[:, WEATHER_FEATURES].to_numpy(float)).all():
        raise ValueError("eligible KMA forecast has non-finite values")
    return out


def target_feature_frame(prices: pd.DataFrame, weather_records: list[dict] | dict[str, list[dict]],
                         target_date: str) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Build SMP and weather predictors from the target's D-3 23:00 snapshot."""
    target = pd.Timestamp(target_date, tz=KST)
    origin = target - pd.Timedelta(days=3) + pd.Timedelta(hours=23)
    known = prices.loc[prices.timestamp <= origin].copy().sort_values("timestamp")
    known = known.drop_duplicates("timestamp", keep="last")
    hours = pd.date_range(target, target + pd.Timedelta(hours=23), freq="h")
    price_features = build_smp_features(known, targets=hours, days_ahead=3, include_labels=False)
    if len(price_features) != 24 or price_features.timestamp.tolist() != hours.tolist():
        raise ValueError("SMP-only D+3 predictors are incomplete at this origin")
    weather = target_weather_features(weather_records, hours, origin.isoformat())
    merged = price_features.merge(weather, on="timestamp", how="inner", validate="one_to_one")
    if len(merged) != 24:
        raise ValueError("SMP/weather hourly alignment is incomplete")
    if (merged.origin_timestamp > origin).any() or known.timestamp.max() > origin:
        raise AssertionError("future SMP input detected")
    return merged, origin


def training_examples(prices: pd.DataFrame, weather_records: list[dict] | dict[str, list[dict]],
                      train_before: str) -> tuple[pd.DataFrame, pd.Series]:
    """Reconstruct each historical training row at that row's own D-3 origin.

    Only target days strictly before train_before are returned. Days without a
    complete weather vintage at their own origin are excluded, never imputed.
    """
    cutoff = pd.Timestamp(train_before, tz=KST)
    dates = sorted(prices.timestamp.dt.strftime("%Y-%m-%d").unique())
    X_rows, y_rows, used_days = [], [], []
    for day in dates:
        if pd.Timestamp(day, tz=KST) >= cutoff:
            continue
        actual = prices.loc[prices.timestamp.dt.strftime("%Y-%m-%d") == day].sort_values("timestamp")
        if len(actual) != 24 or actual.timestamp.dt.hour.tolist() != list(range(24)):
            continue
        try:
            frame, _ = target_feature_frame(prices, weather_records, day)
        except ValueError:
            continue
        if not np.isfinite(frame.loc[:, WEATHER_MODEL_FEATURES].to_numpy(float)).all():
            continue
        y = actual.smp.to_numpy(float) - frame.daily.to_numpy(float)
        if len(y) != 24 or not np.isfinite(y).all():
            continue
        X_rows.extend(frame.loc[:, WEATHER_MODEL_FEATURES].to_dict(orient="records"))
        y_rows.extend(y.tolist())
        used_days.append(day)
    if not X_rows:
        return pd.DataFrame(columns=WEATHER_MODEL_FEATURES), pd.Series(dtype=float, name="residual")
    X = pd.DataFrame(X_rows, columns=WEATHER_MODEL_FEATURES)
    X.attrs["target_dates"] = used_days
    return X, pd.Series(y_rows, name="residual")
