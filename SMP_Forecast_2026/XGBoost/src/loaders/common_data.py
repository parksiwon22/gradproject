"""Model-neutral readers for the shared, normalized SMP source panels.

This module performs source loading and schema/grid checks only. Feature
engineering belongs in each model's own preprocessing code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent


@dataclass(frozen=True)
class SharedHourlyInputs:
    target: pd.DataFrame
    calendar: pd.DataFrame
    asos: pd.DataFrame


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def load_normalized_panel(filename: str, area: str = "processed") -> pd.DataFrame:
    """Load any normalized CSV panel without model-specific transformations.

    Use ``area="monthly"`` for monthly source panels. The filename must be a
    direct child name so a model cannot accidentally load arbitrary paths.
    """
    if area not in {"processed", "monthly"}:
        raise ValueError("area must be 'processed' or 'monthly'")
    if Path(filename).name != filename or not filename.lower().endswith(".csv"):
        raise ValueError("filename must be a CSV basename")
    return _read_csv(PROJECT_ROOT / "data" / area / filename)


def load_normalized_monthly_panel(filename: str) -> pd.DataFrame:
    """Convenience wrapper for model-neutral monthly input panels."""
    return load_normalized_panel(filename, area="monthly")


def load_hourly_target(path: Path | None = None) -> pd.DataFrame:
    """Load the existing land-SMP label series without imputing values."""
    source = path or WORKSPACE_ROOT / "SMP_SARIMAX_2026/data/processed/hourly_smp_2023_2025.csv"
    frame = _read_csv(Path(source))
    required = {"timestamp", "smp"}
    if not required.issubset(frame.columns):
        raise ValueError(f"SMP target is missing columns: {sorted(required - set(frame.columns))}")
    frame = frame[[c for c in ("timestamp", "smp", "is_observed") if c in frame.columns]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["smp"] = pd.to_numeric(frame["smp"], errors="coerce")
    if frame["timestamp"].isna().any() or frame["smp"].isna().any():
        raise ValueError("SMP target contains invalid timestamps or missing prices.")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if frame["timestamp"].duplicated().any():
        raise ValueError("SMP target contains duplicate timestamps.")
    expected = pd.date_range(frame.timestamp.min(), frame.timestamp.max(), freq="h")
    if not frame.timestamp.equals(pd.Series(expected, name="timestamp")):
        raise ValueError("SMP target is not a complete hourly grid.")
    return frame


def load_hourly_calendar(path: Path | None = None) -> pd.DataFrame:
    """Load the normalized target-hour calendar panel as a timestamp index."""
    source = path or PROJECT_ROOT / "data/processed/calendar_features_hourly_2023_2025.csv"
    frame = _read_csv(Path(source))
    required = {"target_date", "hour_ending", "is_weekend"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Calendar panel is missing columns: {sorted(required - set(frame.columns))}")
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    frame["hour_ending"] = pd.to_numeric(frame["hour_ending"], errors="coerce")
    if frame["target_date"].isna().any() or not frame.hour_ending.between(1, 24).all():
        raise ValueError("Calendar panel has invalid dates or hour-ending values.")
    frame["timestamp"] = frame.target_date + pd.to_timedelta(frame.hour_ending - 1, unit="h")
    frame = frame.sort_values("timestamp").set_index("timestamp", drop=False)
    if frame.index.duplicated().any():
        raise ValueError("Calendar panel contains duplicate timestamps.")
    return frame


def load_asos_hourly(path: Path | None = None) -> pd.DataFrame:
    """Load normalized KMA ASOS observations; gaps remain missing."""
    source = path or PROJECT_ROOT / "data/processed/kma_asos_108_hourly_2023_2025.csv"
    frame = _read_csv(Path(source))
    required = {"timestamp_kst", "station_id", "temperature_c", "wind_speed_ms"}
    if not required.issubset(frame.columns):
        raise ValueError(f"ASOS panel is missing columns: {sorted(required - set(frame.columns))}")
    frame["timestamp_kst"] = pd.to_datetime(frame["timestamp_kst"], errors="coerce")
    if frame.timestamp_kst.isna().any():
        raise ValueError("ASOS panel contains invalid timestamps.")
    for col in ("temperature_c", "wind_speed_ms", "solar_radiation_mj_m2"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.sort_values("timestamp_kst").set_index("timestamp_kst", drop=False)
    if frame.index.duplicated().any():
        raise ValueError("ASOS panel contains duplicate timestamps.")
    return frame


def load_shared_hourly_inputs(
    target_path: Path | None = None,
    calendar_path: Path | None = None,
    asos_path: Path | None = None,
) -> SharedHourlyInputs:
    """Read shared sources for reuse by XGBoost, LightGBM, SARIMA, and LSTM."""
    return SharedHourlyInputs(
        target=load_hourly_target(target_path),
        calendar=load_hourly_calendar(calendar_path),
        asos=load_asos_hourly(asos_path),
    )
