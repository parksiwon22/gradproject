from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import END, PROCESSED, START


# The full selected input contract. Actual-observation fields must be lagged;
# forecast fields must be archived vintages available at the forecast origin.
REQUIRED_MODEL_INPUTS = [
    "demand_actual_lag", "demand_forecast_vintage",
    "generation_lng_lag", "generation_coal_lag", "generation_nuclear_lag",
    "generation_other_fuels_lag", "solar_generation_national_lag",
    "wind_generation_national_lag", "solar_generation_forecast_vintage",
    "reserve_supply_lag", "reserve_operating_lag",
    "lng_generation_fuel_unit_cost_available",
    "monthly_fuel_cost_available",
    "wti_available", "brent_available", "japan_lng_available", "australia_coal_available",
    "weather_ta_lag", "weather_si_lag", "weather_ws_lag",
    "weather_forecast_tmp_vintage", "weather_forecast_wsd_vintage",
    "east_west_icsrpred_vintage", "east_west_temp_vintage", "east_west_ws_vintage",
    "is_holiday",
]

INPUT_PANEL = PROCESSED / "model_inputs_2023_2025.csv"


@dataclass
class Readiness:
    ready: bool
    blockers: list[str]
    details: list[str]


def inspect_model_inputs(path: Path = INPUT_PANEL) -> Readiness:
    blockers: list[str] = []
    details: list[str] = []
    if not path.exists():
        blockers.append(f"Complete point-in-time input panel missing: {path}")
        details.append("No selected exogenous field panel has been assembled for 2023-01-01 through 2025-12-31.")
        return Readiness(False, blockers, details)
    try:
        frame = pd.read_csv(path, parse_dates=["timestamp"])
    except Exception as exc:
        return Readiness(False, [f"Cannot read input panel ({type(exc).__name__})."], details)
    missing_columns = [c for c in ["timestamp", "max_source_available_at", *REQUIRED_MODEL_INPUTS] if c not in frame.columns]
    if missing_columns:
        blockers.append("Required panel columns missing: " + ", ".join(missing_columns))
        return Readiness(False, blockers, details)
    frame["max_source_available_at"] = pd.to_datetime(frame.max_source_available_at, errors="coerce")
    frame = frame.set_index("timestamp").sort_index()
    expected = pd.date_range(START, END, freq="h")
    missing_hours = expected.difference(frame.index)
    if len(missing_hours):
        blockers.append(f"Hourly panel lacks {len(missing_hours)} timestamps; first missing={missing_hours[0]}.")
    frame = frame.reindex(expected)
    for col in REQUIRED_MODEL_INPUTS:
        values = pd.to_numeric(frame[col], errors="coerce")
        missing = int(values.isna().sum())
        if missing:
            first = values.index[values.isna()][0]
            blockers.append(f"{col}: {missing} missing/non-numeric hourly values; first missing={first}.")
    unavailable = frame.max_source_available_at.isna()
    if unavailable.any():
        blockers.append(f"max_source_available_at missing for {int(unavailable.sum())} hours.")
    # At each row, feature source timestamps must not postdate that row's
    # forecast origin. Forecast-vintage feature assembly must materialize this.
    row_time = pd.Series(expected, index=expected)
    future = frame.max_source_available_at > row_time
    if future.fillna(False).any():
        blockers.append(f"Look-ahead leakage: {int(future.sum())} rows contain source data later than timestamp.")
    if not blockers:
        details.append("Field coverage and source-availability timestamps pass the input completeness checks.")
    return Readiness(not blockers, blockers, details)


def require_model_inputs() -> None:
    result = inspect_model_inputs()
    if not result.ready:
        lines = ["SARIMAX training blocked: selected input data are incomplete or fail the as-of check.", *result.blockers,
                 "No model was fitted and no Validation/Final Test metrics were written."]
        raise RuntimeError("\n".join(lines))
