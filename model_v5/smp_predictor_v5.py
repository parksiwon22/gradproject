"""
V5 day-ahead 24-hour SMP forecasting model.

At 23:00 on day D, this model forecasts the 24 SMP values for calendar day
D+1 directly.  It creates one stacked training row per horizon and never feeds
predicted SMP from one horizon into another horizon.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "runs" / "v5" / ".matplotlib-cache"))

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
)

try:
    import torch
except ImportError:
    torch = None  # HGB uses CPU; torch is only for optional seed/device reporting.


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_PATH = WORKSPACE_ROOT / "private_data" / "aligned_hourly_all_features_2023.csv"
RESULTS_DIR = WORKSPACE_ROOT / "runs" / "v5" / "results"
MODELS_DIR = WORKSPACE_ROOT / "runs" / "v5" / "models"

RANDOM_SEED = 42
INPUT_WINDOW = 168
FORECAST_HOURS = 24
VALIDATION_START = pd.Timestamp("2023-09-01")
TEST_START = pd.Timestamp("2023-10-01")
LOCAL_RAMP_THRESHOLD = 15.0
STRONG_RAMP_THRESHOLD = 20.0
HIGH_PRICE_THRESHOLD = 150.0
HIGH_PRICE_170_THRESHOLD = 170.0
ALPHA_GRID = np.round(np.linspace(0.0, 1.0, 101), 2)

RISK_LOW = "\ub0ae\uc74c"
RISK_NORMAL = "\ubcf4\ud1b5"
RISK_CAUTION = "\uc8fc\uc758"
RISK_HIGH = "\ub192\uc74c"

WEATHER_GOOD = "\ucda9\uc804 \ub9d1\uc74c"
WEATHER_CLOUD = "\uad6c\ub984 \uc870\uae08"
WEATHER_ALERT = "\uace0\uac00 \ud750\ub9bc"
WEATHER_LIGHTNING = "\uac00\uaca9 \ubc88\uac1c\uc8fc\uc758"
WEATHER_STORM = "\uac00\uaca9 \ud3ed\uc5fc\uacbd\ubcf4"

README_EXACT_NOTE = "\uac00\uaca9 \ub0a0\uc528\ub294 SMP \uc608\uce21 \uae30\ubc18 \uc704\ud5d8\ub3c4\uc774\uba70 \uc2e4\uc81c \uae30\uc0c1\uc608\ubcf4\uc640 \ub2e4\ub985\ub2c8\ub2e4."

CORE_RAW_FEATURES = [
    "smp",
    "national_demand_mw",
    "seoul_solar_generation_mwh",
    "seoul_wind_generation_mwh",
    "west_renewable_generation_sum",
    "weather_temp_c",
    "weather_rainfall_mm",
    "weather_wind_speed_ms",
    "weather_humidity_percent",
    "weather_sunshine_hr",
    "weather_solar_radiation_mj_m2",
    "reserve_power_mw",
    "reserve_rate_percent",
    "fuel_trade_LNG",
]

HOLIDAY_FEATURES = [
    "is_public_holiday",
    "is_substitute_holiday",
    "is_temporary_holiday",
]

ORIGIN_CALENDAR_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
]

ENGINEERED_FEATURES = [
    "smp_lag_1h",
    "smp_lag_2h",
    "smp_lag_24h",
    "smp_lag_168h",
    "smp_rolling_24h_mean",
    "smp_rolling_24h_std",
    "smp_change_1h",
    "smp_change_24h",
    "demand_lag_1h",
    "demand_lag_24h",
    "renewable_generation",
    "net_load",
    "wind_data_available",
    "solar_radiation_missing",
    "solar_generation_missing",
]

TARGET_CALENDAR_FEATURES = [
    "target_hour",
    "target_hour_sin",
    "target_hour_cos",
    "target_dayofweek_sin",
    "target_dayofweek_cos",
    "target_month_sin",
    "target_month_cos",
    "target_is_weekend",
    "target_is_public_holiday",
    "target_is_substitute_holiday",
    "target_is_temporary_holiday",
]

PROFILE_FEATURES = [
    "horizon",
    "same_hour_previous_day_smp",
    "same_hour_previous_week_smp",
    "profile_daily_weekly_spread",
    "cutoff_to_daily_profile_delta",
]

DEPLOYMENT_STATE: dict[str, Any] = {}


@dataclass(frozen=True)
class ModeConfig:
    mode: str
    hgb_max_iter: int
    learning_rate: float
    max_leaf_nodes: int
    save_model_artifacts: bool


@dataclass
class PreparedHourly:
    df: pd.DataFrame
    origin_features: list[str]
    model_features: list[str]
    raw_columns: list[str]


@dataclass
class SplitData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V5 day-ahead SMP forecasting.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--quick", action="store_true", help="Smoke test with small HGB settings and no model artifact writes.")
    group.add_argument("--full", action="store_true", help="Full training and deployment artifact writes.")
    return parser.parse_args()


def mode_config(args: argparse.Namespace) -> ModeConfig:
    if args.quick:
        return ModeConfig(
            mode="quick",
            hgb_max_iter=50,
            learning_rate=0.08,
            max_leaf_nodes=15,
            save_model_artifacts=False,
        )
    return ModeConfig(
        mode="full",
        hgb_max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=31,
        save_model_artifacts=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_status() -> tuple[bool, str | None]:
    if torch is None or not torch.cuda.is_available():
        return False, None
    return True, torch.cuda.get_device_name(0)


def to_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def preprocess_hourly(raw: pd.DataFrame) -> PreparedHourly:
    raw = raw.copy()
    raw_columns = list(raw.columns)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    if raw["timestamp"].isna().any():
        raise ValueError("Invalid timestamp values found.")
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    if raw["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    df = raw.copy()
    gas_features = [column for column in df.columns if column.startswith("gas_")]
    numeric_candidates = sorted(set(CORE_RAW_FEATURES + HOLIDAY_FEATURES + gas_features))
    to_numeric(df, numeric_candidates)

    if "seoul_wind_generation_mwh" in df.columns:
        wind_observed = df["seoul_wind_generation_mwh"].notna()
        df["wind_data_available"] = wind_observed.astype(int)
        df.loc[~wind_observed, "seoul_wind_generation_mwh"] = 0.0

    if "weather_solar_radiation_mj_m2" in df.columns:
        df["solar_radiation_missing"] = df["weather_solar_radiation_mj_m2"].isna().astype(int)
        df["weather_solar_radiation_mj_m2"] = df["weather_solar_radiation_mj_m2"].fillna(0.0)

    if "seoul_solar_generation_mwh" in df.columns:
        df["solar_generation_missing"] = df["seoul_solar_generation_mwh"].isna().astype(int)
        df["seoul_solar_generation_mwh"] = df["seoul_solar_generation_mwh"].fillna(0.0)

    continuity_cols = [
        column
        for column in CORE_RAW_FEATURES + gas_features
        if column in df.columns
        and column
        not in {
            "smp",
            "seoul_wind_generation_mwh",
            "seoul_solar_generation_mwh",
            "weather_solar_radiation_mj_m2",
        }
    ]
    if continuity_cols:
        df[continuity_cols] = df[continuity_cols].ffill()

    for column in HOLIDAY_FEATURES:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)

    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    df["smp_lag_1h"] = df["smp"].shift(1)
    df["smp_lag_2h"] = df["smp"].shift(2)
    df["smp_lag_24h"] = df["smp"].shift(24)
    df["smp_lag_168h"] = df["smp"].shift(168)
    df["smp_rolling_24h_mean"] = df["smp"].shift(1).rolling(24, min_periods=24).mean()
    df["smp_rolling_24h_std"] = df["smp"].shift(1).rolling(24, min_periods=24).std()
    df["smp_change_1h"] = df["smp"] - df["smp"].shift(1)
    df["smp_change_24h"] = df["smp"] - df["smp"].shift(24)

    if "national_demand_mw" in df.columns:
        df["demand_lag_1h"] = df["national_demand_mw"].shift(1)
        df["demand_lag_24h"] = df["national_demand_mw"].shift(24)

    renewable_cols = [
        column
        for column in ["seoul_solar_generation_mwh", "seoul_wind_generation_mwh", "west_renewable_generation_sum"]
        if column in df.columns
    ]
    if renewable_cols:
        df["renewable_generation"] = df[renewable_cols].sum(axis=1, min_count=1)
    if "national_demand_mw" in df.columns and "renewable_generation" in df.columns:
        df["net_load"] = df["national_demand_mw"] - df["renewable_generation"]

    origin_features: list[str] = []
    origin_features.extend([feature for feature in CORE_RAW_FEATURES if feature in df.columns])
    origin_features.extend([feature for feature in gas_features if feature in df.columns])
    origin_features.extend([feature for feature in HOLIDAY_FEATURES if feature in df.columns])
    origin_features.extend([feature for feature in ORIGIN_CALENDAR_FEATURES if feature in df.columns])
    origin_features.extend([feature for feature in ENGINEERED_FEATURES if feature in df.columns])
    origin_features = list(dict.fromkeys(origin_features))

    model_features = PROFILE_FEATURES + TARGET_CALENDAR_FEATURES + [f"origin_{feature}" for feature in origin_features]
    return PreparedHourly(df=df, origin_features=origin_features, model_features=model_features, raw_columns=raw_columns)


def target_calendar_row(target_ts: pd.Timestamp, future_forecast_df: pd.DataFrame | None = None) -> dict[str, float]:
    holiday_values = {
        "target_is_public_holiday": 0.0,
        "target_is_substitute_holiday": 0.0,
        "target_is_temporary_holiday": 0.0,
    }
    if future_forecast_df is not None and not future_forecast_df.empty:
        future = future_forecast_df.copy()
        if "timestamp" in future.columns:
            future["timestamp"] = pd.to_datetime(future["timestamp"], errors="coerce")
            matches = future[future["timestamp"] == target_ts]
        elif "date" in future.columns:
            future["date"] = pd.to_datetime(future["date"], errors="coerce").dt.normalize()
            matches = future[future["date"] == target_ts.normalize()]
        else:
            matches = pd.DataFrame()
        if not matches.empty:
            row = matches.iloc[0]
            for source, target in [
                ("is_public_holiday", "target_is_public_holiday"),
                ("is_substitute_holiday", "target_is_substitute_holiday"),
                ("is_temporary_holiday", "target_is_temporary_holiday"),
            ]:
                if source in row.index and pd.notna(row[source]):
                    holiday_values[target] = float(row[source])

    hour = target_ts.hour
    dayofweek = target_ts.dayofweek
    month = target_ts.month
    return {
        "target_hour": float(hour),
        "target_hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "target_hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "target_dayofweek_sin": float(np.sin(2 * np.pi * dayofweek / 7)),
        "target_dayofweek_cos": float(np.cos(2 * np.pi * dayofweek / 7)),
        "target_month_sin": float(np.sin(2 * np.pi * month / 12)),
        "target_month_cos": float(np.cos(2 * np.pi * month / 12)),
        "target_is_weekend": float(dayofweek in [5, 6]),
        **holiday_values,
    }


def lookup_holiday_from_actual(prepared: PreparedHourly, target_ts: pd.Timestamp) -> dict[str, float]:
    if target_ts not in prepared.df.index:
        return {}
    row = prepared.df.loc[target_ts]
    return {
        "target_is_public_holiday": float(row["is_public_holiday"]) if "is_public_holiday" in row.index else 0.0,
        "target_is_substitute_holiday": float(row["is_substitute_holiday"]) if "is_substitute_holiday" in row.index else 0.0,
        "target_is_temporary_holiday": float(row["is_temporary_holiday"]) if "is_temporary_holiday" in row.index else 0.0,
    }


def build_one_day_rows(
    prepared: PreparedHourly,
    target_date: pd.Timestamp,
    include_actual: bool,
    future_forecast_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    df = prepared.df
    target_date = pd.Timestamp(target_date).normalize()
    cutoff_ts = target_date - pd.Timedelta(hours=1)
    target_timestamps = [target_date + pd.Timedelta(hours=hour) for hour in range(FORECAST_HOURS)]
    required_history = [cutoff_ts] + [ts - pd.Timedelta(hours=24) for ts in target_timestamps] + [ts - pd.Timedelta(hours=168) for ts in target_timestamps]
    if any(ts not in df.index for ts in required_history):
        return []
    if include_actual and any(ts not in df.index for ts in target_timestamps):
        return []
    cutoff_row = df.loc[cutoff_ts]
    origin_values = cutoff_row[prepared.origin_features]
    if not np.isfinite(origin_values.to_numpy(dtype=np.float64)).all():
        return []

    rows: list[dict[str, Any]] = []
    for horizon, target_ts in enumerate(target_timestamps, start=1):
        same_day = float(df.loc[target_ts - pd.Timedelta(hours=24), "smp"])
        same_week = float(df.loc[target_ts - pd.Timedelta(hours=168), "smp"])
        row: dict[str, Any] = {
            "target_date": target_date.date().isoformat(),
            "origin_timestamp": cutoff_ts,
            "timestamp": target_ts,
            "horizon": horizon,
            "same_hour_previous_day_smp": same_day,
            "same_hour_previous_week_smp": same_week,
            "profile_daily_weekly_spread": same_day - same_week,
            "cutoff_to_daily_profile_delta": float(cutoff_row["smp"]) - same_day,
        }
        calendar = target_calendar_row(target_ts, future_forecast_df)
        if include_actual:
            calendar.update(lookup_holiday_from_actual(prepared, target_ts))
        row.update(calendar)
        for feature in prepared.origin_features:
            row[f"origin_{feature}"] = float(cutoff_row[feature])
        if include_actual:
            actual = float(df.loc[target_ts, "smp"])
            previous_ts = target_ts - pd.Timedelta(hours=1)
            previous_actual = float(df.loc[previous_ts, "smp"])
            row["actual_smp"] = actual
            row["previous_actual_smp"] = previous_actual
            row["actual_delta_smp"] = actual - previous_actual
            row["target_residual_vs_daily"] = actual - same_day
            row["target_ramp_ge_15"] = int((actual - previous_actual) >= LOCAL_RAMP_THRESHOLD)
        rows.append(row)
    return rows


def build_stacked_dataset(prepared: PreparedHourly) -> pd.DataFrame:
    prepared.df = prepared.df.set_index("timestamp", drop=False)
    dates = pd.date_range(
        prepared.df["timestamp"].min().normalize(),
        prepared.df["timestamp"].max().normalize(),
        freq="D",
    )
    rows: list[dict[str, Any]] = []
    for target_date in dates:
        day_rows = build_one_day_rows(prepared, target_date, include_actual=True)
        if len(day_rows) == FORECAST_HOURS:
            rows.extend(day_rows)
    data = pd.DataFrame(rows)
    if data.empty:
        raise ValueError("No day-ahead training rows could be constructed.")
    return data.replace([np.inf, -np.inf], np.nan).dropna(subset=prepared.model_features + ["actual_smp", "target_residual_vs_daily"]).reset_index(drop=True)


def split_stacked(data: pd.DataFrame) -> SplitData:
    target_dates = pd.to_datetime(data["target_date"])
    train = data[target_dates < VALIDATION_START].reset_index(drop=True)
    validation = data[(target_dates >= VALIDATION_START) & (target_dates < TEST_START)].reset_index(drop=True)
    test = data[target_dates >= TEST_START].reset_index(drop=True)
    return SplitData(train=train, validation=validation, test=test)


def make_regressor(config: ModeConfig) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=config.hgb_max_iter,
        learning_rate=config.learning_rate,
        max_leaf_nodes=config.max_leaf_nodes,
        l2_regularization=0.05,
        random_state=RANDOM_SEED,
    )


def make_classifier(config: ModeConfig) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=config.hgb_max_iter,
        learning_rate=config.learning_rate,
        max_leaf_nodes=config.max_leaf_nodes,
        l2_regularization=0.05,
        random_state=RANDOM_SEED,
    )


def balanced_sample_weights(binary_target: np.ndarray) -> np.ndarray:
    positives = int(binary_target.sum())
    negatives = int(len(binary_target) - positives)
    if positives == 0 or negatives == 0:
        return np.ones_like(binary_target, dtype=np.float32)
    positive_weight = len(binary_target) / (2 * positives)
    negative_weight = len(binary_target) / (2 * negatives)
    return np.where(binary_target == 1, positive_weight, negative_weight).astype(np.float32)


def safe_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) < 2 or float(np.nanvar(actual)) <= 0:
        return float("nan")
    return float(r2_score(actual, predicted))


def overall_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MAPE": float(mean_absolute_percentage_error(actual, predicted)),
        "R2": safe_r2(actual, predicted),
        "bias": float(np.mean(predicted - actual)),
    }


def choose_alpha(actual: np.ndarray, left: np.ndarray, right: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    for alpha in ALPHA_GRID:
        pred = alpha * left + (1.0 - alpha) * right
        rows.append({"alpha": float(alpha), "MAE": float(mean_absolute_error(actual, pred)), "R2": safe_r2(actual, pred)})
    df = pd.DataFrame(rows).sort_values(["MAE", "alpha"], ascending=[True, True]).reset_index(drop=True)
    return float(df.iloc[0]["alpha"]), df


def attach_predictions(
    frame: pd.DataFrame,
    regressor: HistGradientBoostingRegressor,
    classifier: HistGradientBoostingClassifier,
    feature_order: list[str],
    profile_alpha_daily: float,
    final_alpha_profile: float,
) -> pd.DataFrame:
    out = frame.copy()
    x = out[feature_order].to_numpy(dtype=np.float32)
    out["daily_profile_baseline"] = out["same_hour_previous_day_smp"].astype(float)
    out["weekly_profile_baseline"] = out["same_hour_previous_week_smp"].astype(float)
    out["profile_blend_baseline"] = profile_alpha_daily * out["daily_profile_baseline"] + (1.0 - profile_alpha_daily) * out["weekly_profile_baseline"]
    out["direct_hgb_day_ahead_residual"] = out["daily_profile_baseline"] + regressor.predict(x)
    out["profile_hgb_blend"] = final_alpha_profile * out["profile_blend_baseline"] + (1.0 - final_alpha_profile) * out["direct_hgb_day_ahead_residual"]
    out["ramp_probability"] = classifier.predict_proba(x)[:, 1]
    return out


MODEL_COLUMNS = {
    "daily_profile_baseline": "daily_profile_baseline",
    "weekly_profile_baseline": "weekly_profile_baseline",
    "validation_selected_daily_weekly_blend": "profile_blend_baseline",
    "direct_hgb_day_ahead_residual": "direct_hgb_day_ahead_residual",
    "validation_selected_profile_hgb_blend": "profile_hgb_blend",
}


def model_comparison_rows(split_name: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    actual = frame["actual_smp"].to_numpy(dtype=float)
    for model_name, column in MODEL_COLUMNS.items():
        metrics = overall_metrics(actual, frame[column].to_numpy(dtype=float))
        rows.append({"split": split_name, "model": model_name, "sample_count": len(frame), **metrics})
    return rows


def per_horizon_rows(split_name: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name, column in MODEL_COLUMNS.items():
        for horizon in range(1, FORECAST_HOURS + 1):
            subset = frame[frame["horizon"] == horizon]
            actual = subset["actual_smp"].to_numpy(dtype=float)
            pred = subset[column].to_numpy(dtype=float)
            metrics = overall_metrics(actual, pred)
            rows.append(
                {
                    "split": split_name,
                    "model": model_name,
                    "horizon": horizon,
                    "sample_count": len(subset),
                    "MAE": metrics["MAE"],
                    "MAPE": metrics["MAPE"],
                    "bias": metrics["bias"],
                }
            )
    return rows


def risk_metric_rows(split_name: str, frame: pd.DataFrame, ramp_alert_threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    masks = {
        "actual_smp_ge_150": frame["actual_smp"].to_numpy(dtype=float) >= HIGH_PRICE_THRESHOLD,
        "actual_smp_ge_170": frame["actual_smp"].to_numpy(dtype=float) >= HIGH_PRICE_170_THRESHOLD,
        "local_ramp_ge_15": frame["actual_delta_smp"].to_numpy(dtype=float) >= LOCAL_RAMP_THRESHOLD,
        "strong_ramp_ge_20": frame["actual_delta_smp"].to_numpy(dtype=float) >= STRONG_RAMP_THRESHOLD,
    }
    actual = frame["actual_smp"].to_numpy(dtype=float)
    for model_name, column in MODEL_COLUMNS.items():
        pred = frame[column].to_numpy(dtype=float)
        row: dict[str, Any] = {"split": split_name, "model": model_name}
        for label, mask in masks.items():
            row[f"{label}_count"] = int(mask.sum())
            row[f"{label}_MAE"] = float(mean_absolute_error(actual[mask], pred[mask])) if mask.any() else float("nan")
        rows.append(row)

    actual_ramp = frame["target_ramp_ge_15"].to_numpy(dtype=int).astype(bool)
    predicted_ramp = frame["ramp_probability"].to_numpy(dtype=float) >= ramp_alert_threshold
    rows.append(
        {
            "split": split_name,
            "model": "day_ahead_ramp_classifier",
            "ramp_probability_threshold": ramp_alert_threshold,
            "ramp_precision": float(precision_score(actual_ramp, predicted_ramp, zero_division=0)),
            "ramp_recall": float(recall_score(actual_ramp, predicted_ramp, zero_division=0)),
            "ramp_F1": float(f1_score(actual_ramp, predicted_ramp, zero_division=0)),
            "predicted_ramp_count": int(predicted_ramp.sum()),
            "actual_ramp_count": int(actual_ramp.sum()),
        }
    )
    return rows


def choose_ramp_threshold(validation: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    actual = validation["target_ramp_ge_15"].to_numpy(dtype=int).astype(bool)
    probabilities = validation["ramp_probability"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for threshold in np.round(np.arange(0.05, 0.9501, 0.05), 2):
        pred = probabilities >= threshold
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(actual, pred, zero_division=0)),
                "recall": float(recall_score(actual, pred, zero_division=0)),
                "F1": float(f1_score(actual, pred, zero_division=0)),
                "predicted_count": int(pred.sum()),
                "actual_count": int(actual.sum()),
            }
        )
    df = pd.DataFrame(rows).sort_values(["F1", "recall", "threshold"], ascending=[False, False, True]).reset_index(drop=True)
    return float(df.iloc[0]["threshold"]), df


def cheapest_window(values: np.ndarray, duration: int) -> tuple[int, float]:
    sums = np.array([values[start : start + duration].sum() for start in range(0, FORECAST_HOURS - duration + 1)])
    best_start = int(np.argmin(sums))
    return best_start, float(sums[best_start])


def overlap_rate(start_a: int, start_b: int, duration: int) -> float:
    set_a = set(range(start_a, start_a + duration))
    set_b = set(range(start_b, start_b + duration))
    return len(set_a & set_b) / duration


def charging_backtest_rows(split_name: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    durations = [2, 3, 4, 5, 6, 8]
    rows: list[dict[str, Any]] = []
    for model_name, column in MODEL_COLUMNS.items():
        for duration in durations:
            regrets: list[float] = []
            exact_matches: list[float] = []
            overlaps: list[float] = []
            for _, day in frame.sort_values("timestamp").groupby("target_date"):
                if len(day) != FORECAST_HOURS:
                    continue
                actual = day["actual_smp"].to_numpy(dtype=float)
                predicted = day[column].to_numpy(dtype=float)
                predicted_start, _ = cheapest_window(predicted, duration)
                actual_start, actual_best_cost = cheapest_window(actual, duration)
                predicted_actual_cost = float(actual[predicted_start : predicted_start + duration].sum())
                regrets.append(predicted_actual_cost - actual_best_cost)
                exact_matches.append(float(predicted_start == actual_start))
                overlaps.append(overlap_rate(predicted_start, actual_start, duration))
            rows.append(
                {
                    "split": split_name,
                    "model": model_name,
                    "duration_hours": duration,
                    "day_count": len(regrets),
                    "average_regret": float(np.mean(regrets)) if regrets else float("nan"),
                    "median_regret": float(np.median(regrets)) if regrets else float("nan"),
                    "exact_window_match_rate": float(np.mean(exact_matches)) if exact_matches else float("nan"),
                    "hour_overlap_rate": float(np.mean(overlaps)) if overlaps else float("nan"),
                }
            )
    return rows


def risk_label(predicted_smp: float, ramp_probability: float) -> str:
    if predicted_smp >= 180 or ramp_probability >= 0.70:
        return RISK_HIGH
    if predicted_smp >= 160 or ramp_probability >= 0.45:
        return RISK_CAUTION
    if predicted_smp >= 135 or ramp_probability >= 0.25:
        return RISK_NORMAL
    return RISK_LOW


def price_weather_label(predicted_smp: float, ramp_probability: float) -> str:
    if predicted_smp >= 200 or ramp_probability >= 0.75:
        return WEATHER_STORM
    if predicted_smp >= 170 or ramp_probability >= 0.55:
        return WEATHER_LIGHTNING
    if predicted_smp >= 150 or ramp_probability >= 0.40:
        return WEATHER_ALERT
    if predicted_smp >= 125 or ramp_probability >= 0.20:
        return WEATHER_CLOUD
    return WEATHER_GOOD


def selected_prediction_column(policy: dict[str, Any]) -> str:
    return str(policy.get("selected_model_column", "profile_hgb_blend"))


def load_deployment_state() -> dict[str, Any]:
    if DEPLOYMENT_STATE:
        return DEPLOYMENT_STATE
    model_path = MODELS_DIR / "day_ahead_hgb_residual_model.joblib"
    classifier_path = MODELS_DIR / "day_ahead_ramp_classifier.joblib"
    metadata_path = MODELS_DIR / "day_ahead_feature_metadata.json"
    policy_path = MODELS_DIR / "day_ahead_policy.json"
    if not all(path.exists() for path in [model_path, classifier_path, metadata_path, policy_path]):
        raise FileNotFoundError("V5 deployment artifacts are not available. Run --full first or call from an active training run.")
    state = {
        "regressor": joblib.load(model_path),
        "classifier": joblib.load(classifier_path),
        "metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
        "policy": json.loads(policy_path.read_text(encoding="utf-8")),
    }
    DEPLOYMENT_STATE.update(state)
    return DEPLOYMENT_STATE


def generate_day_ahead_forecast(
    history_df: pd.DataFrame,
    target_date: str | pd.Timestamp,
    future_forecast_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate exactly 24 day-ahead SMP forecast rows for one target date."""

    state = load_deployment_state()
    prepared = preprocess_hourly(history_df)
    prepared.df = prepared.df.set_index("timestamp", drop=False)
    feature_order = list(state["metadata"]["exact_feature_order"])
    day_rows = build_one_day_rows(prepared, pd.Timestamp(target_date), include_actual=False, future_forecast_df=future_forecast_df)
    if len(day_rows) != FORECAST_HOURS:
        raise ValueError("Exactly 24 forecast rows could not be constructed. Check cutoff history and required previous-day/week profiles.")
    forecast_frame = pd.DataFrame(day_rows)
    x = forecast_frame[feature_order].to_numpy(dtype=np.float32)
    regressor = state["regressor"]
    classifier = state["classifier"]
    policy = state["policy"]
    alpha_profile_daily = float(policy["daily_weekly_profile_blend_alpha"])
    alpha_final = float(policy["profile_hgb_blend_alpha"])
    forecast_frame["baseline_smp"] = forecast_frame["same_hour_previous_day_smp"].astype(float)
    forecast_frame["weekly_baseline_smp"] = forecast_frame["same_hour_previous_week_smp"].astype(float)
    forecast_frame["profile_blend_baseline"] = alpha_profile_daily * forecast_frame["baseline_smp"] + (1.0 - alpha_profile_daily) * forecast_frame["weekly_baseline_smp"]
    forecast_frame["hgb_predicted_smp"] = forecast_frame["baseline_smp"] + regressor.predict(x)
    forecast_frame["profile_hgb_blend"] = alpha_final * forecast_frame["profile_blend_baseline"] + (1.0 - alpha_final) * forecast_frame["hgb_predicted_smp"]
    selected_col = selected_prediction_column(policy)
    forecast_frame["predicted_smp"] = forecast_frame[selected_col]
    forecast_frame["ramp_probability"] = classifier.predict_proba(x)[:, 1]
    forecast_frame["risk_level"] = [
        risk_label(float(price), float(prob))
        for price, prob in zip(forecast_frame["predicted_smp"], forecast_frame["ramp_probability"])
    ]
    forecast_frame["price_weather"] = [
        price_weather_label(float(price), float(prob))
        for price, prob in zip(forecast_frame["predicted_smp"], forecast_frame["ramp_probability"])
    ]
    forecast_frame["model_name"] = policy["selected_model_name"]
    forecast_frame["source"] = "V5_DAY_AHEAD_DIRECT_HGB"
    return forecast_frame[
        [
            "timestamp",
            "horizon",
            "predicted_smp",
            "baseline_smp",
            "hgb_predicted_smp",
            "ramp_probability",
            "risk_level",
            "price_weather",
            "model_name",
            "source",
        ]
    ].reset_index(drop=True)


def save_plots(test: pd.DataFrame, per_horizon_df: pd.DataFrame, charging_df: pd.DataFrame, selected_model: str) -> None:
    first_day = sorted(test["target_date"].unique())[0]
    day = test[test["target_date"] == first_day].sort_values("horizon")
    selected_col = MODEL_COLUMNS[selected_model]
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(day["horizon"], day["actual_smp"], label="Actual", color="black", linewidth=1.6)
    ax1.plot(day["horizon"], day["daily_profile_baseline"], label="Daily baseline", linewidth=1.1)
    ax1.plot(day["horizon"], day["direct_hgb_day_ahead_residual"], label="Direct HGB", linewidth=1.1)
    ax1.plot(day["horizon"], day[selected_col], label="Selected", linewidth=1.4)
    ax1.set_xlabel("Horizon")
    ax1.set_ylabel("SMP")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.bar(day["horizon"], day["ramp_probability"], alpha=0.18, label="Ramp probability", color="crimson")
    ax2.set_ylabel("Ramp probability")
    fig.suptitle(f"V5 example day-ahead forecast: {first_day}")
    ax1.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "v5_example_day_forecast.png", dpi=160)
    plt.close(fig)

    horizon_subset = per_horizon_df[(per_horizon_df["split"] == "final_test") & (per_horizon_df["model"] == selected_model)]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(horizon_subset["horizon"], horizon_subset["MAE"], color="#4c78a8")
    ax.set_title("V5 final-test MAE by horizon")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("MAE")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "v5_per_horizon_mae.png", dpi=160)
    plt.close(fig)

    charge_subset = charging_df[(charging_df["split"] == "final_test") & (charging_df["model"].isin(["daily_profile_baseline", selected_model]))]
    pivot = charge_subset.pivot(index="duration_hours", columns="model", values="average_regret")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("V5 EV charging average regret")
    ax.set_xlabel("Charging duration (hours)")
    ax.set_ylabel("Average regret")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "v5_charging_regret.png", dpi=160)
    plt.close(fig)


def write_readme(
    config: ModeConfig,
    summary: dict[str, Any],
    model_comparison: pd.DataFrame,
    charging_df: pd.DataFrame,
    selected_model: str,
) -> None:
    selected_final = model_comparison[(model_comparison["split"] == "final_test") & (model_comparison["model"] == selected_model)].iloc[0]
    daily_final = model_comparison[(model_comparison["split"] == "final_test") & (model_comparison["model"] == "daily_profile_baseline")].iloc[0]
    selected_charge = charging_df[(charging_df["split"] == "final_test") & (charging_df["model"] == selected_model)]
    avg_regret = float(selected_charge["average_regret"].mean())
    readme = f"""# SMP Predictor V5 Day-Ahead

V5 is the first deployable day-ahead experiment for the EV charging web application. It forecasts all 24 SMP values for the next calendar day from the 23:00 cutoff and does not use recursive predicted SMP inputs.

## Forecast Definition

At 23:00 on day D, V5 forecasts day D+1 from 00:00 through 23:00. Each horizon is predicted directly from the same cutoff history. The baseline is same-hour previous-day SMP, and the residual target is `actual_smp(t+h) - smp(t+h-24)`.

## Deployability

The deployable model uses historical SMP, historical demand/weather/renewable/reserve/gas features, same-hour previous-day and previous-week profiles, and known target-day calendar features. It does not use realized target-day demand, weather, renewable generation, reserve, or SMP setter fields.

{README_EXACT_NOTE}

## Selected Model

- mode: `{config.mode}`
- selected model by validation MAE: `{selected_model}`
- final-test selected MAE: {selected_final["MAE"]:.4f}
- final-test daily-profile MAE: {daily_final["MAE"]:.4f}
- mean final-test charging regret across durations: {avg_regret:.4f}

## Output Schema

`generate_day_ahead_forecast(history_df, target_date, future_forecast_df=None)` returns exactly 24 rows with:

`timestamp, horizon, predicted_smp, baseline_smp, hgb_predicted_smp, ramp_probability, risk_level, price_weather, model_name, source`

## Runtime Summary

- elapsed seconds: {summary["elapsed_seconds"]:.3f}
- daily train/validation/test counts: {summary["daily_counts"]}
- hourly row train/validation/test counts: {summary["row_counts"]}
"""
    (RESULTS_DIR / "README_V5.md").write_text(readme, encoding="utf-8")


def save_model_artifacts(
    regressor: HistGradientBoostingRegressor,
    classifier: HistGradientBoostingClassifier,
    prepared: PreparedHourly,
    policy: dict[str, Any],
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(regressor, MODELS_DIR / "day_ahead_hgb_residual_model.joblib")
    joblib.dump(classifier, MODELS_DIR / "day_ahead_ramp_classifier.joblib")
    metadata = {
        "exact_feature_order": prepared.model_features,
        "cutoff_requirement": "Forecast origin must be 23:00 on day D for target day D+1.",
        "required_history_hours": INPUT_WINDOW,
        "prediction_horizon": "24 direct hourly horizons for next calendar day",
        "target_day_calendar_feature_rules": {
            "target_hour": "0 through 23 for target date",
            "sin_cos": "hour/dayofweek/month encoded with standard cyclical sine/cosine",
            "weekend": "Saturday/Sunday from target date",
            "holiday_flags": "provide via future_forecast_df when forecasting future dates; historical backtests use known calendar flags only",
        },
        "optional_future_forecast_schema": {
            "current_model": "No target-day weather/demand/renewable/reserve forecasts are used.",
            "accepted_calendar_columns": ["timestamp", "date", "is_public_holiday", "is_substitute_holiday", "is_temporary_holiday"],
        },
        "required_input_columns": prepared.raw_columns,
        "preprocessing_requirements": {
            "scaler": "none; HGB models consume numeric tabular features directly",
            "missing_wind": "set missing seoul_wind_generation_mwh to 0 and add wind_data_available",
            "missing_solar_radiation": "set missing weather_solar_radiation_mj_m2 to 0 and add solar_radiation_missing",
            "continuity_features": "forward-fill non-SMP continuity columns before feature engineering",
            "history": "history_df must include through target_date - 1 hour 23:00 and previous-day/week profile values",
        },
        "output_schema": [
            "timestamp",
            "horizon",
            "predicted_smp",
            "baseline_smp",
            "hgb_predicted_smp",
            "ramp_probability",
            "risk_level",
            "price_weather",
            "model_name",
            "source",
        ],
    }
    (MODELS_DIR / "day_ahead_feature_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (MODELS_DIR / "day_ahead_policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = mode_config(args)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(RANDOM_SEED)
    started = time.perf_counter()
    cuda_available, cuda_name = cuda_status()
    device = "cuda" if cuda_available else "cpu"

    raw = pd.read_csv(DATA_PATH)
    prepared = preprocess_hourly(raw)
    stacked = build_stacked_dataset(prepared)
    splits = split_stacked(stacked)

    feature_order = prepared.model_features
    regressor = make_regressor(config)
    regressor.fit(splits.train[feature_order].to_numpy(dtype=np.float32), splits.train["target_residual_vs_daily"].to_numpy(dtype=float))

    classifier = make_classifier(config)
    y_train_ramp = splits.train["target_ramp_ge_15"].to_numpy(dtype=int)
    classifier.fit(
        splits.train[feature_order].to_numpy(dtype=np.float32),
        y_train_ramp,
        sample_weight=balanced_sample_weights(y_train_ramp),
    )

    train_pre = attach_predictions(splits.train, regressor, classifier, feature_order, profile_alpha_daily=1.0, final_alpha_profile=1.0)
    validation_pre = attach_predictions(splits.validation, regressor, classifier, feature_order, profile_alpha_daily=1.0, final_alpha_profile=1.0)
    test_pre = attach_predictions(splits.test, regressor, classifier, feature_order, profile_alpha_daily=1.0, final_alpha_profile=1.0)

    profile_alpha_daily, profile_alpha_df = choose_alpha(
        validation_pre["actual_smp"].to_numpy(dtype=float),
        validation_pre["daily_profile_baseline"].to_numpy(dtype=float),
        validation_pre["weekly_profile_baseline"].to_numpy(dtype=float),
    )
    validation_pre = attach_predictions(splits.validation, regressor, classifier, feature_order, profile_alpha_daily, final_alpha_profile=1.0)
    profile_hgb_alpha, profile_hgb_alpha_df = choose_alpha(
        validation_pre["actual_smp"].to_numpy(dtype=float),
        validation_pre["profile_blend_baseline"].to_numpy(dtype=float),
        validation_pre["direct_hgb_day_ahead_residual"].to_numpy(dtype=float),
    )

    train_pred = attach_predictions(splits.train, regressor, classifier, feature_order, profile_alpha_daily, profile_hgb_alpha)
    validation_pred = attach_predictions(splits.validation, regressor, classifier, feature_order, profile_alpha_daily, profile_hgb_alpha)
    test_pred = attach_predictions(splits.test, regressor, classifier, feature_order, profile_alpha_daily, profile_hgb_alpha)

    model_comparison = pd.DataFrame(
        model_comparison_rows("train", train_pred)
        + model_comparison_rows("validation", validation_pred)
        + model_comparison_rows("final_test", test_pred)
    )
    selected_model = str(model_comparison[model_comparison["split"] == "validation"].sort_values(["MAE", "model"]).iloc[0]["model"])
    ramp_threshold, ramp_threshold_df = choose_ramp_threshold(validation_pred)

    per_horizon = pd.DataFrame(per_horizon_rows("validation", validation_pred) + per_horizon_rows("final_test", test_pred))
    risk_metrics = pd.DataFrame(risk_metric_rows("validation", validation_pred, ramp_threshold) + risk_metric_rows("final_test", test_pred, ramp_threshold))
    charging = pd.DataFrame(charging_backtest_rows("validation", validation_pred) + charging_backtest_rows("final_test", test_pred))

    selected_prediction_values = test_pred[MODEL_COLUMNS[selected_model]].to_numpy(dtype=float)
    test_pred["risk_level"] = [
        risk_label(float(price), float(prob))
        for price, prob in zip(selected_prediction_values, test_pred["ramp_probability"].to_numpy(dtype=float))
    ]
    test_pred["price_weather"] = [
        price_weather_label(float(price), float(prob))
        for price, prob in zip(selected_prediction_values, test_pred["ramp_probability"].to_numpy(dtype=float))
    ]

    prediction_columns = [
        "target_date",
        "origin_timestamp",
        "timestamp",
        "horizon",
        "actual_smp",
        "previous_actual_smp",
        "actual_delta_smp",
        "daily_profile_baseline",
        "weekly_profile_baseline",
        "profile_blend_baseline",
        "direct_hgb_day_ahead_residual",
        "profile_hgb_blend",
        "ramp_probability",
        "risk_level",
        "price_weather",
        "target_ramp_ge_15",
    ]
    test_pred[prediction_columns].to_csv(RESULTS_DIR / "v5_day_ahead_predictions.csv", index=False, encoding="utf-8-sig")
    model_comparison.to_csv(RESULTS_DIR / "v5_model_comparison.csv", index=False, encoding="utf-8-sig")
    per_horizon.to_csv(RESULTS_DIR / "v5_per_horizon_metrics.csv", index=False, encoding="utf-8-sig")
    risk_metrics.to_csv(RESULTS_DIR / "v5_risk_metrics.csv", index=False, encoding="utf-8-sig")
    charging.to_csv(RESULTS_DIR / "v5_charging_window_backtest.csv", index=False, encoding="utf-8-sig")

    policy = {
        "selected_model_name": selected_model,
        "selected_model_column": MODEL_COLUMNS[selected_model],
        "daily_weekly_profile_blend_alpha": profile_alpha_daily,
        "profile_hgb_blend_alpha": profile_hgb_alpha,
        "ramp_probability_threshold": ramp_threshold,
        "validation_model_selection": model_comparison[model_comparison["split"] == "validation"].to_dict(orient="records"),
        "daily_weekly_alpha_selection_top": profile_alpha_df.head(10).to_dict(orient="records"),
        "profile_hgb_alpha_selection_top": profile_hgb_alpha_df.head(10).to_dict(orient="records"),
        "ramp_threshold_selection_top": ramp_threshold_df.head(10).to_dict(orient="records"),
        "risk_rules": {
            "risk_level": "thresholds based on predicted_smp and ramp_probability",
            "price_weather": "transparent labels based on predicted_smp and ramp_probability",
            "note": README_EXACT_NOTE,
        },
    }

    DEPLOYMENT_STATE.clear()
    DEPLOYMENT_STATE.update(
        {
            "regressor": regressor,
            "classifier": classifier,
            "metadata": {"exact_feature_order": feature_order},
            "policy": policy,
        }
    )

    if config.save_model_artifacts:
        save_model_artifacts(regressor, classifier, prepared, policy)

    first_test_date = pd.Timestamp(sorted(test_pred["target_date"].unique())[0])
    history_for_example = raw[pd.to_datetime(raw["timestamp"]) <= (first_test_date - pd.Timedelta(hours=1))].copy()
    example_forecast = generate_day_ahead_forecast(history_for_example, first_test_date)
    if len(example_forecast) != FORECAST_HOURS:
        raise AssertionError("generate_day_ahead_forecast did not return exactly 24 rows.")

    save_plots(test_pred, per_horizon, charging, selected_model)

    elapsed = time.perf_counter() - started
    daily_counts = {
        "train": int(splits.train["target_date"].nunique()),
        "validation": int(splits.validation["target_date"].nunique()),
        "test": int(splits.test["target_date"].nunique()),
    }
    row_counts = {
        "train": int(len(splits.train)),
        "validation": int(len(splits.validation)),
        "test": int(len(splits.test)),
    }
    full_command = f'python tools/run_v5.py --full --data "{DATA_PATH}" --run-dir "{RESULTS_DIR.parent}"'
    summary = {
        "mode": config.mode,
        "elapsed_seconds": elapsed,
        "device": device,
        "torch_cuda_is_available": cuda_available,
        "torch_cuda_get_device_name_0": cuda_name,
        "daily_counts": daily_counts,
        "row_counts": row_counts,
        "feature_count": len(feature_order),
        "feature_order": feature_order,
        "selected_model": selected_model,
        "daily_weekly_profile_blend_alpha": profile_alpha_daily,
        "profile_hgb_blend_alpha": profile_hgb_alpha,
        "ramp_probability_threshold": ramp_threshold,
        "model_artifacts_written": bool(config.save_model_artifacts),
        "output_schema": [
            "timestamp",
            "horizon",
            "predicted_smp",
            "baseline_smp",
            "hgb_predicted_smp",
            "ramp_probability",
            "risk_level",
            "price_weather",
            "model_name",
            "source",
        ],
        "example_forecast_first_rows": example_forecast.head(3).to_dict(orient="records"),
        "full_mode_command": full_command,
    }
    (RESULTS_DIR / "v5_training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_readme(config, summary, model_comparison, charging, selected_model)

    print(f"mode={config.mode}", flush=True)
    print(f"elapsed_seconds={elapsed:.3f}", flush=True)
    print(f"device={device}", flush=True)
    print(f"cuda_available={cuda_available}", flush=True)
    print(f"daily_counts={daily_counts}", flush=True)
    print(f"row_counts={row_counts}", flush=True)
    print(f"selected_model={selected_model}", flush=True)
    print(f"ramp_threshold={ramp_threshold}", flush=True)
    print(f"example_forecast_rows={len(example_forecast)}", flush=True)
    print(f"model_artifacts_written={config.save_model_artifacts}", flush=True)
    print(f"full_mode_command={full_command}", flush=True)


if __name__ == "__main__":
    main()
