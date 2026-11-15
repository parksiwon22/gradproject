"""Validation-selected LightGBM comparison using the exact XGBoost panel.

The source data and XGBoost project are read-only. This script rebuilds the
supervised panel in memory with the existing XGBoost panel builder, checks it
against the persisted XGBoost panel, and trains on that saved panel.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[3]
XGBOOST_ROOT = Path(os.environ.get("SMP_XGBOOST_PROJECT", PROJECT_ROOT.parent / "SMP_XGBOOST_2026")).resolve()
SARIMAX_ROOT = Path(os.environ.get("SMP_SARIMAX_PROJECT", PROJECT_ROOT.parent / "SMP_SARIMAX_2026")).resolve()
PANEL_PATH = XGBOOST_ROOT / "data/processed/day_ahead_supervised_panel_2023_2025.csv"
XGBOOST_PREDICTIONS_PATH = XGBOOST_ROOT / "outputs/predictions/xgboost_day_ahead_predictions.csv"
XGBOOST_SUMMARY_PATH = XGBOOST_ROOT / "outputs/metrics/experiment_summary.json"

OUTPUT_ROOT = PROJECT_ROOT / "LightGBM/outputs"
PREDICTION_DIR = OUTPUT_ROOT / "predictions"
METRIC_DIR = OUTPUT_ROOT / "metrics"
FIGURE_DIR = OUTPUT_ROOT / "figures"
MODEL_DIR = PROJECT_ROOT / "LightGBM/models"
REPORT_DIR = PROJECT_ROOT / "LightGBM/reports"
COMPARISON_DIR = PROJECT_ROOT / "Comparison"

sys.path.insert(0, str(XGBOOST_ROOT))

from src.models.xgboost_day_ahead import FEATURES as XGBOOST_FEATURES
from src.models.xgboost_day_ahead import build_supervised_panel

_loader_path = PROJECT_ROOT / "Shared_Data/loaders/common_data.py"
_loader_spec = importlib.util.spec_from_file_location("shared_common_data", _loader_path)
if _loader_spec is None or _loader_spec.loader is None:
    raise ImportError(f"Cannot load shared data loader: {_loader_path}")
_shared_common_data = importlib.util.module_from_spec(_loader_spec)
sys.modules[_loader_spec.name] = _shared_common_data
_loader_spec.loader.exec_module(_shared_common_data)

FEATURES = list(XGBOOST_FEATURES)
SPLITS = ("train", "validation", "final_test")
SEED = 42


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    values = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy(dtype="uint64")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "bias_pred_minus_actual": float(np.mean(predicted - actual)),
    }


def _matrix(frame: pd.DataFrame) -> np.ndarray:
    # Match XGBoost's float32 matrix conversion. Both boosters handle NaN
    # natively; no imputation, filtering, or row dropping is applied here.
    return frame[FEATURES].to_numpy(dtype=np.float32)


def _candidate_configs() -> list[dict[str, int | float]]:
    # Compact, predeclared capacity/regularization sweep. Validation MAE on
    # reconstructed SMP (baseline + residual) selects configuration and trees.
    return [
        {
            "max_depth": 4, "num_leaves": 15, "learning_rate": 0.030,
            "min_child_samples": 40, "feature_fraction": 0.90,
            "bagging_fraction": 0.90, "bagging_freq": 1,
            "reg_lambda": 8.0, "reg_alpha": 0.05,
        },
        {
            "max_depth": 6, "num_leaves": 31, "learning_rate": 0.025,
            "min_child_samples": 60, "feature_fraction": 0.90,
            "bagging_fraction": 0.85, "bagging_freq": 1,
            "reg_lambda": 12.0, "reg_alpha": 0.10,
        },
        {
            "max_depth": 8, "num_leaves": 31, "learning_rate": 0.020,
            "min_child_samples": 100, "feature_fraction": 0.85,
            "bagging_fraction": 0.80, "bagging_freq": 1,
            "reg_lambda": 16.0, "reg_alpha": 0.20,
        },
    ]


def _new_model(config: dict[str, int | float], n_estimators: int = 1800) -> lgb.LGBMRegressor:
    params = dict(config)
    params.update(
        objective="regression",
        n_estimators=n_estimators,
        n_jobs=4,
        random_state=SEED,
        bagging_seed=SEED,
        feature_fraction_seed=SEED,
        data_random_seed=SEED,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    return lgb.LGBMRegressor(**params)


def _load_and_verify_panel() -> tuple[pd.DataFrame, dict]:
    required_paths = (PANEL_PATH, XGBOOST_PREDICTIONS_PATH, XGBOOST_SUMMARY_PATH)
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError("Required XGBoost artifact(s) are missing: " + ", ".join(missing_paths))

    inputs = _shared_common_data.load_shared_hourly_inputs(
        target_path=SARIMAX_ROOT / "data/processed/hourly_smp_2023_2025.csv",
        calendar_path=XGBOOST_ROOT / "data/processed/calendar_features_hourly_2023_2025.csv",
        asos_path=XGBOOST_ROOT / "data/processed/kma_asos_108_hourly_2023_2025.csv",
    )
    rebuilt = build_supervised_panel(inputs)
    source = pd.read_csv(
        PANEL_PATH,
        float_precision="round_trip",
        parse_dates=[
            "target_timestamp", "forecast_origin", "target_date",
            "weather_source_timestamp", "max_source_timestamp",
        ],
    )
    if rebuilt.columns.tolist() != source.columns.tolist():
        raise AssertionError("Rebuilt panel columns/order differ from the saved XGBoost panel.")
    if len(rebuilt) != len(source):
        raise AssertionError(f"Rebuilt panel rows {len(rebuilt)} != saved rows {len(source)}.")

    timestamp_cols = {
        "target_timestamp", "forecast_origin", "target_date",
        "weather_source_timestamp", "max_source_timestamp",
    }
    mismatch_columns = []
    for column in rebuilt.columns:
        if column in timestamp_cols:
            left = pd.to_datetime(rebuilt[column]).astype("int64").to_numpy()
            right = pd.to_datetime(source[column]).astype("int64").to_numpy()
            equal = np.array_equal(left, right)
        elif column == "cutoff_check_pass":
            equal = np.array_equal(rebuilt[column].astype(bool).to_numpy(), source[column].astype(bool).to_numpy())
        elif pd.api.types.is_numeric_dtype(rebuilt[column]):
            left = pd.to_numeric(rebuilt[column], errors="coerce").to_numpy(dtype=float)
            right = pd.to_numeric(source[column], errors="coerce").to_numpy(dtype=float)
            equal = np.array_equal(left, right, equal_nan=True)
        else:
            equal = np.array_equal(rebuilt[column].astype(str).to_numpy(), source[column].astype(str).to_numpy())
        if not equal:
            mismatch_columns.append(column)
    if mismatch_columns:
        raise AssertionError("Rebuilt panel differs from saved XGBoost panel in: " + ", ".join(mismatch_columns))

    xgb_predictions = pd.read_csv(
        XGBOOST_PREDICTIONS_PATH,
        float_precision="round_trip",
        parse_dates=["forecast_origin", "target_timestamp", "target_date"],
    )
    eval_rows = source[source.split.isin(["validation", "final_test"])].copy()
    joined = eval_rows.merge(
        xgb_predictions[["target_timestamp", "split", "actual_smp", "daily_baseline_smp"]],
        on=["target_timestamp", "split"], how="outer", suffixes=("_panel", "_xgb"), indicator=True,
    )
    timestamps_equal = bool((joined["_merge"] == "both").all() and len(joined) == len(xgb_predictions))
    target_equal = bool(
        timestamps_equal
        and np.array_equal(joined.actual_smp_panel.to_numpy(dtype=float), joined.actual_smp_xgb.to_numpy(dtype=float))
        and np.array_equal(joined.smp_same_hour_prev_day.to_numpy(dtype=float), joined.daily_baseline_smp.to_numpy(dtype=float))
    )
    if not target_equal:
        raise AssertionError("XGBoost evaluation timestamps, target SMP, or daily baseline differ from the panel.")

    feature_names_equal = FEATURES == XGBOOST_FEATURES
    if not feature_names_equal or source[FEATURES].columns.tolist() != FEATURES:
        raise AssertionError("LightGBM/XGBoost feature names or order differ.")
    if not bool(source.cutoff_check_pass.astype(bool).all()):
        raise AssertionError("At least one XGBoost panel row fails the forecast-origin cutoff check.")

    counts = source.groupby("split", sort=False).size().reindex(SPLITS).astype(int).to_dict()
    date_ranges = {}
    for split in SPLITS:
        part = source[source.split == split]
        date_ranges[split] = {
            "n": int(len(part)),
            "first_target_timestamp": str(part.target_timestamp.min()),
            "last_target_timestamp": str(part.target_timestamp.max()),
        }
    comparison_columns = ["target_timestamp", "forecast_origin", "actual_smp", "target_residual_vs_daily", *FEATURES]
    missing_by_feature = {
        split: source.loc[source.split == split, FEATURES].isna().sum().astype(int).to_dict()
        for split in SPLITS
    }
    report = {
        "status": "passed",
        "source_project": str(XGBOOST_ROOT),
        "source_panel_path": str(PANEL_PATH),
        "source_panel_bytes": PANEL_PATH.stat().st_size,
        "source_panel_sha256": _sha256(PANEL_PATH),
        "logical_feature_target_timestamp_sha256": _logical_hash(source, comparison_columns),
        "xgboost_predictions_sha256": _sha256(XGBOOST_PREDICTIONS_PATH),
        "feature_names_and_order_identical": feature_names_equal,
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "rebuilt_panel_matches_saved_panel_all_columns_exactly": True,
        "xgboost_eval_timestamp_set_matches_panel": timestamps_equal,
        "xgboost_eval_actual_smp_and_daily_baseline_match_panel": target_equal,
        "split_rows": counts,
        "split_ranges": date_ranges,
        "missing_feature_values_by_split": missing_by_feature,
        "missing_value_policy": "No imputation; preserve NaNs; XGBoost and LightGBM both receive the same float32 matrix values.",
        "model_target": "target_residual_vs_daily = actual_smp - smp_same_hour_prev_day; predicted residual is added to the same daily baseline.",
        "cutoff_check_pass_rows": int(source.cutoff_check_pass.astype(bool).sum()),
        "cutoff_check_fail_rows": int((~source.cutoff_check_pass.astype(bool)).sum()),
        "max_source_after_forecast_origin_rows": int((source.max_source_timestamp > source.forecast_origin).sum()),
    }
    return source, report


def _make_predictions(panel: pd.DataFrame, model: lgb.LGBMRegressor, split: str) -> pd.DataFrame:
    part = panel[panel.split == split].copy()
    residual = model.predict(_matrix(part))
    part["lightgbm_smp"] = part.smp_same_hour_prev_day.to_numpy(dtype=float) + residual
    part["daily_baseline_smp"] = part.smp_same_hour_prev_day
    part["absolute_error_lightgbm"] = (part.actual_smp - part.lightgbm_smp).abs()
    part["absolute_error_baseline"] = (part.actual_smp - part.daily_baseline_smp).abs()
    return part[[
        "forecast_origin", "target_timestamp", "target_date", "horizon", "split",
        "actual_smp", "daily_baseline_smp", "lightgbm_smp",
        "absolute_error_lightgbm", "absolute_error_baseline",
    ]]


def _write_figures(paired_test: pd.DataFrame, hourly: pd.DataFrame, monthly: pd.DataFrame, metrics: pd.DataFrame, importance: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    test = paired_test.copy()
    test["target_timestamp"] = pd.to_datetime(test.target_timestamp)

    fig, ax = plt.subplots(figsize=(17, 6))
    ax.plot(test.target_timestamp, test.actual_smp, color="#222222", lw=0.9, label="Actual SMP")
    ax.plot(test.target_timestamp, test.lightgbm_smp, color="#16865b", lw=0.85, label="LightGBM")
    ax.plot(test.target_timestamp, test.daily_baseline_smp, color="#e07a1f", lw=0.65, alpha=0.75, label="Previous-day baseline")
    ax.set_title("Next-day SMP forecast — LightGBM Final Test (2025-07 to 2025-12)")
    ax.set_xlabel("Target timestamp (Asia/Seoul)")
    ax.set_ylabel("SMP (KRW/kWh)")
    ax.grid(alpha=0.2)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "lightgbm_final_test_actual_vs_predicted.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(17, 6))
    ax.plot(test.target_timestamp, test.actual_smp, color="#222222", lw=0.9, label="Actual SMP")
    ax.plot(test.target_timestamp, test.xgboost_smp, color="#1769aa", lw=0.8, label="XGBoost")
    ax.plot(test.target_timestamp, test.lightgbm_smp, color="#16865b", lw=0.8, label="LightGBM")
    ax.set_title("Same-period model comparison — Final Test (2025-07 to 2025-12)")
    ax.set_xlabel("Target timestamp (Asia/Seoul)")
    ax.set_ylabel("SMP (KRW/kWh)")
    ax.grid(alpha=0.2)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(COMPARISON_DIR / "figures/xgboost_lightgbm_final_test_comparison.png", dpi=180)
    plt.close(fig)

    h = hourly[hourly.split == "final_test"].sort_values("target_hour")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(h.target_hour, h.xgboost_mae, marker="o", label="XGBoost MAE")
    ax.plot(h.target_hour, h.lightgbm_mae, marker="o", label="LightGBM MAE")
    ax.set_xticks(range(24))
    ax.set_xlabel("Target hour (KST; 0–23)")
    ax.set_ylabel("MAE (KRW/kWh)")
    ax.set_title("Final Test MAE by hour")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(COMPARISON_DIR / "figures/hourly_mae_comparison.png", dpi=180)
    plt.close(fig)

    m = monthly[monthly.split == "final_test"].sort_values("month")
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(m))
    width = 0.36
    ax.bar(x - width / 2, m.xgboost_mae, width, label="XGBoost")
    ax.bar(x + width / 2, m.lightgbm_mae, width, label="LightGBM")
    ax.set_xticks(x, m.month_label)
    ax.set_xlabel("Test month (2025)")
    ax.set_ylabel("MAE (KRW/kWh)")
    ax.set_title("Final Test MAE by month")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(COMPARISON_DIR / "figures/monthly_mae_comparison.png", dpi=180)
    plt.close(fig)

    final_metrics = metrics[(metrics.split == "final_test") & metrics.model.isin(["xgboost", "lightgbm"])].set_index("model")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, metric, title, unit in zip(
        axes, ("mae", "rmse", "r2"), ("MAE", "RMSE", "R²"), ("KRW/kWh", "KRW/kWh", "score"),
    ):
        ax.bar(["XGBoost", "LightGBM"], [final_metrics.loc["xgboost", metric], final_metrics.loc["lightgbm", metric]], color=["#1769aa", "#16865b"])
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(handles=[
            Patch(facecolor="#1769aa", label="XGBoost"),
            Patch(facecolor="#16865b", label="LightGBM"),
        ])
    fig.suptitle("Final Test model metrics — identical rows and timestamps")
    fig.tight_layout()
    fig.savefig(COMPARISON_DIR / "figures/model_metrics_comparison.png", dpi=180)
    plt.close(fig)

    all_features = importance.sort_values("gain_importance", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.barh(all_features.feature, all_features.gain_importance, color="#16865b", label="LightGBM gain")
    ax.set_xlabel("Gain importance (unitless)")
    ax.set_ylabel("Input feature")
    ax.set_title("LightGBM feature importance — all 27 inputs")
    ax.grid(axis="x", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "lightgbm_feature_importance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(test.actual_smp, test.lightgbm_smp, s=9, alpha=0.32, color="#16865b", edgecolors="none", label="LightGBM predictions")
    lo = float(min(test.actual_smp.min(), test.lightgbm_smp.min()))
    hi = float(max(test.actual_smp.max(), test.lightgbm_smp.max()))
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color="#333333", label="Perfect prediction")
    ax.set_xlabel("Actual SMP (KRW/kWh)")
    ax.set_ylabel("LightGBM predicted SMP (KRW/kWh)")
    ax.set_title("Actual vs predicted SMP — Final Test")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "lightgbm_actual_vs_predicted_scatter.png", dpi=180)
    plt.close(fig)

    residual = test.lightgbm_smp - test.actual_smp
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(residual, bins=50, color="#16865b", alpha=0.85, edgecolor="white", label="Hourly prediction error")
    ax.axvline(0, color="#222222", ls="--", lw=1, label="Zero error")
    ax.set_xlabel("Prediction error (LightGBM − actual, KRW/kWh)")
    ax.set_ylabel("Hourly observations")
    ax.set_title("LightGBM prediction error distribution — Final Test")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "lightgbm_error_distribution.png", dpi=180)
    plt.close(fig)


def run_experiment() -> dict:
    started = time.perf_counter()
    for path in (PREDICTION_DIR, METRIC_DIR, FIGURE_DIR, MODEL_DIR, REPORT_DIR, COMPARISON_DIR / "figures"):
        path.mkdir(parents=True, exist_ok=True)

    panel, identity = _load_and_verify_panel()
    train = panel[panel.split == "train"].copy()
    validation = panel[panel.split == "validation"].copy()
    final_test = panel[panel.split == "final_test"].copy()
    if [len(train), len(validation), len(final_test)] != [17352, 4344, 4416]:
        raise AssertionError("Unexpected chronological split counts; no model was fit.")

    train_x = _matrix(train)
    train_y = train.target_residual_vs_daily.to_numpy(dtype=float)
    val_x = _matrix(validation)
    val_y = validation.target_residual_vs_daily.to_numpy(dtype=float)
    candidate_rows = []
    selected_model = None
    selected_config = None
    selected_validation_prediction = None
    selected_mae = float("inf")

    for candidate_id, config in enumerate(_candidate_configs(), start=1):
        candidate = _new_model(config)
        candidate.fit(
            train_x, train_y,
            eval_X=val_x,
            eval_y=val_y,
            eval_metric="l1",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        validation_prediction = validation.smp_same_hour_prev_day.to_numpy(dtype=float) + candidate.predict(val_x)
        actual = validation.actual_smp.to_numpy(dtype=float)
        scores = _metrics(actual, validation_prediction)
        best_iteration = int(candidate.best_iteration_ or candidate.n_estimators_)
        candidate_rows.append({"candidate": candidate_id, **config, "best_iteration": best_iteration, **scores})
        if scores["mae"] < selected_mae:
            selected_mae = float(scores["mae"])
            selected_model = candidate
            selected_config = config
            selected_validation_prediction = validation_prediction

    if selected_model is None or selected_config is None or selected_validation_prediction is None:
        raise RuntimeError("No LightGBM candidate completed validation selection.")

    best_iteration = int(selected_model.best_iteration_ or selected_model.n_estimators_)
    train_validation = pd.concat([train, validation], ignore_index=True)
    final_model = _new_model(selected_config, n_estimators=best_iteration)
    final_model.fit(_matrix(train_validation), train_validation.target_residual_vs_daily.to_numpy(dtype=float))

    validation_predictions = _make_predictions(panel, selected_model, "validation")
    final_test_predictions = _make_predictions(panel, final_model, "final_test")
    predictions = pd.concat([validation_predictions, final_test_predictions], ignore_index=True)
    predictions.to_csv(PREDICTION_DIR / "lightgbm_day_ahead_predictions.csv", index=False, float_format="%.8f")

    xgb_predictions = pd.read_csv(
        XGBOOST_PREDICTIONS_PATH,
        float_precision="round_trip",
        parse_dates=["forecast_origin", "target_timestamp", "target_date"],
    )
    paired = xgb_predictions.merge(
        predictions,
        on=["forecast_origin", "target_timestamp", "target_date", "split", "actual_smp", "daily_baseline_smp"],
        how="outer", validate="one_to_one", indicator=True,
    )
    if not bool((paired["_merge"] == "both").all()) or len(paired) != len(xgb_predictions) or len(paired) != len(predictions):
        raise AssertionError("The saved XGBoost and LightGBM prediction rows are not identical.")
    paired = paired.drop(columns="_merge")
    paired["abs_error_xgboost"] = (paired.actual_smp - paired.xgboost_smp).abs()
    paired["abs_error_lightgbm"] = (paired.actual_smp - paired.lightgbm_smp).abs()
    paired["abs_error_baseline"] = (paired.actual_smp - paired.daily_baseline_smp).abs()
    paired.to_csv(COMPARISON_DIR / "paired_predictions_2025.csv", index=False, float_format="%.8f")
    paired_test = paired[paired.split == "final_test"].copy()
    if not paired_test.target_timestamp.is_monotonic_increasing:
        paired_test = paired_test.sort_values("target_timestamp")

    metric_rows = []
    hourly_rows = []
    monthly_rows = []
    for split in ("validation", "final_test"):
        part = paired[paired.split == split]
        y = part.actual_smp.to_numpy(dtype=float)
        for model_name, column in (("xgboost", "xgboost_smp"), ("lightgbm", "lightgbm_smp"), ("previous_day_baseline", "daily_baseline_smp")):
            score = _metrics(y, part[column].to_numpy(dtype=float))
            metric_rows.append({"split": split, "model": model_name, **score})
        baseline_mae = float(mean_absolute_error(y, part.daily_baseline_smp.to_numpy(dtype=float)))
        for model_name, column in (("xgboost", "xgboost_smp"), ("lightgbm", "lightgbm_smp")):
            model_mae = float(mean_absolute_error(y, part[column].to_numpy(dtype=float)))
            metric_rows[-(3 if model_name == "xgboost" else 2)]["improvement_vs_previous_day_baseline_pct"] = (baseline_mae - model_mae) / baseline_mae * 100
        for hour, hour_part in part.groupby(pd.to_datetime(part.target_timestamp).dt.hour):
            yh = hour_part.actual_smp.to_numpy(dtype=float)
            hourly_rows.append({
                "split": split, "target_hour": int(hour), "n": len(hour_part),
                "xgboost_mae": mean_absolute_error(yh, hour_part.xgboost_smp),
                "lightgbm_mae": mean_absolute_error(yh, hour_part.lightgbm_smp),
                "baseline_mae": mean_absolute_error(yh, hour_part.daily_baseline_smp),
            })
        dates = pd.to_datetime(part.target_timestamp)
        month_keys = dates.dt.to_period("M")
        for month, month_part in part.groupby(month_keys):
            ym = month_part.actual_smp.to_numpy(dtype=float)
            monthly_rows.append({
                "split": split, "month": str(month), "month_label": str(month), "n": len(month_part),
                "xgboost_mae": mean_absolute_error(ym, month_part.xgboost_smp),
                "lightgbm_mae": mean_absolute_error(ym, month_part.lightgbm_smp),
                "baseline_mae": mean_absolute_error(ym, month_part.daily_baseline_smp),
            })

    metrics = pd.DataFrame(metric_rows)
    hourly = pd.DataFrame(hourly_rows).sort_values(["split", "target_hour"])
    monthly = pd.DataFrame(monthly_rows).sort_values(["split", "month"])
    metrics.to_csv(METRIC_DIR / "model_metrics.csv", index=False, float_format="%.8f")
    hourly.to_csv(METRIC_DIR / "hourly_metrics.csv", index=False, float_format="%.8f")
    monthly.to_csv(METRIC_DIR / "monthly_metrics.csv", index=False, float_format="%.8f")
    pd.DataFrame(candidate_rows).to_csv(METRIC_DIR / "validation_candidate_metrics.csv", index=False, float_format="%.8f")

    booster = final_model.booster_
    gain = booster.feature_importance(importance_type="gain")
    importance = pd.DataFrame({"feature": FEATURES, "gain_importance": gain})
    importance["gain_importance_normalized"] = importance.gain_importance / importance.gain_importance.sum()
    importance.sort_values("gain_importance", ascending=False).to_csv(METRIC_DIR / "feature_importance.csv", index=False, float_format="%.10f")

    summary_source = json.loads(XGBOOST_SUMMARY_PATH.read_text(encoding="utf-8"))
    xgb_elapsed = summary_source.get("elapsed_seconds")
    lgb_version = lgb.__version__
    _write_figures(paired_test, hourly, monthly, metrics, importance)
    final_model.booster_.save_model(str(MODEL_DIR / "lightgbm_day_ahead_24h.txt"))

    raw_paths = {
        "SMP target": (
            SARIMAX_ROOT / "data/processed/hourly_smp_2023_2025.csv",
            "https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpShdChart.do?menuId=040202",
            "Existing normalized hourly land-SMP series inherited from the local source project; source report refers to EPSIS hourly-SMP CSV/XLSX export.",
        ),
        "Calendar panel": (
            XGBOOST_ROOT / "data/processed/calendar_features_hourly_2023_2025.csv",
            "Generated from hourly target timestamps; no external source download.",
            "Deterministic calendar panel already maintained in the XGBoost project.",
        ),
        "Seoul ASOS panel": (
            XGBOOST_ROOT / "data/processed/kma_asos_108_hourly_2023_2025.csv",
            "https://apihub.kma.go.kr/apiList.do?seqApi=2",
            "Existing station 108 hourly ASOS normalized panel, recovered in monthly requests; a KMA API key is required to reproduce collection.",
        ),
        "XGBoost supervised panel": (
            PANEL_PATH,
            "Derived locally from the SMP target, calendar and KMA ASOS panels.",
            "Feature/target panel saved by the existing XGBoost model; not copied into this repository.",
        ),
    }
    manifest = []
    for label, (path, source_url, retrieval_method) in raw_paths.items():
        manifest.append({
            "dataset": label,
            "local_source_path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "official_source_or_provenance": source_url,
            "retrieval_method": retrieval_method,
            "included_in_repository": False,
            "redistribution_review": "Not confirmed; source data kept local.",
        })
    pd.DataFrame(manifest).to_csv(PROJECT_ROOT / "Shared_Data/data_manifest.csv", index=False)

    identity.update({
        "source_files": {
            label: {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for label, (path, _, _) in raw_paths.items()
        },
        "lightgbm_missing_policy_matrix_dtype": "float32",
        "lightgbm_nan_count_by_split": {
            split: int(np.isnan(_matrix(panel[panel.split == split])).sum()) for split in SPLITS
        },
        "xgboost_test_prediction_rows": int((xgb_predictions.split == "final_test").sum()),
        "lightgbm_test_prediction_rows": int((predictions.split == "final_test").sum()),
        "paired_test_timestamps_identical": True,
        "paired_test_target_smp_identical": True,
    })
    (METRIC_DIR / "data_equivalence_report.json").write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")

    best_config = {**selected_config, "n_estimators": best_iteration}
    elapsed_seconds = time.perf_counter() - started
    comparison_rows = []
    for split in ("validation", "final_test"):
        for model_name in ("xgboost", "lightgbm"):
            row = metrics[(metrics.split == split) & (metrics.model == model_name)].iloc[0]
            comparison_rows.append({
                "split": split, "model": model_name, "n": int(row.n), "mae": row.mae,
                "rmse": row.rmse, "r2": row.r2,
                "improvement_vs_previous_day_baseline_pct": row.improvement_vs_previous_day_baseline_pct,
            })
    pd.DataFrame(comparison_rows).to_csv(COMPARISON_DIR / "model_comparison.csv", index=False, float_format="%.8f")

    test_metrics = metrics[metrics.split == "final_test"].set_index("model")
    validation_metrics = metrics[metrics.split == "validation"].set_index("model")
    summary = {
        "target_and_panel": "Exact XGBoost supervised panel, target_residual_vs_daily = actual SMP minus previous-day same-hour SMP.",
        "features": FEATURES,
        "split_rows": identity["split_rows"],
        "validation_selection_metric": "MAE of reconstructed SMP; candidate iteration selected by validation L1 early stopping on residual.",
        "final_test_used_for_selection": False,
        "seed": SEED,
        "candidate_configs": candidate_rows,
        "selected_config": best_config,
        "selected_validation_mae": float(validation_metrics.loc["lightgbm", "mae"]),
        "validation_metrics": {k: float(validation_metrics.loc["lightgbm", k]) for k in ("mae", "rmse", "r2")},
        "final_test_metrics": {k: float(test_metrics.loc["lightgbm", k]) for k in ("mae", "rmse", "r2")},
        "previous_day_baseline_final_test_mae": float(test_metrics.loc["previous_day_baseline", "mae"]),
        "final_test_mae_improvement_vs_baseline_pct": float(test_metrics.loc["lightgbm", "improvement_vs_previous_day_baseline_pct"]),
        "experiment_elapsed_seconds": round(elapsed_seconds, 3),
        "xgboost_reported_experiment_elapsed_seconds": xgb_elapsed,
        "timing_note": "Both values are end-to-end experiment runtimes from each script (data checks, fitting, metrics, and figures), not isolated booster-fit time.",
        "lightgbm_version": lgb_version,
        "xgboost_version": summary_source.get("xgboost_version"),
        "data_equivalence": identity,
    }
    (METRIC_DIR / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(summary, metrics, hourly, monthly, {"xgboost": xgb_elapsed, "lightgbm": round(elapsed_seconds, 3)}, identity)
    return summary


def _write_report(summary: dict, metrics: pd.DataFrame, hourly: pd.DataFrame, monthly: pd.DataFrame, elapsed: dict, identity: dict) -> None:
    val = metrics[metrics.split == "validation"].set_index("model")
    test = metrics[metrics.split == "final_test"].set_index("model")
    xgb_test, lgb_test = test.loc["xgboost"], test.loc["lightgbm"]
    winner = lambda a, b, lower=True: "XGBoost" if (a < b if lower else a > b) else ("LightGBM" if (b < a if lower else b > a) else "동률")
    lines = [
        "# LightGBM 하루전 SMP 예측 및 XGBoost 비교",
        "",
        "## 데이터 동일성 및 예측 기준",
        "",
        f"- 저장된 XGBoost panel을 읽어 학습했고, 원본 공통 로더와 XGBoost panel builder로 다시 생성한 결과가 모든 열에서 정확히 일치했습니다. panel SHA-256: {identity['source_panel_sha256']}.",
        f"- 27개 특성의 이름과 순서, split별 행 수, 평가 시각, 실제 SMP, 전일 baseline이 모두 일치했습니다. Final Test 공통 시각은 {identity['split_rows']['final_test']:,}개입니다.",
        "- Train/Validation/Final Test 행: 17,352 / 4,344 / 4,416. 각각 2023-01-01~2024-12-31, 2025-01-01~2025-06-30, 2025-07-01~2025-12-31.",
        "- 예측 시점은 전일 23시(KST)입니다. 모델 목표는 actual_smp - 전일 같은 시간 SMP 잔차이고, 예측 잔차를 전일 baseline에 더해 SMP를 복원합니다.",
        "- 원래 XGBoost 특성 결측은 대치하지 않았습니다. 동일 float32 matrix를 사용하고 NaN은 두 트리 모델의 기본 결측 분기로 처리했습니다.",
        f"- 예측 cutoff 검사 실패 {identity['cutoff_check_fail_rows']}행; cutoff 뒤 source timestamp {identity['max_source_after_forecast_origin_rows']}행.",
        "",
        "## LightGBM 선택 기준과 설정",
        "",
        "- 사전 정의한 3개 트리 용량/규제 설정을 Train으로 학습하고 Validation의 복원 SMP MAE로 선택했습니다. Early stopping은 Validation 잔차 MAE 기준 100회 patience를 사용했습니다.",
        "- Final Test 값은 파라미터 또는 반복 횟수 선택에 사용하지 않았으며, 최종 선택 반복 횟수로 Train+Validation을 재학습해 Final Test를 평가했습니다.",
        f"- random seed: {summary['seed']}; LightGBM {summary['lightgbm_version']}; 최종 설정: {json.dumps(summary['selected_config'], ensure_ascii=False)}.",
        "",
        "## 성능 비교",
        "",
        "| 항목 | XGBoost | LightGBM | 비교 결과 |",
        "|---|---:|---:|---|",
        f"| Validation MAE | {val.loc['xgboost','mae']:.4f} | {val.loc['lightgbm','mae']:.4f} | {winner(val.loc['xgboost','mae'],val.loc['lightgbm','mae'])} |",
        f"| Final Test MAE | {xgb_test.mae:.4f} | {lgb_test.mae:.4f} | {winner(xgb_test.mae,lgb_test.mae)} |",
        f"| Final Test RMSE | {xgb_test.rmse:.4f} | {lgb_test.rmse:.4f} | {winner(xgb_test.rmse,lgb_test.rmse)} |",
        f"| Final Test R² | {xgb_test.r2:.4f} | {lgb_test.r2:.4f} | {winner(xgb_test.r2,lgb_test.r2,lower=False)} |",
        f"| 실험 총 시간 | {elapsed['xgboost']:.3f}초 | {elapsed['lightgbm']:.3f}초 | 각 실행 스크립트 end-to-end 경과 시간 |",
        f"| Final Test 전일 baseline 대비 MAE 개선율 | {xgb_test.improvement_vs_previous_day_baseline_pct:.2f}% | {lgb_test.improvement_vs_previous_day_baseline_pct:.2f}% | 동일 baseline 기준 |",
        "",
        f"Final Test에서 MAE는 {winner(xgb_test.mae,lgb_test.mae)}가 낮고, RMSE는 {winner(xgb_test.rmse,lgb_test.rmse)}가 낮으며, R²는 {winner(xgb_test.r2,lgb_test.r2,lower=False)}가 높습니다. MAE 차이는 {abs(xgb_test.mae-lgb_test.mae):.4f} KRW/kWh입니다.",
        "이 비교는 2025년 하반기 한 기간의 결과이며, 다른 시장 국면이나 가격 상품에 일반화한다고 해석하지 않습니다.",
        "",
        "## 시간대별·월별 MAE",
        "",
        "Final Test 시간대별 MAE (hour-ending 1–24):",
        "",
        "| 시간대 | XGBoost MAE | LightGBM MAE |",
        "|---:|---:|---:|",
    ]
    for row in hourly[hourly.split == "final_test"].sort_values("target_hour").itertuples(index=False):
        lines.append(f"| {row.target_hour + 1} | {row.xgboost_mae:.4f} | {row.lightgbm_mae:.4f} |")
    lines += ["", "Final Test 월별 MAE:", "", "| 월 | XGBoost MAE | LightGBM MAE |", "|---|---:|---:|"]
    for row in monthly[monthly.split == "final_test"].itertuples(index=False):
        lines.append(f"| {row.month_label} | {row.xgboost_mae:.4f} | {row.lightgbm_mae:.4f} |")
    lines += [
        "",
        "## 데이터 취급 및 재현 조건",
        "",
        "- 원본 및 supervised panel은 SMP_XGBOOST_2026 / SMP_SARIMAX_2026 경로에서 읽었습니다. 본 저장소에는 원본 데이터나 full panel을 복사하지 않았습니다.",
        "- 파일 크기와 SHA-256, 로컬 경로는 Shared_Data/data_manifest.csv 및 LightGBM/outputs/metrics/data_equivalence_report.json에 기록했습니다. 공개 조건이 명확하지 않은 SMP·ASOS 파일은 업로드하지 않았습니다.",
        "- 재실행하려면 SMP_XGBOOST_PROJECT와 SMP_SARIMAX_PROJECT 환경 변수를 해당 로컬 프로젝트 경로로 지정하세요. 키가 필요한 API 호출은 사용하지 않습니다.",
        "",
        "## 결과 파일",
        "",
        "- LightGBM 예측/지표/설정/모델: LightGBM/outputs/, LightGBM/models/.",
        "- 동일 시각 XGBoost/LightGBM 비교 및 비교 그래프: Comparison/.",
        "- 모든 8개 PNG 목록은 LightGBM README와 최상위 README에 있습니다.",
        "",
    ]
    (REPORT_DIR / "lightgbm_xgboost_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
    (COMPARISON_DIR / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    result = run_experiment()
    print(json.dumps({
        "validation_metrics": result["validation_metrics"],
        "final_test_metrics": result["final_test_metrics"],
        "selected_config": result["selected_config"],
        "elapsed_seconds": result["experiment_elapsed_seconds"],
        "data_identity": {
            "source_panel_sha256": result["data_equivalence"]["source_panel_sha256"],
            "split_rows": result["split_rows"],
            "same_timestamps": result["data_equivalence"]["paired_test_timestamps_identical"],
        },
    }, ensure_ascii=False, indent=2))
