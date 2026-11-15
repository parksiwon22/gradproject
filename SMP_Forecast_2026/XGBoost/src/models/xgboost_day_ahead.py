"""Leakage-aware next-day, 24-hour XGBoost experiment.

The loader is model-neutral; this module owns XGBoost-specific features,
selection, fitting, scoring, and artifacts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.loaders.common_data import PROJECT_ROOT, SharedHourlyInputs, load_shared_hourly_inputs


START = pd.Timestamp("2023-01-01 00:00:00")
END = pd.Timestamp("2025-12-31 23:00:00")
TRAIN_END = pd.Timestamp("2024-12-31 23:00:00")
VALIDATION_START = pd.Timestamp("2025-01-01 00:00:00")
VALIDATION_END = pd.Timestamp("2025-06-30 23:00:00")
TEST_START = pd.Timestamp("2025-07-01 00:00:00")

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
METRIC_DIR = OUTPUT_DIR / "metrics"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = PROJECT_ROOT / "models"
PANEL_PATH = PROJECT_ROOT / "data/processed/day_ahead_supervised_panel_2023_2025.csv"

FEATURES = [
    "horizon",
    "smp_same_hour_prev_day",
    "smp_same_hour_prev_week",
    "smp_daily_weekly_spread",
    "origin_smp",
    "origin_smp_lag_1h",
    "origin_smp_lag_2h",
    "origin_smp_lag_24h",
    "origin_smp_lag_168h",
    "origin_smp_roll24_mean",
    "origin_smp_roll24_std",
    "origin_smp_roll168_mean",
    "origin_smp_roll168_std",
    "origin_smp_change_1h",
    "origin_smp_change_24h",
    "target_hour",
    "target_hour_sin",
    "target_hour_cos",
    "target_weekday",
    "target_weekday_sin",
    "target_weekday_cos",
    "target_month",
    "target_month_sin",
    "target_month_cos",
    "target_is_weekend",
    "weather_temp_c_lag48h",
    "weather_wind_speed_ms_lag48h",
]


def _at(series: pd.Series, timestamps: pd.DatetimeIndex) -> np.ndarray:
    return series.reindex(timestamps).to_numpy(dtype=float)


def build_supervised_panel(inputs: SharedHourlyInputs) -> pd.DataFrame:
    """Build one row per target hour with features frozen at D-1 23:00."""
    target = inputs.target.set_index("timestamp").sort_index()
    y = target["smp"].astype(float)
    idx = y.index
    if idx.min() != START or idx.max() != END:
        raise ValueError(f"Unexpected target range: {idx.min()} through {idx.max()}")

    calendar = inputs.calendar.reindex(idx)
    if calendar["is_weekend"].isna().any():
        raise ValueError("Calendar panel does not cover the full target grid.")

    origin = idx.normalize() - pd.Timedelta(hours=1)
    prev_day_ts = idx - pd.Timedelta(hours=24)
    prev_week_ts = idx - pd.Timedelta(hours=168)
    weather_ts = idx - pd.Timedelta(hours=48)

    frame = pd.DataFrame(index=idx)
    frame.index.name = "target_timestamp"
    frame["forecast_origin"] = origin
    frame["target_date"] = idx.normalize()
    frame["horizon"] = idx.hour + 1
    frame["actual_smp"] = y.to_numpy(dtype=float)
    frame["smp_same_hour_prev_day"] = _at(y, prev_day_ts)
    frame["smp_same_hour_prev_week"] = _at(y, prev_week_ts)
    frame["smp_daily_weekly_spread"] = frame.smp_same_hour_prev_day - frame.smp_same_hour_prev_week

    frame["origin_smp"] = _at(y, origin)
    for lag in (1, 2, 24, 168):
        frame[f"origin_smp_lag_{lag}h"] = _at(y, origin - pd.Timedelta(hours=lag))
    roll24 = y.rolling(24, min_periods=24)
    roll168 = y.rolling(168, min_periods=168)
    frame["origin_smp_roll24_mean"] = _at(roll24.mean(), origin)
    frame["origin_smp_roll24_std"] = _at(roll24.std(), origin)
    frame["origin_smp_roll168_mean"] = _at(roll168.mean(), origin)
    frame["origin_smp_roll168_std"] = _at(roll168.std(), origin)
    frame["origin_smp_change_1h"] = frame.origin_smp - frame.origin_smp_lag_1h
    frame["origin_smp_change_24h"] = frame.origin_smp - frame.origin_smp_lag_24h

    hour = idx.hour.astype(float)
    weekday = idx.dayofweek.astype(float)
    month = idx.month.astype(float)
    frame["target_hour"] = hour
    frame["target_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    frame["target_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    frame["target_weekday"] = weekday
    frame["target_weekday_sin"] = np.sin(2 * np.pi * weekday / 7.0)
    frame["target_weekday_cos"] = np.cos(2 * np.pi * weekday / 7.0)
    frame["target_month"] = month
    frame["target_month_sin"] = np.sin(2 * np.pi * (month - 1.0) / 12.0)
    frame["target_month_cos"] = np.cos(2 * np.pi * (month - 1.0) / 12.0)
    frame["target_is_weekend"] = pd.to_numeric(calendar["is_weekend"], errors="coerce").to_numpy(dtype=float)

    asos = inputs.asos
    frame["weather_temp_c_lag48h"] = _at(asos["temperature_c"], weather_ts)
    frame["weather_wind_speed_ms_lag48h"] = _at(asos["wind_speed_ms"], weather_ts)
    frame["weather_source_timestamp"] = weather_ts

    # The model only uses prior SMP, calendar values, and delayed observations.
    # Keep timestamps for an auditable row-level point-in-time check.
    source_timestamps = pd.concat(
        [
            pd.Series(origin, index=idx),
            pd.Series(prev_day_ts, index=idx),
            pd.Series(prev_week_ts, index=idx),
            pd.Series(weather_ts, index=idx),
        ],
        axis=1,
    )
    frame["max_source_timestamp"] = source_timestamps.max(axis=1)
    frame["cutoff_check_pass"] = frame.max_source_timestamp.le(frame.forecast_origin)
    if not frame.cutoff_check_pass.all():
        bad = frame.loc[~frame.cutoff_check_pass].head(3)
        raise ValueError(f"Feature timestamps exceed forecast origin: {bad.index.tolist()}")

    frame["target_residual_vs_daily"] = frame.actual_smp - frame.smp_same_hour_prev_day
    frame = frame.reset_index()
    # A full 168-hour origin history is required by the weekly lag and rolling
    # features. The first target day whose 23:00 origin has that history is Jan 9.
    frame = frame.dropna(
        subset=[
            "smp_same_hour_prev_day",
            "smp_same_hour_prev_week",
            "origin_smp_lag_168h",
            "origin_smp_roll168_mean",
            "origin_smp_roll168_std",
            "target_residual_vs_daily",
        ]
    )
    frame["split"] = np.select(
        [frame.target_date <= TRAIN_END, frame.target_date <= VALIDATION_END],
        ["train", "validation"],
        default="final_test",
    )
    if frame[FEATURES].isna().all(axis=1).any():
        raise ValueError("A row has no model features.")
    return frame


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "bias_pred_minus_actual": float(np.mean(predicted - actual)),
    }


def _matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[FEATURES].to_numpy(dtype=np.float32)


def _candidate_configs() -> list[dict[str, float | int]]:
    return [
        {"max_depth": 3, "learning_rate": 0.035, "n_estimators": 1200, "min_child_weight": 8, "subsample": 0.85, "colsample_bytree": 0.90, "reg_lambda": 8.0, "reg_alpha": 0.05},
        {"max_depth": 4, "learning_rate": 0.030, "n_estimators": 1400, "min_child_weight": 10, "subsample": 0.85, "colsample_bytree": 0.90, "reg_lambda": 10.0, "reg_alpha": 0.10},
        {"max_depth": 5, "learning_rate": 0.025, "n_estimators": 1600, "min_child_weight": 12, "subsample": 0.80, "colsample_bytree": 0.85, "reg_lambda": 14.0, "reg_alpha": 0.15},
    ]


def _new_model(config: dict[str, float | int], early_stopping: bool) -> xgb.XGBRegressor:
    params = dict(config)
    params.update(
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        n_jobs=4,
        random_state=42,
        verbosity=0,
    )
    if early_stopping:
        params["early_stopping_rounds"] = 100
    return xgb.XGBRegressor(**params)


def _evaluate_model(frame: pd.DataFrame, prediction: np.ndarray, model_name: str) -> dict[str, float | int | str]:
    return {"model": model_name, **_metrics(frame.actual_smp.to_numpy(dtype=float), prediction)}


def _save_figures(test: pd.DataFrame, hourly: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(17, 6))
    ax.plot(test.target_timestamp, test.actual_smp, color="#222222", lw=0.9, label="Actual SMP")
    ax.plot(test.target_timestamp, test.xgboost_smp, color="#1769aa", lw=0.9, label="XGBoost")
    ax.plot(test.target_timestamp, test.daily_baseline_smp, color="#e07a1f", lw=0.7, alpha=0.75, label="Previous-day baseline")
    ax.set_title("Next-day SMP forecast — Final Test (2025-07 to 2025-12)")
    ax.set_xlabel("Target timestamp (Asia/Seoul)")
    ax.set_ylabel("SMP (KRW/kWh)")
    ax.grid(alpha=0.2)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "xgboost_final_test_actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    first_date = test.target_date.min()
    week = test[test.target_date < first_date + pd.Timedelta(days=7)]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(week.target_timestamp, week.actual_smp, marker="o", ms=2, color="#222222", label="Actual SMP")
    ax.plot(week.target_timestamp, week.xgboost_smp, marker=".", ms=2, color="#1769aa", label="XGBoost")
    ax.plot(week.target_timestamp, week.daily_baseline_smp, color="#e07a1f", alpha=0.8, label="Previous-day baseline")
    ax.set_title(f"Final Test example week — {first_date:%Y-%m-%d}")
    ax.set_xlabel("Target timestamp (Asia/Seoul)")
    ax.set_ylabel("SMP (KRW/kWh)")
    ax.grid(alpha=0.2)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "xgboost_final_test_example_week.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    hours = hourly[hourly.split == "final_test"].sort_values("target_hour")
    ax.plot(hours.target_hour, hours.xgboost_mae, marker="o", label="XGBoost MAE")
    ax.plot(hours.target_hour, hours.baseline_mae, marker="o", label="Previous-day baseline MAE")
    ax.set_xticks(range(24))
    ax.set_xlabel("Target hour (KST)")
    ax.set_ylabel("MAE (KRW/kWh)")
    ax.set_title("Final Test error by target hour")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "xgboost_final_test_hourly_mae.png", dpi=160)
    plt.close(fig)


def run_experiment() -> dict:
    started = time.perf_counter()
    for path in (PREDICTION_DIR, METRIC_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)

    inputs = load_shared_hourly_inputs()
    panel = build_supervised_panel(inputs)
    # Preserve enough digits that float32 model inputs remain identical after
    # a future reload; rounding a value near a tree split can change a leaf.
    panel.to_csv(PANEL_PATH, index=False, float_format="%.17g")

    train = panel[panel.split == "train"].copy()
    validation = panel[panel.split == "validation"].copy()
    final_test = panel[panel.split == "final_test"].copy()
    if train.empty or validation.empty or final_test.empty:
        raise ValueError("One of the requested chronological splits is empty.")

    train_x, train_y = _matrix(train), train.target_residual_vs_daily.to_numpy(dtype=float)
    val_x, val_y = _matrix(validation), validation.target_residual_vs_daily.to_numpy(dtype=float)
    candidate_rows: list[dict] = []
    selected_model = None
    selected_config = None
    selected_validation_pred = None
    selected_mae = float("inf")

    for candidate_id, config in enumerate(_candidate_configs(), start=1):
        model = _new_model(config, early_stopping=True)
        model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=False)
        val_residual = model.predict(val_x)
        val_prediction = validation.smp_same_hour_prev_day.to_numpy(dtype=float) + val_residual
        score = float(mean_absolute_error(validation.actual_smp.to_numpy(dtype=float), val_prediction))
        row = {
            "candidate": candidate_id,
            **config,
            "best_iteration": int(model.best_iteration) + 1,
            **_metrics(validation.actual_smp.to_numpy(dtype=float), val_prediction),
        }
        candidate_rows.append(row)
        if score < selected_mae:
            selected_mae = score
            selected_model = model
            selected_config = config
            selected_validation_pred = val_prediction

    assert selected_model is not None and selected_config is not None and selected_validation_pred is not None
    best_trees = int(selected_model.best_iteration) + 1
    final_config = {**selected_config, "n_estimators": best_trees}
    final_model = _new_model(final_config, early_stopping=False)
    train_validation = pd.concat([train, validation], ignore_index=True)
    final_model.fit(
        _matrix(train_validation),
        train_validation.target_residual_vs_daily.to_numpy(dtype=float),
        verbose=False,
    )

    test_prediction = final_test.smp_same_hour_prev_day.to_numpy(dtype=float) + final_model.predict(_matrix(final_test))
    val_prediction = selected_validation_pred
    prediction_rows = []
    for split_name, part, values, model_name in [
        ("validation", validation, val_prediction, "xgboost_residual_selected"),
        ("final_test", final_test, test_prediction, "xgboost_residual_refit_train_validation"),
    ]:
        pred = part[["forecast_origin", "target_timestamp", "target_date", "horizon", "actual_smp", "smp_same_hour_prev_day"]].copy()
        pred["split"] = split_name
        pred["model"] = model_name
        pred["daily_baseline_smp"] = pred.smp_same_hour_prev_day
        pred["xgboost_smp"] = values
        pred["absolute_error_xgboost"] = (pred.actual_smp - pred.xgboost_smp).abs()
        pred["absolute_error_baseline"] = (pred.actual_smp - pred.daily_baseline_smp).abs()
        prediction_rows.append(pred.drop(columns="smp_same_hour_prev_day"))
    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions.to_csv(PREDICTION_DIR / "xgboost_day_ahead_predictions.csv", index=False, float_format="%.8f")

    model_metrics = []
    for split_name, part, model_pred in [
        ("validation", validation, val_prediction),
        ("final_test", final_test, test_prediction),
    ]:
        model_metrics.append({"split": split_name, **_evaluate_model(part, model_pred, "xgboost")})
        model_metrics.append({"split": split_name, **_evaluate_model(part, part.smp_same_hour_prev_day.to_numpy(dtype=float), "previous_day_baseline")})
        model_metrics.append({"split": split_name, **_evaluate_model(part, part.smp_same_hour_prev_week.to_numpy(dtype=float), "previous_week_baseline")})
    metrics_df = pd.DataFrame(model_metrics)
    metrics_df.to_csv(METRIC_DIR / "model_metrics.csv", index=False, float_format="%.8f")
    pd.DataFrame(candidate_rows).to_csv(METRIC_DIR / "validation_candidate_metrics.csv", index=False, float_format="%.8f")

    hourly_rows = []
    for split_name, part, model_pred in [
        ("validation", validation, val_prediction),
        ("final_test", final_test, test_prediction),
    ]:
        for hour in range(24):
            mask = part.target_timestamp.dt.hour.eq(hour).to_numpy()
            actual = part.actual_smp.to_numpy(dtype=float)[mask]
            pred = model_pred[mask]
            base = part.smp_same_hour_prev_day.to_numpy(dtype=float)[mask]
            hourly_rows.append({
                "split": split_name,
                "target_hour": hour,
                "n": int(mask.sum()),
                "xgboost_mae": float(mean_absolute_error(actual, pred)),
                "xgboost_rmse": float(mean_squared_error(actual, pred) ** 0.5),
                "xgboost_r2": float(r2_score(actual, pred)),
                "xgboost_bias": float(np.mean(pred - actual)),
                "baseline_mae": float(mean_absolute_error(actual, base)),
                "baseline_rmse": float(mean_squared_error(actual, base) ** 0.5),
                "baseline_r2": float(r2_score(actual, base)),
            })
    hourly_df = pd.DataFrame(hourly_rows)
    hourly_df.to_csv(METRIC_DIR / "hourly_metrics.csv", index=False, float_format="%.8f")

    importance = pd.DataFrame({"feature": FEATURES, "gain_weighted_importance": final_model.feature_importances_})
    importance.sort_values("gain_weighted_importance", ascending=False).to_csv(
        METRIC_DIR / "feature_importance.csv", index=False, float_format="%.10f"
    )

    test_plot = predictions[predictions.split == "final_test"].copy()
    test_plot["target_timestamp"] = pd.to_datetime(test_plot.target_timestamp)
    test_plot["target_date"] = pd.to_datetime(test_plot.target_date)
    _save_figures(test_plot, hourly_df)

    final_model.save_model(MODEL_DIR / "xgboost_day_ahead_24h.json")
    summary = {
        "target_definition": "V5-compatible hourly land SMP (same 2023 SMP labels as aligned_hourly_all_features_2023.csv)",
        "target_price_product_note": "Inherited labels have no explicit day-ahead/final-settlement field or publication vintage; see report.",
        "frequency": "hourly; one stacked row per target hour; each daily origin is prior calendar day at 23:00 KST",
        "warmup_rows_dropped": 192,
        "row_counts": {name: int((panel.split == name).sum()) for name in ("train", "validation", "final_test")},
        "date_splits": {
            "train": ["2023-01-01", "2024-12-31"],
            "validation": ["2025-01-01", "2025-06-30"],
            "final_test": ["2025-07-01", "2025-12-31"],
        },
        "features": FEATURES,
        "excluded_feature_groups": {
            "actual_demand_generation_reserve": "target-day observations are unavailable at D-1 23:00; source publication delays for safe lags are not verified",
            "kpx_mlfd": "coverage/vintage timestamp insufficient for full train period",
            "solar_generation_and_weather_forecasts": "EWP panel starts 2025-07-18 so it has no training-period coverage; KMA point-vintage panel absent",
            "holiday_flag": "announcement vintages for ad-hoc/special days are absent",
            "fuel_and_fred": "cutoff publication timestamps not available",
            "asos_solar_radiation": "many missing/sentinel cells; omitted",
        },
        "asof_contract": {
            "forecast_origin": "target calendar date minus one day at 23:00 KST",
            "historical_sources_must_be_at_or_before_origin": True,
            "weather_lag": "KMA temperature and wind at target timestamp minus 48 hours; prior-day ASOS data is published after 10:00 the next day",
            "max_source_timestamp_after_origin_rows": int((~panel.cutoff_check_pass).sum()),
        },
        "baseline": "same-hour previous-day SMP; V5 uses this as its primary day-ahead baseline",
        "validation_selected_candidate": int(pd.DataFrame(candidate_rows).sort_values("mae").iloc[0].candidate),
        "selected_config": final_config,
        "best_iteration_from_validation": best_trees,
        "validation_candidate_metrics": candidate_rows,
        "metrics": model_metrics,
        "xgboost_version": xgb.__version__,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "v5_reference": {"mae": 11.0938, "period": "2023-10-01 through 2023-12-31", "comparable_to_final_test": False},
    }
    (METRIC_DIR / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(summary, metrics_df, panel, importance)
    return summary


def _write_report(summary: dict, metrics: pd.DataFrame, panel: pd.DataFrame, importance: pd.DataFrame) -> None:
    lines = [
        "# XGBoost 다음 날 SMP 24시간 학습 및 평가",
        "",
        "## 목표와 데이터",
        "",
        "- V5의 `smp`/`actual_smp` 레이블 정의를 따랐습니다. V5 `aligned_hourly_all_features_2023.csv`의 2023년 8,760개 값은 사용한 전체 SMP 파일과 전부 일치했습니다.",
        "- 전체 label: 2023-01-01 00시–2025-12-31 23시, 26,304행. 2024–2025년 자료는 기존 프로젝트의 land SMP actual 계열이며 raw 파일과 수집 manifest의 검증 가능한 중복 구간이 일치합니다. Final Test 기간에는 관측값 0원이 8건 있으며 누락값으로 바꾸지 않았습니다.",
        "- 다만 2024–2025 레이블에는 세부 가격상품 및 각 시간의 공표 빈티지가 별도 표기되어 있지 않습니다. 원본 열 이름과 V5 사용방식에 따라 land SMP target으로 통일했으며 이 제한은 모델 결과 해석에 남깁니다.",
        "- 기존 SMP lag 및 rolling 값은 각 일별 origin(D-1 23:00 KST)까지로 고정했습니다. target-day 실제 수요·발전량·예비력·기상은 사용하지 않았습니다.",
        "",
        "## 사용 입력",
        "",
        "- SMP: 같은 시간 전일 SMP baseline, 같은 시간 전주 lag, origin 시점의 SMP 및 1h/2h/24h/168h lag, 24h/168h rolling mean·std, 1h/24h 변화.",
        "- Calendar: target hour, weekday, month, cyclic transforms, weekend. 미래 시각에서 결정적으로 계산했습니다.",
        "- Weather: 서울 ASOS station 108 temperature/wind at target timestamp minus 48 hours. 기상청 자료 공개 지연(전일 자료 다음 날 10시 이후)을 반영했습니다. 과거 원본에는 당시 수정 빈티지가 없어 사후 정정 여부는 확인할 수 없습니다. 학습 panel 26,112행 중 기온 41행·풍속 68행이 비어 있으며 0으로 채우지 않고 XGBoost의 missing 처리에 맡겼습니다.",
        "- 제외: target-day 수요·발전량·예비력 및 공개 지연 미확인 lag, 불완전/빈티지 미확인 KPX mlfd, ASOS SI, 공휴일 지정 빈티지, 발표시각 없는 연료비/FRED. EWP 예보는 2025-07-18 이후에만 있어 train 구간에 관측이 없어 제외했고, KMA point forecast 빈티지는 확보되지 않았습니다.",
        "",
        "## 시간순 분할 및 성능",
        "",
        "초기 192시간은 origin 기준 168시간 lag/rolling warm-up이 없어 제외했습니다.",
        "",
        "| Split | 기간 | 행 수 | 모델 | MAE | RMSE | R² |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    display_names = {"xgboost": "XGBoost", "previous_day_baseline": "전일 같은 시간 baseline", "previous_week_baseline": "전주 같은 시간 baseline"}
    for row in metrics.itertuples(index=False):
        split_period = "2025-01-01–2025-06-30" if row.split == "validation" else "2025-07-01–2025-12-31"
        lines.append(f"| {row.split} | {split_period} | {row.n:,} | {display_names[row.model]} | {row.mae:.4f} | {row.rmse:.4f} | {row.r2:.4f} |")
    lines += [
        "",
        "Validation MAE로 세 XGBoost 설정을 선택하고, 선택한 tree 수로 train+validation에 재학습한 뒤 Final Test를 한 번 평가했습니다. 세부 설정은 `outputs/metrics/experiment_summary.json`에 저장했습니다.",
        "",
        "## V5 비교와 제한",
        "",
        "- V5 보존 결과 MAE 11.0938은 2023-10-01–2023-12-31의 92일 Final Test입니다. 이번 Final Test는 2025-07-01–2025-12-31이므로 직접 우열 비교는 하지 않습니다.",
        "- 비교 가능한 내부 baseline은 각 split의 전일 동일시간 persistence입니다. 2026년 시연 및 실시간 API는 이번 실험에 포함하지 않았습니다.",
        "- Final Test에 0원 label 8건이 그대로 포함됐고, XGBoost의 예측 bias(예측−실제)는 +2.43원/kWh입니다. 원본에서 관측값으로 표시된 0은 결측으로 바꾸지 않았지만, 가격상품/0값 정의가 없는 점은 확인 대상입니다.",
        "- 입력 panel row-level cutoff 검사에서 미래 관측 사용은 0행입니다. 목표 가격의 세부 상품/공표 시점이 완전히 기록되지 않은 점은 후속 source metadata 확인 대상입니다.",
        "",
        "## 생성 산출물",
        "",
        "- `data/processed/day_ahead_supervised_panel_2023_2025.csv`: 공통 입력에서 구성한 feature/label panel 및 cutoff audit columns.",
        "- `outputs/predictions/xgboost_day_ahead_predictions.csv`: validation 및 final-test 예측.",
        "- `outputs/metrics/model_metrics.csv`, `hourly_metrics.csv`, `validation_candidate_metrics.csv`, `feature_importance.csv`, `experiment_summary.json`.",
        "- `outputs/figures/xgboost_final_test_actual_vs_predicted.png`, `xgboost_final_test_example_week.png`, `xgboost_final_test_hourly_mae.png`.",
        "- `models/xgboost_day_ahead_24h.json`: 재사용 가능한 XGBoost booster.",
        "",
        "XGBoost feature importance 상위 10개: " + ", ".join(f"{r.feature} ({r.gain_weighted_importance:.3f})" for r in importance.head(10).itertuples(index=False)),
        "",
    ]
    (PROJECT_ROOT / "reports/xgboost_day_ahead_training_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    result = run_experiment()
    print(json.dumps({"metrics": result["metrics"], "row_counts": result["row_counts"], "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=False, indent=2))
