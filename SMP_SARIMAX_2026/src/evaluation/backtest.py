from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import ARTIFACTS, OUT, PROCESSED
from src.features.calendar import calendar_features
from src.evaluation.rows import make_forecast_row
from src.models.sarimax import CANDIDATES, fit, forecast_72


def _metric_rows(df: pd.DataFrame, split: str) -> list[dict]:
    rows = []
    def add(group: pd.DataFrame, label: str, value: str):
        clean = group.dropna(subset=["actual_smp", value])
        if clean.empty: return
        rows.append({"split": split, "group": label, "count": len(clean), "mae": mean_absolute_error(clean.actual_smp, clean[value]), "rmse": mean_squared_error(clean.actual_smp, clean[value]) ** 0.5, "r2": r2_score(clean.actual_smp, clean[value]) if len(clean) > 1 else np.nan})
    add(df, "overall_sarimax", "predicted_smp")
    for day, part in df.groupby("forecast_day"): add(part, f"sarimax_{day}", "predicted_smp")
    for period, part in df.groupby("forecast_date"): add(part, f"daily_{period}", "predicted_smp")
    for period, part in df.groupby("target_month"): add(part, f"monthly_{period}", "predicted_smp")
    for h, part in df.groupby("horizon"): add(part, f"horizon_{h}", "predicted_smp")
    for method in ("previous_day_smp", "previous_week_smp"): add(df, method, method)
    return rows


def run_backtest() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = PROCESSED / "hourly_smp_2023_2025.csv"
    if not path.exists(): raise FileNotFoundError("Run `python -m src.cli preprocess` first.")
    frame = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
    y = pd.to_numeric(frame.smp, errors="coerce").asfreq("h")
    # Validation targets have to be complete; report gaps instead of filling actual SMP.
    required_train = y.loc["2023-01-01":"2024-12-31 23:00"]
    if required_train.isna().any() or len(required_train) != 17544:
        raise ValueError(f"Train requires 17,544 hourly SMP observations (2023–2024 includes leap year); found {int(required_train.notna().sum())}. No missing target values are imputed.")
    candidates = []
    candidate_validation = {}
    val_origins = pd.date_range("2024-12-31 23:00", "2025-06-27 23:00", freq="24h")
    if y.loc["2025-01-01":"2025-06-30 23:00"].isna().any():
        raise ValueError("Validation window has missing hourly SMP. Missing data are not imputed; see processed coverage report.")
    for candidate in CANDIDATES:
        print(f"Fitting validation candidate {candidate.name}...", flush=True)
        started = time.perf_counter()
        result, fit_seconds = fit(required_train, candidate)
        converged = bool(result.mle_retvals.get("converged", False))
        rows = []
        current = required_train.index[-1]
        for origin in val_origins:
            if origin > current:
                new_index = pd.date_range(current + pd.Timedelta(hours=1), origin, freq="h")
                result = result.extend(endog=y.reindex(new_index), exog=calendar_features(new_index))
                current = origin
            pred = forecast_72(result, origin)
            rows.append(make_forecast_row(origin, y, pred, "validation"))
        val = pd.concat(rows, ignore_index=True)
        mae = float(val.absolute_error.mean())
        candidate_validation[candidate.name] = val
        candidates.append({"candidate": candidate.name, "order": candidate.order, "seasonal_order": candidate.seasonal_order, "validation_mae": mae, "fit_seconds": fit_seconds, "elapsed_seconds": time.perf_counter()-started, "converged": converged})
        print(f"Validation candidate {candidate.name}: MAE={mae:.4f}, fit={fit_seconds:.1f}s, converged={converged}", flush=True)
    selected = min(candidates, key=lambda x: x["validation_mae"])
    chosen = next(c for c in CANDIDATES if c.name == selected["candidate"])
    validation = candidate_validation[selected["candidate"]]

    # Final test: same state-update rule, refit at the end of the complete validation period.
    final_train = y.loc[:"2025-06-30 23:00"]
    if final_train.isna().any() or len(final_train) != 21888:
        raise ValueError(f"Final-test initial fit requires complete train+validation (21,888 hours); found {int(final_train.notna().sum())}.")
    test_y = y.loc["2025-07-01":"2025-12-31 23:00"]
    if test_y.isna().any(): raise ValueError("Final-test window has missing SMP; missing targets are not imputed.")
    print(f"Refitting selected candidate {chosen.name} through 2025-06-30...", flush=True)
    test_result, final_fit_seconds = fit(final_train, chosen)
    final_converged = bool(test_result.mle_retvals.get("converged", False))
    test_origins = pd.date_range("2025-06-30 23:00", "2025-12-28 23:00", freq="24h")
    test_rows, current = [], final_train.index[-1]
    for origin in test_origins:
        if origin > current:
            new_index = pd.date_range(current + pd.Timedelta(hours=1), origin, freq="h")
            test_result = test_result.extend(endog=y.reindex(new_index), exog=calendar_features(new_index)); current = origin
        test_rows.append(make_forecast_row(origin, y, forecast_72(test_result, origin), "final_test"))
    final_test = pd.concat(test_rows, ignore_index=True)
    OUT.joinpath("predictions").mkdir(parents=True, exist_ok=True); OUT.joinpath("metrics").mkdir(parents=True, exist_ok=True)
    validation.to_csv(OUT / "predictions" / "validation_predictions.csv", index=False)
    final_test.to_csv(OUT / "predictions" / "final_test_predictions.csv", index=False)
    metrics = pd.DataFrame(_metric_rows(validation, "validation") + _metric_rows(final_test, "final_test"))
    metrics.to_csv(OUT / "metrics" / "metrics_by_split.csv", index=False)
    pd.DataFrame(candidates).to_json(OUT / "metrics" / "validation_experiments.json", orient="records", indent=2)
    ARTIFACTS.mkdir(exist_ok=True)
    with (ARTIFACTS / "selected_model.json").open("w", encoding="utf-8") as f:
        json.dump({"candidate": selected, "final_fit_seconds": final_fit_seconds, "final_fit_converged": final_converged, "state_update_rule": "daily extend observed actuals through forecast origin using filtered states; refit disabled"}, f, indent=2)
    return validation, final_test
