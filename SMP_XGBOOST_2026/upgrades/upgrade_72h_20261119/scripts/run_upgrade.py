"""Run E1 blending and E2/E3 Train+Validation refit on the fixed dataset."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb

from evaluate_upgrade import UPGRADE_DIR, DATASET_VERSION, metric_values, select_blend_weights


MODEL_DIR = UPGRADE_DIR.parents[1]
REPO_ROOT = MODEL_DIR.parent
DATASET_DIR = REPO_ROOT / "SMP_SHARED_DATA" / "versions" / DATASET_VERSION
LEGACY_PREDICTIONS = MODEL_DIR / "outputs" / "predictions" / "72h" / DATASET_VERSION / "predictions_72h.csv"
LEGACY_METRICS = MODEL_DIR / "outputs" / "metrics" / "72h" / DATASET_VERSION / "metrics_overall.csv"
LEGACY_SEARCH = MODEL_DIR / "outputs" / "metrics" / "72h" / DATASET_VERSION / "hyperparameter_search.csv"
LEGACY_CONFIG = MODEL_DIR / "artifacts" / "72h" / DATASET_VERSION / "model_config.json"
LEGACY_MODEL = MODEL_DIR / "artifacts" / "72h" / DATASET_VERSION / "xgboost_model.json"
KEYS = ["split", "forecast_origin", "observed_cutoff", "target_timestamp", "horizon", "forecast_day"]
MODEL_PATH = UPGRADE_DIR / "models" / "final_xgboost_model.json"
VALIDATION_PATH = UPGRADE_DIR / "outputs" / "predictions" / "experiments_validation.csv"
TEST_PATH = UPGRADE_DIR / "outputs" / "predictions" / "final_test_predictions.csv"
CONFIG_PATH = UPGRADE_DIR / "config" / "experiment_config.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _equal_numeric(left: pd.Series, right: pd.Series, name: str) -> None:
    a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    if not np.allclose(a, b, rtol=0, atol=1e-10, equal_nan=True):
        delta = np.nanmax(np.abs(a - b))
        raise ValueError(f"Existing prediction CSV and fixed dataset disagree for {name}; max difference={delta}")


def _check_expected_split(panel: pd.DataFrame) -> None:
    limits = {
        "train": (pd.Timestamp("2023-01-01"), pd.Timestamp("2025-01-01")),
        "validation": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-07-01")),
        "final_test": (pd.Timestamp("2025-07-01"), pd.Timestamp("2026-01-01")),
    }
    for split, (start, end) in limits.items():
        part = panel.loc[panel["split"] == split]
        if part.empty or not part["target_timestamp"].ge(start).all() or not part["target_timestamp"].lt(end).all():
            raise ValueError(f"Target timestamps do not match the fixed {split} period")
        per_origin = part.groupby("forecast_origin")["horizon"].agg(["count", "min", "max"])
        if not ((per_origin["count"] == 72) & (per_origin["min"] == 1) & (per_origin["max"] == 72)).all():
            raise ValueError(f"Some {split} forecast blocks do not contain horizons 1..72")


def _verify_baselines(panel: pd.DataFrame) -> None:
    expected_cutoff = panel["forecast_origin"] - pd.Timedelta(hours=72)
    if not expected_cutoff.equals(panel["observed_cutoff"]):
        raise ValueError("The shared observed cutoff is not forecast_origin minus 72 hours")
    horizon = (panel["target_timestamp"] - panel["forecast_origin"]).dt.total_seconds() / 3600
    if not np.array_equal(horizon.to_numpy(dtype=int), panel["horizon"].to_numpy(dtype=int)):
        raise ValueError("Target timestamps do not agree with the recorded forecast horizon")
    if not panel["forecast_origin"].dt.hour.eq(23).all():
        raise ValueError("A forecast origin is not at local 23:00")
    daily_date = pd.to_datetime(panel["baseline_origin_day"], errors="coerce").dt.normalize()
    if daily_date.isna().any():
        raise ValueError("Recent-daily baseline reference dates are missing")
    daily_reference_end = daily_date + pd.Timedelta(hours=23)
    weekly_reference = panel["target_timestamp"] - pd.Timedelta(days=7)
    if (daily_reference_end > panel["observed_cutoff"]).any():
        raise ValueError("Recent-daily baseline uses SMP after observed_cutoff")
    if (weekly_reference > panel["observed_cutoff"]).any():
        raise ValueError("Weekly baseline uses SMP after observed_cutoff")
    if panel[["baseline_recent_daily", "baseline_weekly"]].isna().any().any():
        raise ValueError("Baseline predictions contain missing values")


def _verify_e0_metrics(panel: pd.DataFrame, stored: pd.DataFrame, split: str) -> dict:
    part = panel.loc[panel["split"] == split]
    checks = {
        "mae": metric_values(part["actual_smp"], part["predicted_e0"])["mae"],
        "rmse": metric_values(part["actual_smp"], part["predicted_e0"])["rmse"],
        "r2": metric_values(part["actual_smp"], part["predicted_e0"])["r2"],
        "baseline_recent_daily_mae": metric_values(part["actual_smp"], part["baseline_recent_daily"])["mae"],
        "baseline_weekly_mae": metric_values(part["actual_smp"], part["baseline_weekly"])["mae"],
    }
    saved = stored.loc[stored["split"] == split]
    if len(saved) != 1:
        raise ValueError(f"Saved E0 overall metrics have no unique row for {split}")
    saved = saved.iloc[0]
    for name, value in checks.items():
        if not np.isclose(value, float(saved[name]), rtol=0, atol=1e-10):
            raise ValueError(f"Recomputed E0 {split} {name}={value} differs from saved value {saved[name]}")
    return {"split": split, "n_rows": len(part), **checks, "matches_existing_report": True}


def load_and_verify() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    for path in [DATASET_DIR / "manifest.json", DATASET_DIR / "forecast_inputs.csv", DATASET_DIR / "evaluation_targets.csv",
                 LEGACY_PREDICTIONS, LEGACY_METRICS, LEGACY_SEARCH, LEGACY_CONFIG, LEGACY_MODEL]:
        if not path.exists():
            raise FileNotFoundError(f"Required read-only E0 input is missing: {path}")
    manifest = json.loads((DATASET_DIR / "manifest.json").read_text(encoding="utf-8"))
    quality = json.loads((DATASET_DIR / "quality_report.json").read_text(encoding="utf-8"))
    legacy_config = json.loads(LEGACY_CONFIG.read_text(encoding="utf-8"))
    feature_cols = manifest["model_feature_columns"]
    if len(feature_cols) != len(set(feature_cols)):
        raise ValueError("The fixed feature manifest has duplicate columns")
    inputs = pd.read_csv(DATASET_DIR / "forecast_inputs.csv",
                         usecols=lambda col: col in set(KEYS + feature_cols), low_memory=False)
    targets = pd.read_csv(DATASET_DIR / "evaluation_targets.csv", low_memory=False)
    e0 = pd.read_csv(LEGACY_PREDICTIONS, low_memory=False)
    for frame in [inputs, targets, e0]:
        for column in ["forecast_origin", "observed_cutoff", "target_timestamp"]:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    if inputs["target_timestamp"].isna().any() or targets["target_timestamp"].isna().any() or e0["target_timestamp"].isna().any():
        raise ValueError("Invalid timestamp found in the fixed input or E0 result")
    panel = inputs.merge(targets, on=KEYS, how="inner", validate="one_to_one")
    if len(panel) != len(targets) or len(e0) != len(targets):
        raise ValueError(f"E0 and fixed target row counts differ: targets={len(targets)}, panel={len(panel)}, E0={len(e0)}")
    if e0.duplicated(KEYS).any():
        raise ValueError("The E0 prediction file contains duplicate evaluation keys")
    e0_key_match = panel[KEYS].merge(e0[KEYS], on=KEYS, how="outer", indicator=True)
    if not e0_key_match["_merge"].eq("both").all():
        raise ValueError("E0 predictions do not use the fixed common evaluation rows")
    e0_columns = KEYS + ["actual_smp", "predicted_smp", "baseline_origin_day", "baseline_recent_daily", "baseline_weekly", "prediction_status"]
    e0 = e0[e0_columns].rename(columns={
        "actual_smp": "actual_smp_e0", "baseline_origin_day": "baseline_origin_day_e0",
        "baseline_recent_daily": "baseline_recent_daily_e0", "baseline_weekly": "baseline_weekly_e0",
    })
    panel = panel.merge(e0, on=KEYS, how="inner", validate="one_to_one")
    _equal_numeric(panel["actual_smp"], panel["actual_smp_e0"], "actual_smp")
    _equal_numeric(panel["baseline_recent_daily"], panel["baseline_recent_daily_e0"], "baseline_recent_daily")
    _equal_numeric(panel["baseline_weekly"], panel["baseline_weekly_e0"], "baseline_weekly")
    panel["predicted_e0"] = pd.to_numeric(panel["predicted_smp"], errors="coerce")
    if panel["predicted_e0"].isna().any():
        raise ValueError("Existing E0 prediction is missing on a fixed evaluation row")

    _check_expected_split(panel)
    _verify_baselines(panel)
    stored = pd.read_csv(LEGACY_METRICS)
    verified = [_verify_e0_metrics(panel, stored, "validation")]
    return panel, {"manifest": manifest, "quality": quality, "legacy_config": legacy_config,
                   "feature_columns": feature_cols, "e0_verification": verified}, stored


def _prediction_rows(frame: pd.DataFrame, experiment_id: str, model_name: str,
                     prediction: pd.Series | np.ndarray, baseline_prediction: pd.Series | None,
                     alpha: float | None, baseline_source: pd.Series | None = None) -> pd.DataFrame:
    result = frame[KEYS + ["actual_smp", "baseline_origin_day", "baseline_recent_daily", "baseline_weekly"]].copy()
    result["dataset_version"] = DATASET_VERSION
    result["experiment_id"] = experiment_id
    result["model_name"] = model_name
    result["predicted_smp"] = np.asarray(prediction, dtype=float)
    result["absolute_error"] = (pd.to_numeric(result["actual_smp"], errors="coerce") - result["predicted_smp"]).abs()
    result["baseline_prediction"] = np.asarray(baseline_prediction, dtype=float) if baseline_prediction is not None else np.nan
    result["blending_weight"] = alpha if alpha is not None else np.nan
    result["prediction_status"] = np.where(np.isfinite(result["predicted_smp"]), "completed", "failed")
    result["baseline_reference_timestamp"] = ""
    if baseline_source is not None:
        result["baseline_reference_timestamp"] = pd.to_datetime(baseline_source).dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
    return result


def run() -> dict:
    if any(path.exists() for path in [MODEL_PATH, VALIDATION_PATH, TEST_PATH]):
        raise FileExistsError("Upgrade model or prediction outputs already exist; refusing to overwrite them")
    for directory in [MODEL_PATH.parent, VALIDATION_PATH.parent, TEST_PATH.parent,
                      UPGRADE_DIR / "outputs" / "metrics", CONFIG_PATH.parent]:
        directory.mkdir(parents=True, exist_ok=True)
    panel, context, stored_metrics = load_and_verify()
    feature_cols = context["feature_columns"]
    for column in feature_cols:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")

    validation = panel.loc[panel["split"] == "validation"].copy()
    selection = select_blend_weights(validation)
    selection_rows = pd.DataFrame(selection["candidate_rows"])
    selection_rows.to_csv(UPGRADE_DIR / "outputs" / "metrics" / "blend_candidates_validation.csv", index=False)
    pd.read_csv(LEGACY_SEARCH).to_csv(UPGRADE_DIR / "outputs" / "metrics" / "existing_hyperparameter_search.csv", index=False)
    parameters = context["legacy_config"]["selected_parameters"].copy()
    verified_base = context["e0_verification"]
    rows_by_split = {str(split): int(count) for split, count in panel.groupby("split").size().items()}
    origins_by_split = {str(split): int(count) for split, count in panel.groupby("split")["forecast_origin"].nunique().items()}

    # Lock the Validation-only selection before fitting the refit model or scoring Final Test.
    locked_config = {
        "dataset_version": DATASET_VERSION, "model_name": "XGBoost", "seed": context["legacy_config"]["seed"],
        "target": "land hourly SMP (KRW/kWh)", "timezone": "Asia/Seoul", "forecast_origin": "D 23:00",
        "horizon_hours": 72, "observed_cutoff": "forecast_origin - 72 hours",
        "data_and_feature_source": "Existing common dataset; no input groups, feature definitions, or data sources changed",
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "e0_source_files": {"predictions_sha256": sha256(LEGACY_PREDICTIONS), "metrics_sha256": sha256(LEGACY_METRICS),
                            "hyperparameter_search_sha256": sha256(LEGACY_SEARCH),
                            "model_sha256": sha256(LEGACY_MODEL), "dataset_manifest_sha256": sha256(DATASET_DIR / "manifest.json")},
        "e0_verification": verified_base,
        "evaluation_coverage": {"rows_by_split": rows_by_split, "origins_by_split": origins_by_split,
                                 "targets_per_origin": 72, "same_fixed_rows_as_e0": True},
        "baseline_availability_checks": {"recent_daily_source_day_23h_at_or_before_cutoff": True,
                                         "weekly_target_minus_7days_at_or_before_cutoff": True},
        "blend_selection": selection,
        "retrain_parameters": parameters,
        "final_test_used_for_selection": False,
        "final_test_period_previously_examined": True,
        "bias_correction_applied": False,
        "separate_forecast_day_models_applied": False,
        "random_state": context["legacy_config"]["seed"],
        "library_versions": {"python": platform.python_version(), "xgboost": xgb.__version__,
                              "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__},
    }
    CONFIG_PATH.write_text(json.dumps(locked_config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    # Only after the Validation selection has been written, recheck the known E0 Final Test score.
    final_test_e0_check = _verify_e0_metrics(panel, stored_metrics, "final_test")
    locked_config["e0_verification"].append(final_test_e0_check)
    CONFIG_PATH.write_text(json.dumps(locked_config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    baseline_name = selection["baseline"]
    baseline_col = "baseline_recent_daily" if baseline_name == "recent_daily" else "baseline_weekly"
    selected_alpha = float(selection["alpha"])
    alpha_by_baseline = {name: float(row["alpha"]) for name, row in selection["by_baseline"].items()}
    validation_rows = []
    for split_frame in [validation]:
        validation_rows.append(_prediction_rows(split_frame, "E0_xgb", "Existing XGBoost", split_frame["predicted_e0"], None, 1.0))
        for exp_id, name, baseline_col_one in [
            ("baseline_recent_daily", "Recent daily baseline", "baseline_recent_daily"),
            ("baseline_weekly", "Weekly baseline", "baseline_weekly"),
        ]:
            baseline = split_frame[baseline_col_one]
            source = (pd.to_datetime(split_frame["baseline_origin_day"]).dt.normalize() + pd.Timedelta(hours=23)
                      if baseline_col_one == "baseline_recent_daily" else split_frame["target_timestamp"] - pd.Timedelta(days=7))
            validation_rows.append(_prediction_rows(split_frame, exp_id, name, baseline, baseline, 0.0, source))
        for family, col in [("recent_daily", "baseline_recent_daily"), ("weekly", "baseline_weekly")]:
            alpha = alpha_by_baseline[family]
            blend = alpha * split_frame["predicted_e0"].to_numpy(dtype=float) + (1 - alpha) * split_frame[col].to_numpy(dtype=float)
            source = (pd.to_datetime(split_frame["baseline_origin_day"]).dt.normalize() + pd.Timedelta(hours=23)
                      if family == "recent_daily" else split_frame["target_timestamp"] - pd.Timedelta(days=7))
            validation_rows.append(_prediction_rows(split_frame, f"E1_{family}_blend", f"E1 XGBoost + {family} baseline", blend, split_frame[col], alpha, source))
        selected_blend = selected_alpha * split_frame["predicted_e0"].to_numpy(dtype=float) + (1 - selected_alpha) * split_frame[baseline_col].to_numpy(dtype=float)
        selected_source = (pd.to_datetime(split_frame["baseline_origin_day"]).dt.normalize() + pd.Timedelta(hours=23)
                           if baseline_name == "recent_daily" else split_frame["target_timestamp"] - pd.Timedelta(days=7))
        validation_rows.append(_prediction_rows(split_frame, "E1_selected_blend", "E1 validation-selected blend", selected_blend, split_frame[baseline_col], selected_alpha, selected_source))
    pd.concat(validation_rows, ignore_index=True).to_csv(VALIDATION_PATH, index=False)

    x = panel[feature_cols].to_numpy(dtype=np.float32, na_value=np.nan, copy=True)
    y = pd.to_numeric(panel["actual_smp"], errors="coerce").to_numpy(dtype=np.float32)
    split = panel["split"].astype(str).to_numpy()
    train_val_mask = np.isin(split, ["train", "validation"])
    test_mask = split == "final_test"
    if np.isnan(y[train_val_mask]).any() or np.isnan(y[test_mask]).any():
        raise ValueError("SMP labels are missing in the Train+Validation fit or fixed Final Test set")
    if not train_val_mask.any() or not test_mask.any():
        raise ValueError("Train+Validation or Final Test partition is empty")
    n_jobs = max(1, min(8, (os.cpu_count() or 4) - 1))
    model = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="mae", tree_method="hist",
                             max_bin=256, n_jobs=n_jobs, random_state=int(context["legacy_config"]["seed"]),
                             verbosity=0, **parameters)
    fit_start = time.perf_counter()
    model.fit(x[train_val_mask], y[train_val_mask], verbose=False)
    fit_seconds = time.perf_counter() - fit_start
    model.save_model(MODEL_PATH)
    test_prediction = model.predict(x[test_mask]).astype(float)

    locked_config["retraining"] = {"training_rows": int(train_val_mask.sum()), "feature_count": len(feature_cols),
                                   "train_period": "2023-01-01 through 2024-12-31",
                                   "validation_period": "2025-01-01 through 2025-06-30",
                                   "fit_period": "Train + Validation through 2025-06-30",
                                   "fit_seconds": fit_seconds, "n_jobs": n_jobs,
                                   "final_fit_policy": "fixed Validation-selected parameters refit on Train+Validation; no Final Test fitting"}
    locked_config["model_file_sha256"] = sha256(MODEL_PATH)
    CONFIG_PATH.write_text(json.dumps(locked_config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    weights = booster.get_score(importance_type="weight")
    pd.DataFrame([{"feature": feature, "gain": float(gain.get(f"f{i}", 0.0)),
                   "split_count": int(weights.get(f"f{i}", 0.0))} for i, feature in enumerate(feature_cols)]) \
      .sort_values(["gain", "split_count"], ascending=False).to_csv(MODEL_PATH.parent / "feature_importance.csv", index=False)

    final_test = panel.loc[test_mask].copy()
    final_test["predicted_e2"] = test_prediction
    chosen_baseline = final_test[baseline_col]
    e1_daily_alpha = alpha_by_baseline["recent_daily"]
    e1_weekly_alpha = alpha_by_baseline["weekly"]
    e1_daily = e1_daily_alpha * final_test["predicted_e0"].to_numpy(dtype=float) + (1 - e1_daily_alpha) * final_test["baseline_recent_daily"].to_numpy(dtype=float)
    e1_weekly = e1_weekly_alpha * final_test["predicted_e0"].to_numpy(dtype=float) + (1 - e1_weekly_alpha) * final_test["baseline_weekly"].to_numpy(dtype=float)
    e1_selected = selected_alpha * final_test["predicted_e0"].to_numpy(dtype=float) + (1 - selected_alpha) * chosen_baseline.to_numpy(dtype=float)
    e3 = selected_alpha * final_test["predicted_e2"].to_numpy(dtype=float) + (1 - selected_alpha) * chosen_baseline.to_numpy(dtype=float)
    final_rows = [
        _prediction_rows(final_test, "E0_xgb", "Existing XGBoost", final_test["predicted_e0"], None, 1.0),
        _prediction_rows(final_test, "baseline_recent_daily", "Recent daily baseline", final_test["baseline_recent_daily"], final_test["baseline_recent_daily"], 0.0,
                         pd.to_datetime(final_test["baseline_origin_day"]).dt.normalize() + pd.Timedelta(hours=23)),
        _prediction_rows(final_test, "baseline_weekly", "Weekly baseline", final_test["baseline_weekly"], final_test["baseline_weekly"], 0.0,
                         final_test["target_timestamp"] - pd.Timedelta(days=7)),
        _prediction_rows(final_test, "E1_recent_daily_blend", "E1 XGBoost + recent daily baseline", e1_daily, final_test["baseline_recent_daily"], e1_daily_alpha,
                         pd.to_datetime(final_test["baseline_origin_day"]).dt.normalize() + pd.Timedelta(hours=23)),
        _prediction_rows(final_test, "E1_weekly_blend", "E1 XGBoost + weekly baseline", e1_weekly, final_test["baseline_weekly"], e1_weekly_alpha,
                         final_test["target_timestamp"] - pd.Timedelta(days=7)),
        _prediction_rows(final_test, "E1_selected_blend", "E1 validation-selected blend", e1_selected, chosen_baseline, selected_alpha,
                         pd.to_datetime(final_test["baseline_origin_day"]).dt.normalize() + pd.Timedelta(hours=23)
                         if baseline_name == "recent_daily" else final_test["target_timestamp"] - pd.Timedelta(days=7)),
        _prediction_rows(final_test, "E2_train_plus_validation", "Train+Validation refit XGBoost", final_test["predicted_e2"], None, 1.0),
        _prediction_rows(final_test, "E3_retrained_blend", "Train+Validation refit + selected blend", e3, chosen_baseline, selected_alpha,
                         pd.to_datetime(final_test["baseline_origin_day"]).dt.normalize() + pd.Timedelta(hours=23)
                         if baseline_name == "recent_daily" else final_test["target_timestamp"] - pd.Timedelta(days=7)),
    ]
    pd.concat(final_rows, ignore_index=True).to_csv(TEST_PATH, index=False)
    run_meta = {"dataset_version": DATASET_VERSION, "experiment_ids": sorted(pd.concat(final_rows)["experiment_id"].unique().tolist()),
                "validation_rows_per_experiment": rows_by_split["validation"],
                "final_test_rows_per_experiment": rows_by_split["final_test"],
                "validation_origins": origins_by_split["validation"], "final_test_origins": origins_by_split["final_test"],
                "model_file": str(MODEL_PATH.relative_to(UPGRADE_DIR)), "fit_seconds": fit_seconds,
                "selection_locked_before_final_test_scoring": True}
    (UPGRADE_DIR / "outputs" / "metrics" / "run_metadata.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"blend_selection": {"baseline": baseline_name, "alpha": selected_alpha,
                                           "validation_mae": selection["validation_mae"]},
                      "retraining_rows": int(train_val_mask.sum()), "fit_seconds": fit_seconds,
                      "model": str(MODEL_PATH), "validation_predictions": str(VALIDATION_PATH),
                      "final_test_predictions": str(TEST_PATH)}, ensure_ascii=False, indent=2))
    return locked_config


def main() -> None:
    run()


if __name__ == "__main__":
    main()
