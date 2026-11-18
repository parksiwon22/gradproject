"""Train LightGBM from the shared 12-group smp72h_common_v3 panel.

This adapter keeps the shared source panel immutable, joins its frozen labels,
and records whether source gates are fully cleared or the run is conditional.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

from train_lightgbm_72h import candidate_configs, metrics, plot_outputs


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SHARED_ROOT = PROJECT_ROOT.parent / "SMP_SHARED_DATA"
DEFAULT_VERSION = "smp72h_common_v3_20261119"
SEED = 42
KEYS = ["forecast_origin", "target_timestamp"]
SPLIT_RANGES = {
    "train": (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31 23:00")),
    "validation": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30 23:00")),
    "final_test": (pd.Timestamp("2025-07-01"), pd.Timestamp("2025-12-31 23:00")),
}


def load_and_validate(version: str, version_root: Path, gate_path: Path, allow_conditional: bool):
    manifest_path = version_root / "manifest.json"
    quality_path = version_root / "quality_report.json"
    panel_path = version_root / "forecast_inputs.csv"
    target_path = version_root / "evaluation_targets.csv"
    feature_path = version_root / "feature_manifest.csv"
    for path in (manifest_path, quality_path, panel_path, target_path, feature_path, gate_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required training input is missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    quality = json.loads(quality_path.read_text(encoding="utf-8-sig"))
    gate = json.loads(gate_path.read_text(encoding="utf-8-sig"))
    if manifest.get("dataset_version") != version:
        raise ValueError("Requested version does not match the shared dataset manifest.")
    if gate.get("dataset_version") != version:
        raise ValueError("Gate report is for a different shared dataset version.")
    gate_passed = bool(gate.get("data_gate_passed"))
    if not gate_passed and not allow_conditional:
        raise RuntimeError("The source gate is conditional. Use --allow-conditional only for a documented research run.")
    if not gate.get("training_authorized_by_user"):
        raise RuntimeError("The gate report does not record the user's instruction to train.")

    feature_manifest = pd.read_csv(feature_path)
    groups = set(pd.to_numeric(feature_manifest["input_group"], errors="coerce").dropna().astype(int))
    if groups != set(range(1, 13)):
        raise ValueError(f"Feature manifest must contain all 12 groups; found {sorted(groups)}.")
    features = manifest.get("model_feature_columns", [])
    if not features or len(features) != len(set(features)):
        raise ValueError("Shared manifest must provide a unique, non-empty model_feature_columns list.")

    frame = pd.read_csv(panel_path, parse_dates=["forecast_origin", "observed_cutoff", "target_timestamp"])
    targets = pd.read_csv(target_path, parse_dates=["forecast_origin", "observed_cutoff", "target_timestamp"])
    missing_features = sorted(set(features).difference(frame.columns))
    if missing_features:
        raise ValueError("Shared panel is missing model features: " + ", ".join(missing_features[:20]))
    if not set(KEYS + ["split", "actual_smp", "baseline_recent_daily", "baseline_weekly"]).issubset(targets.columns):
        raise ValueError("Frozen evaluation_targets.csv is missing target labels or causal baseline columns.")
    if frame.duplicated(KEYS).any() or targets.duplicated(KEYS).any():
        raise ValueError("Duplicate origin-target keys exist in the panel or target schedule.")
    if len(frame) != len(targets):
        raise ValueError("Feature panel and frozen target schedule have different row counts.")
    if not frame[KEYS + ["split", "observed_cutoff"]].equals(targets[KEYS + ["split", "observed_cutoff"]]):
        targets = targets.rename(columns={"split": "target_split", "observed_cutoff": "target_cutoff"})
        frame = frame.merge(targets, on=KEYS, how="left", validate="one_to_one")
        if frame["split"].ne(frame["target_split"]).any() or frame["observed_cutoff"].ne(frame["target_cutoff"]).any():
            raise ValueError("Feature panel and frozen target schedule do not align exactly.")
        frame = frame.drop(columns=["target_split", "target_cutoff"])
    else:
        label_cols = [c for c in targets.columns if c not in frame.columns or c in {"actual_smp", "baseline_recent_daily", "baseline_weekly", "baseline_origin_day"}]
        frame = frame.merge(targets[KEYS + label_cols], on=KEYS, how="left", validate="one_to_one", suffixes=("", "_target"))

    if frame["actual_smp"].isna().any():
        raise ValueError("Frozen SMP labels contain missing values.")
    if not set(frame["split"].unique()) == set(SPLIT_RANGES):
        raise ValueError("The panel must contain train, validation, and final_test splits.")
    origins = frame["forecast_origin"]
    if not ((origins.dt.hour == 23) & (origins.dt.minute == 0) & (origins.dt.second == 0)).all():
        raise ValueError("Every forecast origin must be D 23:00 local time.")
    if not (frame["observed_cutoff"] == origins - pd.Timedelta(hours=72)).all():
        raise ValueError("At least one origin does not use the common origin-minus-72-hour cutoff.")
    expected_targets = origins.dt.floor("D") + pd.Timedelta(days=1) + pd.to_timedelta(frame["horizon"] - 1, unit="h")
    if not (frame["target_timestamp"] == expected_targets).all():
        raise ValueError("Target timestamps do not match horizons 1..72.")
    counts = frame.groupby("forecast_origin")["horizon"].agg(["count", "min", "max"])
    if not ((counts["count"] == 72) & (counts["min"] == 1) & (counts["max"] == 72)).all():
        raise ValueError("Each issue must contain all 72 forecast horizons exactly once.")
    if (frame.groupby("forecast_origin")["split"].nunique() != 1).any():
        raise ValueError("A 72-hour issue crosses split assignments.")
    for split, (start, end) in SPLIT_RANGES.items():
        part = frame.loc[frame["split"].eq(split), "target_timestamp"]
        if part.empty or part.lt(start).any() or part.gt(end).any():
            raise ValueError(f"Target timestamps fall outside the declared {split} range.")

    # Archived forecast issue times and vintage dates are retained in the shared panel.
    for column in ("kma_temp_issue_time", "kma_weather_metro_am_issue_time", "kma_weather_metro_pm_issue_time"):
        if column in frame:
            issue = pd.to_datetime(frame[column], errors="coerce")
            if (issue.notna() & (issue > frame["forecast_origin"])).any():
                raise ValueError(f"A future-issued forecast was found in {column}.")
    for column in ("fred_oil_asof_date", "fred_energy_asof_date"):
        if column in frame:
            asof = pd.to_datetime(frame[column], errors="coerce")
            if (asof.notna() & (asof > frame["forecast_origin"])).any():
                raise ValueError(f"A future FRED vintage was found in {column}.")

    checks = {
        "observed_cutoff_check": quality.get("observed_cutoff_check", {}).get("passed") is True,
        "future_observation_perturbation_check": quality.get("future_observation_perturbation_check", {}).get("passed") is True,
        "fred_d2_vintage_check": quality.get("fred_d2_vintage_check", {}).get("passed") is True,
    }
    if not all(checks.values()):
        raise ValueError(f"Shared dataset leakage checks are not all passed: {checks}")

    frame["dataset_version"] = version
    frame["forecast_date"] = frame["target_timestamp"].dt.strftime("%Y-%m-%d")
    frame["fuel_input_policy"] = "same_target_month_conditional_experiment"
    numeric_features = frame[features].apply(pd.to_numeric, errors="coerce")
    frame.loc[:, features] = numeric_features
    missing_values = int(numeric_features.isna().sum().sum())
    frame.loc[:, features] = numeric_features
    return frame, features, manifest, quality, gate, missing_values, checks


def run(version: str, version_root: Path, output_root: Path, gate_path: Path, allow_conditional: bool) -> dict[str, Any]:
    frame, features, manifest, quality, gate, missing_values, checks = load_and_validate(
        version, version_root, gate_path, allow_conditional
    )
    started = time.perf_counter()
    train = frame.loc[frame.split.eq("train")]
    validation = frame.loc[frame.split.eq("validation")]
    test = frame.loc[frame.split.eq("final_test")]
    x_train, y_train = train[features], train["actual_smp"].to_numpy(float)
    x_val, y_val = validation[features], validation["actual_smp"].to_numpy(float)

    best: dict[str, Any] | None = None
    candidate_rows: list[dict[str, Any]] = []
    for index, config in enumerate(candidate_configs(), start=1):
        model = lgb.LGBMRegressor(
            objective="regression", n_estimators=1800, n_jobs=4, random_state=SEED,
            deterministic=True, force_col_wise=True, verbosity=-1, **config,
        )
        model.fit(
            x_train, y_train, eval_set=[(x_val, y_val)], eval_metric="l1",
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
        )
        row = {
            "candidate_id": index, **config,
            "best_iteration": int(model.best_iteration_ or model.n_estimators),
            **metrics(y_val, model.predict(x_val)),
        }
        candidate_rows.append(row)
        if best is None or row["mae"] < best["validation_mae"]:
            best = {"validation_mae": row["mae"], "config": config, "best_iteration": row["best_iteration"]}
    assert best is not None

    final_model = lgb.LGBMRegressor(
        objective="regression", n_estimators=best["best_iteration"], n_jobs=4,
        random_state=SEED, deterministic=True, force_col_wise=True, verbosity=-1,
        **best["config"],
    )
    final_model.fit(x_train, y_train)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    frame["predicted_smp"] = final_model.predict(frame[features])
    frame["absolute_error"] = (frame["actual_smp"] - frame["predicted_smp"]).abs()
    frame["model_name"] = "LightGBM"
    frame["prediction_status"] = "predicted_conditional" if not gate.get("data_gate_passed") else "predicted"
    target_columns = [
        "dataset_version", "model_name", "split", "forecast_origin", "observed_cutoff",
        "target_timestamp", "forecast_date", "horizon", "forecast_day", "actual_smp",
        "predicted_smp", "absolute_error", "baseline_recent_daily", "baseline_weekly",
        "prediction_status", "fuel_input_policy",
    ]
    predictions = frame[target_columns].copy()
    predictions.to_csv(output_root / "predictions.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    pd.DataFrame(candidate_rows).to_csv(output_root / "validation_candidates.csv", index=False)

    metric_rows = []
    comparisons = []
    for split, part in predictions.groupby("split", sort=False):
        metric_rows.append({"split": split, "model": "LightGBM", **metrics(part.actual_smp.to_numpy(float), part.predicted_smp.to_numpy(float)), "forecast_coverage_hours": int(len(part))})
        for baseline in ("baseline_recent_daily", "baseline_weekly"):
            valid = part[baseline].notna() & part.actual_smp.notna()
            actual = part.loc[valid, "actual_smp"].to_numpy(float)
            base_pred = part.loc[valid, baseline].to_numpy(float)
            model_pred = part.loc[valid, "predicted_smp"].to_numpy(float)
            base_score = metrics(actual, base_pred) if valid.any() else {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "bias_pred_minus_actual": np.nan}
            model_score = metrics(actual, model_pred) if valid.any() else base_score
            improvement = (base_score["mae"] - model_score["mae"]) / base_score["mae"] * 100 if np.isfinite(base_score["mae"]) and base_score["mae"] else np.nan
            metric_rows.append({"split": split, "model": baseline, **base_score, "forecast_coverage_hours": int(valid.sum()), "lightgbm_mae_on_same_rows": model_score["mae"], "lightgbm_improvement_pct": improvement})
            comparisons.append({"split": split, "baseline": baseline, "n_shared_rows": int(valid.sum()), "lightgbm_mae": model_score["mae"], "baseline_mae": base_score["mae"], "improvement_pct": improvement})
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(output_root / "overall_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output_root / "baseline_comparisons.csv", index=False)

    predictions["target_timestamp"] = pd.to_datetime(predictions["target_timestamp"])
    by_day = predictions.groupby(["split", "forecast_day"], as_index=False).agg(n=("absolute_error", "size"), mae=("absolute_error", "mean"), rmse=("absolute_error", lambda x: float(np.sqrt(np.mean(np.square(x))))) )
    by_horizon = predictions.groupby(["split", "horizon"], as_index=False).agg(n=("absolute_error", "size"), mae=("absolute_error", "mean"))
    by_month = predictions.assign(target_month=predictions.target_timestamp.dt.to_period("M").astype(str)).groupby(["split", "target_month", "forecast_day"], as_index=False).agg(n=("absolute_error", "size"), mae=("absolute_error", "mean"))
    by_origin = predictions.groupby(["split", "forecast_origin"], as_index=False).agg(n=("absolute_error", "size"), mae=("absolute_error", "mean"), rmse=("absolute_error", lambda x: float(np.sqrt(np.mean(np.square(x))))) )
    by_day.to_csv(output_root / "metrics_by_forecast_day.csv", index=False)
    by_horizon.to_csv(output_root / "metrics_by_horizon.csv", index=False)
    by_month.to_csv(output_root / "metrics_by_target_month.csv", index=False)
    by_origin.to_csv(output_root / "metrics_by_origin.csv", index=False)

    final_model.booster_.save_model(str(output_root / "lightgbm_72h.txt"))
    importance = pd.DataFrame({
        "feature": features,
        "gain_importance": final_model.booster_.feature_importance(importance_type="gain"),
        "split_importance": final_model.booster_.feature_importance(importance_type="split"),
    }).sort_values("gain_importance", ascending=False)
    importance.to_csv(output_root / "feature_importance.csv", index=False)
    plot_outputs(
        predictions.loc[predictions["split"].isin(["validation", "final_test"])],
        metric_frame,
        output_root,
    )

    summary = {
        "dataset_version": version,
        "model": "LightGBM",
        "status": "trained_and_scored" if gate.get("data_gate_passed") else "trained_and_scored_conditionally",
        "data_gate_passed": bool(gate.get("data_gate_passed")),
        "github_upload_ready": bool(gate.get("github_upload_ready")),
        "gate_report": str(gate_path.resolve()),
        "shared_dataset_manifest_model_target": manifest.get("model_target"),
        "shared_quality_checks": checks,
        "selected_config": best["config"],
        "selected_n_estimators": best["best_iteration"],
        "validation_mae": best["validation_mae"],
        "feature_count": len(features),
        "features": features,
        "feature_nan_cells_before_fit": missing_values,
        "lightgbm_native_missing_handling": True,
        "seed": SEED,
        "library_versions": {"lightgbm": lgb.__version__, "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "fit_and_score_seconds": time.perf_counter() - started,
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "final_test_rows": int(len(test)),
        "final_test_used_for_tuning": False,
        "conditional_input_policy": manifest.get("conditional_input_policy"),
    }
    (output_root / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--version-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--allow-conditional", action="store_true", help="Train a clearly marked research run while a source-scope gate remains unresolved.")
    args = parser.parse_args()
    version_root = (args.version_root or SHARED_ROOT / "versions" / args.version).resolve()
    output_root = (args.output_root or PROJECT_ROOT / "LightGBM_72H" / "outputs" / "runs" / f"{args.version}_lightgbm").resolve()
    result = run(args.version, version_root, output_root, args.gate_report.resolve(), args.allow_conditional)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
