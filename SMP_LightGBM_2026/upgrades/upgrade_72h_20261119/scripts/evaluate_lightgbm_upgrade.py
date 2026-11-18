from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


UPGRADE = Path(__file__).resolve().parents[1]
OUTPUTS = UPGRADE / "outputs"
REQUIRED = {
    "dataset_version", "source_dataset_version", "experiment_id", "model_name", "split", "forecast_origin",
    "observed_cutoff", "target_timestamp", "horizon", "forecast_day", "actual_smp", "predicted_smp",
    "absolute_error", "baseline_prediction", "blending_weight", "prediction_status",
    "selected_for_final",
}


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - actual
    denom = np.square(actual - actual.mean()).sum()
    return {"n": int(len(actual)), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "r2": float(1 - np.square(error).sum() / denom) if denom else float("nan"),
            "bias": float(error.mean())}


def main() -> None:
    path = OUTPUTS / "predictions" / "final_test_predictions.csv"
    predictions = pd.read_csv(path, parse_dates=["forecast_origin", "observed_cutoff", "target_timestamp"])
    absent = sorted(REQUIRED - set(predictions.columns))
    if absent:
        raise ValueError(f"Prediction output misses required columns: {absent}")
    if not predictions["dataset_version"].eq("smp72h_common_v5_generation_v2_lightgbm_20261119").all():
        raise ValueError("Prediction output uses an unexpected LightGBM dataset alias.")
    if not (predictions["observed_cutoff"] == predictions["forecast_origin"] - pd.Timedelta(hours=72)).all():
        raise ValueError("Observed cutoff rule failed in output rows.")
    if not (predictions["target_timestamp"] == predictions["forecast_origin"].dt.floor("D") + pd.Timedelta(days=1) + pd.to_timedelta(predictions["horizon"] - 1, unit="h")).all():
        raise ValueError("Target timestamp and horizon alignment failed.")
    comparison = pd.read_csv(OUTPUTS / "metrics" / "experiment_comparison.csv")
    validation_comparison = comparison.loc[comparison.split.eq("validation")].sort_values(["mae", "experiment_id"])
    if validation_comparison.empty or validation_comparison.iloc[0].experiment_id != "E1_existing_best_validation":
        raise ValueError("Final rule does not have the lowest Validation MAE among recorded candidates.")
    baseline_check_path = OUTPUTS / "metrics" / "baseline_reference_check.json"
    if not baseline_check_path.is_file():
        raise FileNotFoundError("Baseline reference verification record is missing.")
    baseline_check = json.loads(baseline_check_path.read_text(encoding="utf-8"))
    if baseline_check.get("status") != "passed" or not all(item.get("all_reference_hours_at_or_before_cutoff") for item in baseline_check.get("checks", [])):
        raise ValueError("A causal baseline reference check did not pass.")
    recalculated = []
    selected_mask = predictions["selected_for_final"].astype(str).str.lower().eq("true")
    if int(selected_mask.sum()) != 13104 or predictions.loc[selected_mask, "experiment_id"].nunique() != 1 or not predictions.loc[selected_mask, "experiment_id"].eq("E1_existing_best_validation").all():
        raise ValueError("The final selected prediction rule is not the Validation-selected E1 blend.")
    for experiment, part in predictions.groupby("experiment_id", sort=False):
        by_origin = part.groupby("forecast_origin")["horizon"].agg(["count", "min", "max"])
        if not ((by_origin["count"] == 72) & (by_origin["min"] == 1) & (by_origin["max"] == 72)).all():
            raise ValueError(f"Incomplete 72-hour origin found in {experiment}.")
        score = metrics(part.actual_smp.to_numpy(), part.predicted_smp.to_numpy())
        stored = comparison.loc[comparison.experiment_id.eq(experiment) & comparison.split.eq("final_test")]
        if len(stored) != 1 or not np.isclose(score["mae"], float(stored.iloc[0].mae), atol=1e-9):
            raise ValueError(f"Final-test metric does not reproduce for {experiment}.")
        recalculated.append({"experiment_id": experiment, **score,
                             "forecast_origins": int(part.forecast_origin.nunique()),
                             "observed_cutoff_passed": True, "72h_horizon_coverage_passed": True})
    model_files = [UPGRADE / "models" / "final_lightgbm_model.txt", UPGRADE / "models" / "train_validation_refit_lightgbm_model.txt"]
    if any(not p.is_file() or p.stat().st_size == 0 for p in model_files):
        raise FileNotFoundError("Final LightGBM or Train+Validation refit model is absent or empty.")
    result = {"status": "passed", "final_selected_experiment_id": "E1_existing_best_validation", "lowest_validation_mae": float(validation_comparison.iloc[0].mae), "checks": ["required prediction fields", "dataset identity", "common 72-hour cutoff", "target-horizon alignment", "causal baseline source-hour verification", "lowest Validation MAE selection", "selected-final output identity", "72 horizons per origin", "final-test MAE recomputation", "saved final and refit LightGBM models"], "experiments": recalculated}
    destination = OUTPUTS / "metrics" / "artifact_verification.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
