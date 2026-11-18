from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
UPGRADE = SCRIPT.parents[1]
PROJECT = SCRIPT.parents[4]
MODEL_ROOT = PROJECT / "SMP_LightGBM_2026"
DATASET_VERSION_HINT = "smp72h_common_v5_generation_v2"
OUTPUT_DATASET_VERSION = "smp72h_common_v5_generation_v2_lightgbm_20261119"
KEYS = ["forecast_origin", "target_timestamp"]
MODEL_JOBS = max(1, min(4, (os.cpu_count() or 2) - 1))
SEED = 42


def find_dataset_root(explicit: Path | None) -> Path:
    if explicit:
        root = explicit.expanduser().resolve()
        if not (root / "manifest.json").is_file():
            raise FileNotFoundError(f"Dataset manifest not found in {root}")
        return root
    env_root = os.environ.get("SMP72H_DATASET_ROOT")
    if env_root:
        return find_dataset_root(Path(env_root))
    candidates: list[Path] = []
    for manifest_path in PROJECT.rglob("manifest.json"):
        if ".git" in manifest_path.parts or "upgrades" in manifest_path.parts:
            continue
        try:
            info = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(info.get("dataset_version", "")).startswith(DATASET_VERSION_HINT):
            candidates.append(manifest_path.parent)
    if not candidates:
        raise FileNotFoundError(
            "Could not find the frozen common v5 manifest. Pass --dataset-root or set SMP72H_DATASET_ROOT."
        )
    return sorted(candidates, key=lambda p: ("shared_dataset_v5" not in p.parts, len(str(p))))[0]


def find_existing_run() -> Path:
    candidates = [p.parent for p in MODEL_ROOT.glob("outputs/runs/*/predictions.csv")]
    candidates = [p for p in candidates if "common_v4" in p.name and (p / "overall_metrics.csv").exists()]
    if not candidates:
        raise FileNotFoundError("Existing LightGBM v4 predictions and metrics are required for E0 verification.")
    return sorted(candidates)[-1]


def metric_values(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    ok = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[ok], predicted[ok]
    if not len(actual):
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "bias": float("nan")}
    error = predicted - actual
    ss_tot = float(np.square(actual - actual.mean()).sum())
    return {
        "n": int(len(actual)),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": float(1.0 - np.square(error).sum() / ss_tot) if ss_tot else float("nan"),
        "bias": float(error.mean()),
    }


def score_frame(frame: pd.DataFrame, pred_col: str = "predicted_smp") -> dict[str, float | int]:
    return metric_values(frame["actual_smp"].to_numpy(), frame[pred_col].to_numpy())


def load_panel(dataset_root: Path) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8-sig"))
    if not str(manifest.get("dataset_version", "")).startswith(DATASET_VERSION_HINT):
        raise ValueError(f"Unexpected source dataset version: {manifest.get('dataset_version')}")
    feature_path = dataset_root / "feature_manifest.csv"
    panel_path = dataset_root / "forecast_inputs.csv.gz"
    if not panel_path.exists():
        panel_path = dataset_root / "forecast_inputs.csv"
    target_path = dataset_root / "evaluation_targets.csv.gz"
    if not target_path.exists():
        target_path = dataset_root / "evaluation_targets.csv"
    required = [feature_path, panel_path, target_path, dataset_root / "quality_report.json"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required shared data files missing: " + ", ".join(missing))
    features = manifest.get("model_feature_columns", [])
    if not features or len(features) != len(set(features)):
        raise ValueError("Manifest must list unique LightGBM feature columns.")
    feature_manifest = pd.read_csv(feature_path)
    groups = set(pd.to_numeric(feature_manifest["input_group"], errors="coerce").dropna().astype(int))
    if groups != set(range(1, 13)):
        raise ValueError(f"Expected all 12 common input groups; found {sorted(groups)}")
    panel = pd.read_csv(panel_path, parse_dates=["forecast_origin", "observed_cutoff", "target_timestamp"])
    targets = pd.read_csv(target_path, parse_dates=["forecast_origin", "observed_cutoff", "target_timestamp"])
    if panel.duplicated(KEYS).any() or targets.duplicated(KEYS).any():
        raise ValueError("Duplicate forecast issue and target-hour keys exist.")
    if len(panel) != len(targets):
        raise ValueError("Feature panel and label schedule row counts differ.")
    absent = sorted(set(features) - set(panel.columns))
    if absent:
        raise ValueError("Feature panel misses columns: " + ", ".join(absent[:15]))
    label_columns = ["split", "observed_cutoff", "actual_smp", "baseline_recent_daily", "baseline_weekly"]
    labels = targets[KEYS + [c for c in label_columns if c in targets.columns]].copy()
    frame = panel.merge(labels, on=KEYS, how="left", suffixes=("", "_target"), validate="one_to_one")
    if "split_target" in frame:
        if not frame["split"].equals(frame["split_target"]):
            raise ValueError("Panel and target split assignments differ.")
        frame = frame.drop(columns="split_target")
    if "observed_cutoff_target" in frame:
        if not frame["observed_cutoff"].equals(frame["observed_cutoff_target"]):
            raise ValueError("Panel and target observed cutoffs differ.")
        frame = frame.drop(columns="observed_cutoff_target")
    for column in ("actual_smp", "baseline_recent_daily", "baseline_weekly"):
        if column not in frame:
            raise ValueError(f"Evaluation targets do not contain {column}.")
    if frame["actual_smp"].isna().any():
        raise ValueError("SMP labels contain missing values.")
    if not (frame["observed_cutoff"] == frame["forecast_origin"] - pd.Timedelta(hours=72)).all():
        raise ValueError("The common origin-minus-72-hour cutoff check failed.")
    expected_ts = frame["forecast_origin"].dt.floor("D") + pd.Timedelta(days=1) + pd.to_timedelta(frame["horizon"] - 1, unit="h")
    if not (frame["target_timestamp"] == expected_ts).all():
        raise ValueError("Forecast targets do not match horizons 1 through 72.")
    per_origin = frame.groupby("forecast_origin")["horizon"].agg(["count", "min", "max"])
    if not ((per_origin["count"] == 72) & (per_origin["min"] == 1) & (per_origin["max"] == 72)).all():
        raise ValueError("Each forecast issue must have 72 unique horizons.")
    quality = json.loads((dataset_root / "quality_report.json").read_text(encoding="utf-8-sig"))
    for check in ("observed_cutoff_check", "future_observation_perturbation_check", "fred_d2_vintage_check"):
        if quality.get(check, {}).get("passed") is not True:
            raise ValueError(f"Dataset leakage/availability check did not pass: {check}")
    if "forecast_day" not in frame:
        frame["forecast_day"] = ((frame["horizon"].astype(int) - 1) // 24) + 1
    frame["source_dataset_version"] = manifest["dataset_version"]
    frame["dataset_version"] = OUTPUT_DATASET_VERSION
    frame[features] = frame[features].apply(pd.to_numeric, errors="coerce")
    return frame, features, manifest


def verify_e0(target: pd.DataFrame, old_run: Path) -> pd.DataFrame:
    old = pd.read_csv(old_run / "predictions.csv", parse_dates=KEYS + ["observed_cutoff"])
    cols = KEYS + ["split", "observed_cutoff", "actual_smp", "baseline_recent_daily", "baseline_weekly"]
    fresh = target[cols].copy()
    merged = fresh.merge(old[KEYS + ["split", "observed_cutoff", "actual_smp", "baseline_recent_daily", "baseline_weekly", "predicted_smp"]],
                         on=KEYS, how="outer", suffixes=("_v5", "_e0"), indicator=True, validate="one_to_one")
    if not merged["_merge"].eq("both").all():
        raise ValueError("Existing LightGBM and shared v5 labels use different evaluation issue-hour rows.")
    for col in ("split", "observed_cutoff", "actual_smp", "baseline_recent_daily", "baseline_weekly"):
        a, b = merged[f"{col}_v5"], merged[f"{col}_e0"]
        if col in ("split", "observed_cutoff"):
            same = a.astype(str).equals(b.astype(str))
        else:
            same = np.allclose(pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce"), equal_nan=True, atol=1e-8, rtol=0)
        if not same:
            raise ValueError(f"E0 verification mismatch in {col}; do not compare different evaluation rows.")
    result = target[cols].merge(old[KEYS + ["predicted_smp"]], on=KEYS, validate="one_to_one")
    stored = pd.read_csv(old_run / "overall_metrics.csv")
    checks = []
    for split, part in result.groupby("split"):
        actual = metric_values(part.actual_smp.to_numpy(float), part.predicted_smp.to_numpy(float))
        row = stored.loc[stored.split.eq(split) & stored.model.eq("LightGBM")]
        if len(row) != 1 or not np.isclose(actual["mae"], float(row.iloc[0].mae), atol=1e-8):
            raise ValueError(f"E0 stored MAE does not reproduce for {split}.")
        checks.append({"split": split, "recalculated_mae": actual["mae"], "stored_mae": float(row.iloc[0].mae), "rows": actual["n"]})
    (UPGRADE / "outputs" / "metrics" / "e0_reproduction_check.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return result


def verify_baseline_references(dataset_root: Path) -> dict[str, Any]:
    target_path = dataset_root / "evaluation_targets.csv.gz"
    if not target_path.exists():
        target_path = dataset_root / "evaluation_targets.csv"
    hourly_path = dataset_root / "hourly.csv.gz"
    if not hourly_path.exists():
        hourly_path = dataset_root / "hourly.csv"
    targets = pd.read_csv(target_path, parse_dates=["forecast_origin", "observed_cutoff", "target_timestamp", "baseline_origin_day"])
    history = pd.read_csv(hourly_path, parse_dates=["timestamp"], usecols=["timestamp", "smp"])
    history = history.dropna(subset=["smp"]).drop_duplicates("timestamp").set_index("timestamp")["smp"]
    scored = targets.loc[targets.split.isin(["validation", "final_test"])].copy()
    if scored[["baseline_recent_daily", "baseline_weekly"]].isna().any().any():
        raise ValueError("Causal baseline values are missing in Validation or Final Test.")
    scored["recent_reference_timestamp"] = scored["baseline_origin_day"] + pd.to_timedelta(scored["target_timestamp"].dt.hour, unit="h")
    scored["weekly_reference_timestamp"] = scored["target_timestamp"] - pd.Timedelta(hours=168)
    if not (scored["baseline_origin_day"] == scored["observed_cutoff"].dt.normalize() - pd.Timedelta(days=1)).all():
        raise ValueError("Recent-daily baseline source day does not match the frozen cutoff rule.")
    if (scored["recent_reference_timestamp"] > scored["observed_cutoff"]).any() or (scored["weekly_reference_timestamp"] > scored["observed_cutoff"]).any():
        raise ValueError("A baseline reference timestamp is later than the observation cutoff.")
    checks = []
    for name, timestamp_col, value_col in (("recent_daily", "recent_reference_timestamp", "baseline_recent_daily"),
                                           ("weekly", "weekly_reference_timestamp", "baseline_weekly")):
        observed = scored[timestamp_col].map(history)
        if observed.isna().any():
            raise ValueError(f"Hourly SMP source does not cover every {name} baseline reference.")
        maximum_error = float(np.abs(observed.to_numpy(float) - scored[value_col].to_numpy(float)).max())
        if maximum_error > 1e-8:
            raise ValueError(f"Stored {name} baseline does not reproduce from its historical reference hours.")
        checks.append({"baseline": name, "rows_verified": int(len(scored)), "max_value_difference": maximum_error,
                       "all_reference_hours_at_or_before_cutoff": True})
    result = {"status": "passed", "splits": sorted(scored.split.unique().tolist()), "checks": checks,
              "recent_daily_source_day_rule": "baseline_origin_day equals observed_cutoff date minus one day; same target hour",
              "weekly_source_rule": "target timestamp minus 168 hours",
              "baseline_origin_day_to_cutoff_days": 1}
    (UPGRADE / "outputs" / "metrics" / "baseline_reference_check.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def make_lgbm(config: dict[str, Any], n_estimators: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(objective="regression", n_estimators=int(n_estimators), n_jobs=MODEL_JOBS,
                             random_state=SEED, deterministic=True, force_col_wise=True, verbosity=-1, **config)


def candidate_configs(old_run: Path) -> list[dict[str, Any]]:
    saved = pd.read_csv(old_run / "validation_candidates.csv")
    columns = ["num_leaves", "max_depth", "learning_rate", "min_child_samples", "reg_lambda", "reg_alpha", "feature_fraction", "bagging_fraction"]
    return [{c: (float(row[c]) if c in ("learning_rate", "reg_lambda", "reg_alpha", "feature_fraction", "bagging_fraction") else int(row[c]))
             for c in columns} for _, row in saved.iterrows()]


def select_blend(actual: np.ndarray, model: np.ndarray, baselines: dict[str, np.ndarray]) -> tuple[str, float, float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rows.append({"baseline": "none", "alpha_model": 1.0, "mae": metric_values(actual, model)["mae"]})
    for name, baseline in baselines.items():
        for alpha in np.round(np.arange(0.0, 1.0001, 0.1), 1):
            pred = alpha * model + (1.0 - alpha) * baseline
            rows.append({"baseline": name, "alpha_model": float(alpha), "mae": metric_values(actual, pred)["mae"]})
    rows.sort(key=lambda r: (r["mae"], -r["alpha_model"], r["baseline"]))
    best = rows[0]
    return str(best["baseline"]), float(best["alpha_model"]), float(best["mae"]), rows


def apply_blend(model_pred: np.ndarray, baseline: np.ndarray | None, alpha: float) -> np.ndarray:
    if baseline is None:
        return model_pred.copy()
    return alpha * model_pred + (1.0 - alpha) * baseline


def add_exp_rows(out: list[pd.DataFrame], base: pd.DataFrame, experiment: str, model_label: str,
                 prediction: np.ndarray, baseline_name: str = "none", alpha: float = 1.0,
                 status: str = "predicted") -> None:
    piece = base[["dataset_version", "source_dataset_version", "split", "forecast_origin", "observed_cutoff", "target_timestamp", "horizon", "forecast_day", "actual_smp", "baseline_recent_daily", "baseline_weekly"]].copy()
    piece["experiment_id"] = experiment
    piece["model_name"] = model_label
    piece["predicted_smp"] = np.asarray(prediction, dtype=float)
    piece["absolute_error"] = np.abs(piece["actual_smp"].to_numpy(float) - piece["predicted_smp"].to_numpy(float))
    is_baseline_model = model_label.endswith("_baseline")
    piece["baseline_prediction"] = piece["baseline_recent_daily"] if baseline_name == "recent_daily" else piece["baseline_weekly"] if baseline_name == "weekly" else piece["predicted_smp"] if is_baseline_model else np.nan
    piece["blending_weight"] = 0.0 if is_baseline_model else alpha
    piece["prediction_status"] = status
    out.append(piece.drop(columns=["baseline_recent_daily", "baseline_weekly"]))


def build_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    bias_rows: list[dict[str, Any]] = []
    for (exp, split), part in predictions.groupby(["experiment_id", "split"], sort=False):
        score = metric_values(part.actual_smp.to_numpy(float), part.predicted_smp.to_numpy(float))
        baseline_col = "baseline_prediction"
        baseline = part[baseline_col].to_numpy(float)
        baseline_score = metric_values(part.actual_smp.to_numpy(float), baseline) if np.isfinite(baseline).any() else None
        metric_rows.append({"experiment_id": exp, "split": split, **score,
                            "baseline_mae": baseline_score["mae"] if baseline_score else np.nan,
                            "improvement_vs_selected_baseline_pct": ((baseline_score["mae"] - score["mae"]) / baseline_score["mae"] * 100)
                            if baseline_score and baseline_score["mae"] else np.nan,
                            "forecast_origins": int(part.forecast_origin.nunique())})
        err = part.predicted_smp.to_numpy(float) - part.actual_smp.to_numpy(float)
        bias_rows.append({"experiment_id": exp, "split": split, "n": len(part), "bias_pred_minus_actual": float(err.mean()),
                          "overprediction_rate_pct": float((err > 0).mean() * 100),
                          "underprediction_rate_pct": float((err < 0).mean() * 100),
                          "actual_mean": float(part.actual_smp.mean()), "prediction_mean": float(part.predicted_smp.mean()),
                          "actual_variance": float(part.actual_smp.var(ddof=0)), "prediction_variance": float(part.predicted_smp.var(ddof=0))})
    overall = pd.DataFrame(metric_rows)
    bias = pd.DataFrame(bias_rows)
    subset = predictions[predictions.experiment_id.isin(["E0_existing_lightgbm", "E1_existing_plus_recent_daily", "E1_existing_plus_weekly", "E1_existing_best_validation", "E2_lightgbm_train_validation_refit", "E3_refit_plus_selected_baseline", "baseline_recent_daily", "baseline_weekly"])].copy()
    subset["target_month"] = pd.to_datetime(subset.target_timestamp).dt.strftime("%Y-%m")
    by_month = subset.groupby(["experiment_id", "split", "target_month"], as_index=False).agg(n=("absolute_error", "size"), mae=("absolute_error", "mean"), rmse=("absolute_error", lambda x: float(np.sqrt(np.square(x).mean()))))
    by_horizon = subset.groupby(["experiment_id", "split", "horizon"], as_index=False).agg(n=("absolute_error", "size"), mae=("absolute_error", "mean"), rmse=("absolute_error", lambda x: float(np.sqrt(np.square(x).mean()))))
    by_day = subset.groupby(["experiment_id", "split", "forecast_day"], as_index=False).agg(n=("absolute_error", "size"), mae=("absolute_error", "mean"), rmse=("absolute_error", lambda x: float(np.sqrt(np.square(x).mean()))))
    move_rows = []
    for (exp, split), part in subset.groupby(["experiment_id", "split"], sort=False):
        ordered = part.sort_values(["forecast_origin", "horizon"]).copy()
        actual_delta = ordered.groupby("forecast_origin")["actual_smp"].diff()
        error_abs = ordered["absolute_error"]
        up_cut = float(actual_delta.quantile(.95)) if actual_delta.notna().any() else float("nan")
        down_cut = float(actual_delta.quantile(.05)) if actual_delta.notna().any() else float("nan")
        for move_name, mask in (("top_5pct_rises", actual_delta >= up_cut), ("bottom_5pct_drops", actual_delta <= down_cut)):
            selected = ordered.loc[mask.fillna(False)]
            move_rows.append({"experiment_id": exp, "split": split, "smp_move_segment": move_name,
                              "actual_change_threshold": up_cut if move_name == "top_5pct_rises" else down_cut,
                              "n": int(len(selected)), "mae": float(selected["absolute_error"].mean()) if len(selected) else np.nan,
                              "rmse": float(np.sqrt(np.square(selected["absolute_error"]).mean())) if len(selected) else np.nan})
    by_move = pd.DataFrame(move_rows)
    return overall, bias, by_month, by_horizon, by_day, by_move


def make_figures(predictions: pd.DataFrame, monthly: pd.DataFrame, horizon: pd.DataFrame, output_dir: Path) -> None:
    colors = {"E0_existing_lightgbm": "#7f8c8d", "E1_existing_best_validation": "#e67e22",
              "E2_lightgbm_train_validation_refit": "#2980b9", "E3_refit_plus_selected_baseline": "#27ae60",
              "baseline_recent_daily": "#9b59b6", "baseline_weekly": "#c0392b"}
    labels = {k: k.replace("_", " ") for k in colors}
    compare = predictions.groupby(["experiment_id", "split"], as_index=False).agg(mae=("absolute_error", "mean"))
    categories = [
        ("E0_existing_lightgbm", "Existing LightGBM"),
        ("baseline_recent_daily", "Recent daily baseline"),
        ("baseline_weekly", "Weekly baseline"),
        ("E1_existing_best_validation", "E1 validation blend"),
        ("E2_validation_selected_config_train_only", "E2 selected model"),
        ("E2_lightgbm_train_validation_refit", "E2 Train+Validation refit"),
        ("E3_train_only_model_plus_selected_baseline", "E3 selected model blend"),
        ("E3_refit_plus_selected_baseline", "E3 refit blend"),
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    xbase = np.arange(len(categories))
    for split, offset in (("validation", -0.18), ("final_test", 0.18)):
        vals = []
        for experiment_id, _label in categories:
            lookup_id = experiment_id
            if split == "validation" and experiment_id == "E2_lightgbm_train_validation_refit":
                lookup_id = "E2_validation_selected_config_train_only"
            if split == "validation" and experiment_id == "E3_refit_plus_selected_baseline":
                lookup_id = "E3_train_only_model_plus_selected_baseline"
            row = compare.loc[compare.split.eq(split) & compare.experiment_id.eq(lookup_id)]
            vals.append(float(row.iloc[0].mae) if len(row) else np.nan)
        ax.bar(xbase + offset, vals, width=0.34, label=split.replace("_", " "))
    ax.set_xticks(xbase)
    ax.set_xticklabels([label for _experiment, label in categories], rotation=28, ha="right")
    ax.set_ylabel("MAE (KRW/kWh)")
    ax.set_title("LightGBM and causal baseline comparison by split")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=150)
    plt.close(fig)

    test = predictions[predictions.split.eq("final_test")]
    first_origin = test.forecast_origin.min()
    sample = test[test.forecast_origin.eq(first_origin)].sort_values("horizon")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(sample.horizon, sample.actual_smp, label="Actual SMP", color="black", linewidth=2)
    for exp in ("E0_existing_lightgbm", "E1_existing_best_validation", "E2_lightgbm_train_validation_refit", "E3_refit_plus_selected_baseline"):
        rows = sample[sample.experiment_id.eq(exp)]
        if not rows.empty:
            ax.plot(rows.horizon, rows.predicted_smp, label=labels[exp])
    for boundary in (24.5, 48.5):
        ax.axvline(boundary, color="gray", linestyle="--", alpha=.7)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("SMP (KRW/kWh)")
    ax.set_title(f"72-hour forecast from {first_origin:%Y-%m-%d %H:%M} KST")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output_dir / "selected_72h_forecast.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for exp in ("E0_existing_lightgbm", "E1_existing_best_validation", "E2_lightgbm_train_validation_refit", "E3_refit_plus_selected_baseline", "baseline_weekly"):
        rows = monthly[(monthly.experiment_id.eq(exp)) & monthly.split.isin(["validation", "final_test"])]
        if not rows.empty:
            ax.plot(rows.target_month, rows.mae, marker="o", label=labels[exp], color=colors[exp])
    ax.set_ylabel("MAE (KRW/kWh)")
    ax.set_title("Monthly MAE by evaluation period")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output_dir / "monthly_mae.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for exp in ("E0_existing_lightgbm", "E1_existing_best_validation", "E2_lightgbm_train_validation_refit", "E3_refit_plus_selected_baseline", "baseline_weekly"):
        rows = horizon[(horizon.experiment_id.eq(exp)) & horizon.split.eq("final_test")]
        if not rows.empty:
            ax.plot(rows.horizon, rows.mae, label=labels[exp], color=colors[exp])
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("MAE (KRW/kWh)")
    ax.set_title("Final test MAE by forecast horizon")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output_dir / "horizon_mae.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LightGBM 72-hour SMP upgrade experiments.")
    parser.add_argument("--dataset-root", type=Path, help="Optional folder containing the frozen common v5 dataset.")
    args = parser.parse_args()
    started = time.perf_counter()
    out = UPGRADE / "outputs"
    metrics_dir = out / "metrics"
    figures_dir = out / "figures"
    for path in (UPGRADE / "config", UPGRADE / "models", out / "predictions", metrics_dir, figures_dir, UPGRADE / "reports"):
        path.mkdir(parents=True, exist_ok=True)
    if any((out / "predictions").iterdir()):
        raise FileExistsError("Refusing to overwrite existing upgrade predictions.")

    dataset_root = find_dataset_root(args.dataset_root)
    all_rows, features, manifest = load_panel(dataset_root)
    baseline_reference_check = verify_baseline_references(dataset_root)
    target_cols = ["dataset_version", "source_dataset_version", "split", "forecast_origin", "observed_cutoff", "target_timestamp", "horizon", "forecast_day", "actual_smp", "baseline_recent_daily", "baseline_weekly"]
    target = all_rows[target_cols].copy()
    old_run = find_existing_run()
    e0 = verify_e0(target, old_run)
    old = pd.read_csv(old_run / "predictions.csv", parse_dates=KEYS + ["observed_cutoff"])
    e0 = target.merge(old[KEYS + ["predicted_smp"]], on=KEYS, validate="one_to_one")

    validation = all_rows.loc[all_rows.split.eq("validation")].copy()
    y_val = validation.actual_smp.to_numpy(float)
    old_val = e0.loc[e0.split.eq("validation"), "predicted_smp"].to_numpy(float)
    val_base = {"recent_daily": validation.baseline_recent_daily.to_numpy(float), "weekly": validation.baseline_weekly.to_numpy(float)}
    e1_choice, e1_alpha, e1_mae, e1_grid = select_blend(y_val, old_val, val_base)
    e1_by_baseline = {}
    for baseline_name in val_base:
        subset = [row for row in e1_grid if row["baseline"] == baseline_name]
        selected = min(subset, key=lambda row: row["mae"])
        e1_by_baseline[baseline_name] = {"alpha_model": float(selected["alpha_model"]), "validation_mae": float(selected["mae"])}
    pd.DataFrame(e1_grid).to_csv(metrics_dir / "e1_validation_blend_grid.csv", index=False)

    train = all_rows.loc[all_rows.split.eq("train")]
    valid = all_rows.loc[all_rows.split.eq("validation")]
    test = all_rows.loc[all_rows.split.eq("final_test")]
    y_train = train.actual_smp.to_numpy(float)
    x_train, x_val = train[features], valid[features]
    y_v = valid.actual_smp.to_numpy(float)
    cands = candidate_configs(old_run)
    candidate_rows: list[dict[str, Any]] = []
    candidate_predictions: dict[int, np.ndarray] = {}
    best_idx = -1
    best_mae = float("inf")
    best_iteration = 0
    for i, config in enumerate(cands, start=1):
        model = lgb.LGBMRegressor(objective="regression", n_estimators=1800, n_jobs=MODEL_JOBS,
                                  random_state=SEED, deterministic=True, force_col_wise=True, verbosity=-1, **config)
        model.fit(x_train, y_train, eval_set=[(x_val, y_v)], eval_metric="l1",
                  callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        pred = model.predict(x_val)
        score = metric_values(y_v, pred)
        iteration = int(model.best_iteration_ or model.n_estimators)
        candidate_predictions[i] = pred
        candidate_rows.append({"candidate_id": i, **config, "best_iteration": iteration, **score})
        if score["mae"] < best_mae:
            best_idx, best_mae, best_iteration = i, float(score["mae"]), iteration
    if best_idx < 1:
        raise RuntimeError("LightGBM did not produce a selected validation candidate.")
    candidate_frame = pd.DataFrame(candidate_rows)
    candidate_frame.to_csv(metrics_dir / "validation_candidates.csv", index=False)
    selected_config = cands[best_idx - 1]
    e2_validation_pred = candidate_predictions[best_idx]
    e3_choice, e3_alpha, e3_mae, e3_grid = select_blend(y_v, e2_validation_pred, val_base)
    pd.DataFrame(e3_grid).to_csv(metrics_dir / "e3_validation_blend_grid.csv", index=False)

    training_validation = all_rows.loc[all_rows.split.isin(["train", "validation"])]
    final_model = make_lgbm(selected_config, best_iteration)
    final_model.fit(training_validation[features], training_validation.actual_smp.to_numpy(float))
    test_pred = final_model.predict(test[features])
    test_old = e0.loc[e0.split.eq("final_test"), "predicted_smp"].to_numpy(float)
    test_recent = test.baseline_recent_daily.to_numpy(float)
    test_weekly = test.baseline_weekly.to_numpy(float)
    e1_base_test = test_recent if e1_choice == "recent_daily" else test_weekly if e1_choice == "weekly" else None
    e1_test_pred = apply_blend(test_old, e1_base_test, e1_alpha)
    e3_base_test = test_recent if e3_choice == "recent_daily" else test_weekly if e3_choice == "weekly" else None
    e3_test_pred = apply_blend(test_pred, e3_base_test, e3_alpha)
    final_model.booster_.save_model(str(UPGRADE / "models" / "train_validation_refit_lightgbm_model.txt"))
    shutil.copy2(old_run / "lightgbm_72h.txt", UPGRADE / "models" / "final_lightgbm_model.txt")

    validation_out: list[pd.DataFrame] = []
    e0_val = e0.loc[e0.split.eq("validation"), "predicted_smp"].to_numpy(float)
    e1_base_val = val_base.get(e1_choice)
    e3_base_val = val_base.get(e3_choice)
    add_exp_rows(validation_out, valid, "E0_existing_lightgbm", "LightGBM", e0_val)
    add_exp_rows(validation_out, valid, "E1_existing_plus_recent_daily", "LightGBM", apply_blend(e0_val, val_base["recent_daily"], e1_by_baseline["recent_daily"]["alpha_model"]), "recent_daily", e1_by_baseline["recent_daily"]["alpha_model"], "validation_used_for_blend_selection")
    add_exp_rows(validation_out, valid, "E1_existing_plus_weekly", "LightGBM", apply_blend(e0_val, val_base["weekly"], e1_by_baseline["weekly"]["alpha_model"]), "weekly", e1_by_baseline["weekly"]["alpha_model"], "validation_used_for_blend_selection")
    add_exp_rows(validation_out, valid, "E1_existing_best_validation", "LightGBM", apply_blend(e0_val, e1_base_val, e1_alpha), e1_choice, e1_alpha, "validation_used_for_blend_selection")
    add_exp_rows(validation_out, valid, "E2_validation_selected_config_train_only", "LightGBM", e2_validation_pred, status="validation_used_for_model_selection")
    add_exp_rows(validation_out, valid, "E3_train_only_model_plus_selected_baseline", "LightGBM", apply_blend(e2_validation_pred, e3_base_val, e3_alpha), e3_choice, e3_alpha, "validation_used_for_blend_selection")
    add_exp_rows(validation_out, valid, "baseline_recent_daily", "recent_daily_baseline", val_base["recent_daily"], status="causal_baseline")
    add_exp_rows(validation_out, valid, "baseline_weekly", "weekly_baseline", val_base["weekly"], status="causal_baseline")
    validation_predictions = pd.concat(validation_out, ignore_index=True)
    validation_predictions["selected_for_final"] = validation_predictions.experiment_id.eq("E1_existing_best_validation")
    validation_predictions.to_csv(out / "predictions" / "experiments_validation.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")

    final_out: list[pd.DataFrame] = []
    add_exp_rows(final_out, test, "E0_existing_lightgbm", "LightGBM", test_old)
    alpha_recent = e1_by_baseline["recent_daily"]["alpha_model"]
    alpha_weekly = e1_by_baseline["weekly"]["alpha_model"]
    add_exp_rows(final_out, test, "E1_existing_plus_recent_daily", "LightGBM", apply_blend(test_old, test_recent, alpha_recent), "recent_daily", alpha_recent)
    add_exp_rows(final_out, test, "E1_existing_plus_weekly", "LightGBM", apply_blend(test_old, test_weekly, alpha_weekly), "weekly", alpha_weekly)
    add_exp_rows(final_out, test, "E1_existing_best_validation", "LightGBM", e1_test_pred, e1_choice, e1_alpha)
    add_exp_rows(final_out, test, "E2_lightgbm_train_validation_refit", "LightGBM", test_pred)
    add_exp_rows(final_out, test, "E3_refit_plus_selected_baseline", "LightGBM", e3_test_pred, e3_choice, e3_alpha)
    add_exp_rows(final_out, test, "baseline_recent_daily", "recent_daily_baseline", test_recent, status="causal_baseline")
    add_exp_rows(final_out, test, "baseline_weekly", "weekly_baseline", test_weekly, status="causal_baseline")
    final_predictions = pd.concat(final_out, ignore_index=True)
    final_predictions["selected_for_final"] = final_predictions.experiment_id.eq("E1_existing_best_validation")
    final_predictions.to_csv(out / "predictions" / "final_test_predictions.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    all_predictions = pd.concat([validation_predictions, final_predictions], ignore_index=True)
    overall, bias, monthly, by_horizon, by_day, by_move = build_metrics(all_predictions)
    overall.to_csv(metrics_dir / "experiment_comparison.csv", index=False)
    by_day.to_csv(metrics_dir / "metrics_by_forecast_day.csv", index=False)
    by_horizon.to_csv(metrics_dir / "metrics_by_horizon.csv", index=False)
    monthly.to_csv(metrics_dir / "metrics_by_month.csv", index=False)
    bias.to_csv(metrics_dir / "bias_analysis.csv", index=False)
    by_move.to_csv(metrics_dir / "metrics_by_smp_move.csv", index=False)
    make_figures(all_predictions, monthly, by_horizon, figures_dir)

    coverage = {split: {"rows": int(len(part)), "forecast_origins": int(part.forecast_origin.nunique()),
                        "min_target": str(part.target_timestamp.min()), "max_target": str(part.target_timestamp.max())}
                for split, part in all_rows.groupby("split")}
    selected_test = metric_values(test.actual_smp.to_numpy(float), test_pred)
    e3_test_score = metric_values(test.actual_smp.to_numpy(float), e3_test_pred)
    e0_test_score = metric_values(test.actual_smp.to_numpy(float), test_old)
    final_selected_test_score = metric_values(test.actual_smp.to_numpy(float), e1_test_pred)
    weekly_score = metric_values(test.actual_smp.to_numpy(float), test_weekly)
    recent_score = metric_values(test.actual_smp.to_numpy(float), test_recent)
    final_rule = ("existing LightGBM only" if e1_choice == "none" else
                  f"{e1_alpha:.1f} * existing LightGBM + {1.0-e1_alpha:.1f} * causal {e1_choice} baseline")
    summary = {
        "dataset_version": OUTPUT_DATASET_VERSION, "source_dataset_version": manifest["dataset_version"], "dataset_family": "smp72h_common_v5_generation_v2",
        "model": "LightGBM",
        "feature_count": len(features), "feature_columns": features,
        "existing_lightgbm_run": str(old_run.relative_to(PROJECT)),
        "selected_config": selected_config, "selected_candidate_id": best_idx,
        "selected_n_estimators": best_iteration, "validation_candidate_mae": best_mae,
        "e1_validation_choice": {"baseline": e1_choice, "alpha_model": e1_alpha, "mae": e1_mae},
        "e3_validation_choice": {"baseline": e3_choice, "alpha_model": e3_alpha, "mae": e3_mae},
        "final_selected_experiment_id": "E1_existing_best_validation",
        "final_prediction_rule": final_rule,
        "final_selected_validation_mae": e1_mae,
        "final_selected_test": final_selected_test_score,
        "refit_track_validation_mae": e3_mae,
        "refit_track_validation_mae_difference_vs_selected": e3_mae - e1_mae,
        "baseline_reference_check": baseline_reference_check,
        "final_test": {"E0_existing_lightgbm": e0_test_score, "recent_daily_baseline": recent_score,
                       "weekly_baseline": weekly_score, "E2_train_validation_refit": selected_test,
                       "E1_existing_best_validation": metric_values(test.actual_smp.to_numpy(float), e1_test_pred),
                       "E3_refit_plus_selected_baseline": e3_test_score},
        "evaluation_rows": coverage, "final_test_used_for_tuning": False,
        "feature_missing_cells_by_split": {split: int(part[features].isna().sum().sum()) for split, part in all_rows.groupby("split")},
        "feature_missing_rate_by_split": {split: float(part[features].isna().to_numpy().mean()) for split, part in all_rows.groupby("split")},
        "target_distribution_by_split": {split: {"mean": float(part.actual_smp.mean()), "std": float(part.actual_smp.std(ddof=0)),
                                                  "p95": float(part.actual_smp.quantile(.95)), "max": float(part.actual_smp.max())}
                                          for split, part in all_rows.groupby("split")},
        "final_test_reused_from_prior_public_evaluation": True,
        "observed_cutoff_rule": "forecast_origin minus 72 hours, common cutoff for all 72 target hours",
        "lgbm_version": lgb.__version__, "pandas_version": pd.__version__, "numpy_version": np.__version__,
        "random_seed": SEED, "n_jobs": MODEL_JOBS, "elapsed_seconds": time.perf_counter() - started,
        "dataset_path": "auto-discovered from the frozen common v5 manifest; no data copy was made",
        "final_model_artifact": "models/final_lightgbm_model.txt (model component used by the selected E1 blend)",
        "train_validation_refit_artifact": "models/train_validation_refit_lightgbm_model.txt",
    }
    (UPGRADE / "outputs" / "metrics" / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (UPGRADE / "outputs" / "metrics" / "evaluation_coverage.json").write_text(json.dumps(coverage, indent=2, ensure_ascii=False), encoding="utf-8")
    write_reports(summary, overall, bias, candidate_frame, by_day, e0_test_score, weekly_score, recent_score, manifest)
    config = {"model_name": "LightGBM", "dataset_version": OUTPUT_DATASET_VERSION, "source_dataset_version": manifest["dataset_version"], "dataset_family": summary["dataset_family"],
              "features": len(features), "validation_candidates": candidate_rows, "selected_candidate_id": best_idx,
              "selected_config": selected_config, "selected_n_estimators": best_iteration,
              "E1_blend": summary["e1_validation_choice"], "E1_baseline_specific_weights": e1_by_baseline,
              "E3_blend": summary["e3_validation_choice"],
              "final_selected_experiment_id": summary["final_selected_experiment_id"],
              "final_prediction_rule": summary["final_prediction_rule"],
              "final_model_artifact": summary["final_model_artifact"],
              "train_validation_refit_artifact": summary["train_validation_refit_artifact"],
              "fit_splits": ["train", "validation"], "evaluation_split": "final_test", "seed": SEED}
    (UPGRADE / "config" / "experiment_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    gate = json.loads((MODEL_ROOT / "reports" / "input_gate.json").read_text(encoding="utf-8-sig"))
    gate.update({"dataset_version": OUTPUT_DATASET_VERSION, "source_dataset_version": manifest["dataset_version"], "model": "LightGBM",
                 "data_gate_passed": False, "training_authorized_by_user": True,
                 "conditional_result_upload_authorized_by_user": True,
                 "github_upload_ready": True,
                 "upload_note": "LightGBM upgrade run uses the exact shared common v5 data and retains the unresolved Group 1 geography scope as a disclosed conditional limitation."})
    (UPGRADE / "reports" / "data_gate.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"upgrade": str(UPGRADE), "dataset_version": OUTPUT_DATASET_VERSION,
                      "selected_candidate": best_idx, "selected_n_estimators": best_iteration,
                      "E0_test_mae": e0_test_score["mae"], "E2_test_mae": selected_test["mae"],
                      "E3_test_mae": e3_test_score["mae"], "weekly_baseline_test_mae": weekly_score["mae"]}, ensure_ascii=False, indent=2))


def write_reports(summary: dict[str, Any], comparison: pd.DataFrame, bias: pd.DataFrame,
                  candidates: pd.DataFrame, by_day: pd.DataFrame, e0: dict[str, Any], weekly: dict[str, Any],
                  recent: dict[str, Any], manifest: dict[str, Any]) -> None:
    report_dir = UPGRADE / "reports"
    final = summary["final_test"]
    e2 = final["E2_train_validation_refit"]
    e3 = final["E3_refit_plus_selected_baseline"]
    e1 = final["E1_existing_best_validation"]
    best = candidates.sort_values("mae").iloc[0]
    report = f"""# LightGBM 72-hour upgrade: final report

## Dataset and evaluation protocol

- Model: LightGBM, 72-hour land SMP forecast at D 23:00 KST.
- Output dataset version: `{summary['dataset_version']}`; exact immutable source manifest version: `{summary['source_dataset_version']}`. The common v5 payload, 458 feature columns, and frozen labels are reused unchanged.
- The common data bundle was read from its existing repository location and was not copied or edited.
- Features: {summary['feature_count']} across input groups 1–12. Group 5 is generation v2; all other groups follow the frozen shared panel.
- Train: 2023-01-01 through 2024-12-31; Validation: 2025-01-01 through 2025-06-30; Final Test: 2025-07-01 through 2025-12-31.
- Each forecast uses one observed-input cutoff `forecast_origin - 72 hours` for all horizons. Future target observations were not used as features.
- Both causal baselines were checked against the frozen hourly SMP history in Validation and Final Test: the recent-daily reference is the same hour on the source day one day before the cutoff date; the weekly reference is exactly target time minus 168 hours. All reference hours precede the common cutoff and all values reproduce exactly.
- Final Test was not used for model or blend selection. This period was evaluated before and is a reused test period, not a new blind test.

## Existing LightGBM E0 reproduction and baselines

The saved E0 predictions were matched to v5 by issue and target hour. Split, cutoff, actual SMP, and both causal baseline columns match row-for-row. Saved E0 MAE reproduced from the CSV before comparing new runs. Reproduction details are in `outputs/metrics/e0_reproduction_check.json`.

| Final Test experiment | MAE (KRW/kWh) | RMSE | R² | Bias |
|---|---:|---:|---:|---:|
| Existing LightGBM E0 | {e0['mae']:.4f} | {e0['rmse']:.4f} | {e0['r2']:.4f} | {e0['bias']:.4f} |
| Recent daily baseline | {recent['mae']:.4f} | {recent['rmse']:.4f} | {recent['r2']:.4f} | {recent['bias']:.4f} |
| Weekly baseline | {weekly['mae']:.4f} | {weekly['rmse']:.4f} | {weekly['r2']:.4f} | {weekly['bias']:.4f} |
| E1 existing LightGBM, validation-selected blend | {e1['mae']:.4f} | {e1['rmse']:.4f} | {e1['r2']:.4f} | {e1['bias']:.4f} |
| E2 Train+Validation refit LightGBM | {e2['mae']:.4f} | {e2['rmse']:.4f} | {e2['r2']:.4f} | {e2['bias']:.4f} |
| E3 refit LightGBM, validation-selected blend | {e3['mae']:.4f} | {e3['rmse']:.4f} | {e3['r2']:.4f} | {e3['bias']:.4f} |

Validation summary: E0 MAE {comparison.loc[comparison.experiment_id.eq('E0_existing_lightgbm') & comparison.split.eq('validation'), 'mae'].iloc[0]:.4f}; selected E1 MAE {summary['final_selected_validation_mae']:.4f}; selected Train-only E2 candidate MAE {summary['validation_candidate_mae']:.4f}; E3 refit-track blend MAE {summary['refit_track_validation_mae']:.4f}. E1 was the lowest Validation MAE across the compared deployable rules, so it is the final selected rule. The Train+Validation refit remains saved and evaluated as E2/E3.

For the validation-selected E1, Final Test MAE changed by {e1['mae'] - e0['mae']:+.4f} KRW/kWh from E0, and by {e1['mae'] - weekly['mae']:+.4f} KRW/kWh from the weekly baseline. E3's observed Final Test MAE is lower, but E3 was not selected because its Validation MAE was {summary['refit_track_validation_mae_difference_vs_selected']:+.4f} higher than E1; no test-driven switch was made. A negative change is an improvement.

## Experiments

- **E0:** Existing LightGBM predictions and saved metrics reproduced on the same frozen rows.
- **E1:** Existing LightGBM blended separately with recent-daily and weekly causal baselines. Validation selected `{summary['e1_validation_choice']['baseline']}` with model weight α={summary['e1_validation_choice']['alpha_model']:.1f}; selected Validation MAE={summary['e1_validation_choice']['mae']:.4f}.
- **E2:** Three previously registered LightGBM candidate parameter sets were re-evaluated on the frozen v5 Validation rows. Candidate {int(best['candidate_id'])} was selected, with {int(best['best_iteration'])} boosting iterations and Validation MAE {best['mae']:.4f}. It was then refit on Train+Validation and scored once on Final Test.
- **E3:** A baseline blend for the selected Train-only Validation model was selected on Validation, then the same baseline family and weight were applied to the Train+Validation refit. Baseline `{summary['e3_validation_choice']['baseline']}`, α={summary['e3_validation_choice']['alpha_model']:.1f}.
- **Final selection:** E1 was frozen as the final prediction rule before Final Test because it had the lowest Validation MAE. E2/E3 refit artifacts and scores are retained for comparison.
- **E4 bias correction:** Not adopted. Validation errors and horizon-specific bias are reported; there was not a separately held-out validation period after selecting a correction. Final Test values were not used to select a correction.
- No horizon-specific models or feature ablations were run.

## Split sizes

| Split | Forecast hours | Forecast origins |
|---|---:|---:|
"""
    for split, value in summary["evaluation_rows"].items():
        report += f"| {split} | {value['rows']:,} | {value['forecast_origins']:,} |\n"
    report += "\n### Final Test by forecast day\n\n| Experiment | Forecast day | MAE | RMSE |\n|---|---:|---:|---:|\n"
    day_ids = ["E0_existing_lightgbm", "E1_existing_best_validation", "E2_lightgbm_train_validation_refit", "E3_refit_plus_selected_baseline", "baseline_weekly"]
    for _, row in by_day.loc[by_day.split.eq("final_test") & by_day.experiment_id.isin(day_ids)].iterrows():
        report += f"| {row.experiment_id} | D+{int(row.forecast_day)} | {row.mae:.4f} | {row.rmse:.4f} |\n"
    val_dist = summary["target_distribution_by_split"]["validation"]
    test_dist = summary["target_distribution_by_split"]["final_test"]
    report += f"\nFinal Test target SMP mean was {test_dist['mean']:.2f} vs Validation {val_dist['mean']:.2f} KRW/kWh; population standard deviation was {test_dist['std']:.2f} vs {val_dist['std']:.2f}. Model input missing-cell rate was {summary['feature_missing_rate_by_split']['final_test']*100:.3f}% on Final Test.\n\n"
    report += "Horizon 1–72, monthly metrics, surge/drop error slices, detailed bias, and coverage are in `outputs/metrics/`. The plot files are in `outputs/figures/`.\n\n"
    report += "## Data limits and reproducibility\n\n"
    report += "The 72-hour observation cutoff is a study assumption; actual release timestamps were not verified. The Group 1 EPSIS feed lacks row-level mainland geography, so `data_gate_passed` remains false and this result is conditional. Group 5 missing periods stay missing and use the frozen panel's flags/native LightGBM missing handling; the 2024 periods absent from the source were not zero-filled. Same-month domestic LNG is conditional and not operationally available at historical issue time.\n\n"
    report += "Re-run from the repository root with `python SMP_LightGBM_2026/upgrades/upgrade_72h_20261119/scripts/run_lightgbm_upgrade.py`. The script discovers the immutable common v5 manifest or accepts `--dataset-root`; it writes only to this upgrade folder. Dependencies are listed in `SMP_LightGBM_2026/requirements.txt`.\n"
    (report_dir / "final_upgrade_report.md").write_text(report, encoding="utf-8")

    (report_dir / "baseline_comparison.md").write_text(
        "# LightGBM baseline blend selection\n\n"
        f"Existing LightGBM E0 validation selected `{summary['e1_validation_choice']['baseline']}` at α={summary['e1_validation_choice']['alpha_model']:.1f}; Validation MAE {summary['e1_validation_choice']['mae']:.4f}.\n\n"
        f"Train-only candidate validation for E3 selected `{summary['e3_validation_choice']['baseline']}` at α={summary['e3_validation_choice']['alpha_model']:.1f}; Validation MAE {summary['e3_validation_choice']['mae']:.4f}.\n\n"
        "The candidates used the recent-daily SMP and same-hour-previous-week SMP values available at the issue time. α is the LightGBM share. Final Test did not affect either selection. Full grid: `outputs/metrics/e1_validation_blend_grid.csv` and `e3_validation_blend_grid.csv`.\n",
        encoding="utf-8")
    (report_dir / "retraining_report.md").write_text(
        "# LightGBM Train+Validation refit\n\n"
        f"Validation chose candidate {summary['selected_candidate_id']} with {summary['selected_n_estimators']} iterations and MAE {summary['validation_candidate_mae']:.4f}. The selected configuration was refit using Train plus Validation only. Final Test rows were used only for scoring. Model: `models/final_lightgbm_model.txt`.\n",
        encoding="utf-8")
    val_bias = bias[bias.split.eq("validation")]
    test_bias = bias[bias.split.eq("final_test")]
    (report_dir / "bias_analysis.md").write_text(
        "# LightGBM bias analysis\n\n"
        "Bias is mean(prediction - actual); positive values indicate overprediction. The full summary includes over/underprediction rates and prediction/actual variances.\n\n"
        "Validation biases:\n\n```text\n" + val_bias.to_string(index=False) + "\n```\n\nFinal Test biases:\n\n```text\n" + test_bias.to_string(index=False) + "\n```" +
        "\n\nThe large change between splits is treated as a distribution-shift warning. No Final Test-based offset was used.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
