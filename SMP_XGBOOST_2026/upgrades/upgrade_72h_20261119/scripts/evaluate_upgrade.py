"""Evaluate locked XGBoost upgrade predictions without changing selections."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


UPGRADE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = UPGRADE_DIR.parents[1]
DATASET_VERSION = "smp72h_common_v5_generation_v2_xgb_20261119"
DATASET_DIR = MODEL_DIR.parent / "SMP_SHARED_DATA" / "versions" / DATASET_VERSION
BLEND_ALPHAS = [round(i / 10, 1) for i in range(11)]
METRICS_DIR = UPGRADE_DIR / "outputs" / "metrics"
PREDICTIONS_DIR = UPGRADE_DIR / "outputs" / "predictions"
FIGURES_DIR = UPGRADE_DIR / "outputs" / "figures"
REPORTS_DIR = UPGRADE_DIR / "reports"


def metric_values(actual: pd.Series, predicted: pd.Series) -> dict:
    y = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(predicted, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if not len(y):
        return {"n_rows": 0, "coverage": 0.0, "mae": np.nan, "rmse": np.nan, "r2": np.nan,
                "bias_pred_minus_actual": np.nan, "overprediction_share": np.nan,
                "underprediction_share": np.nan, "actual_std": np.nan, "predicted_std": np.nan,
                "std_ratio_pred_to_actual": np.nan}
    actual_std = float(np.std(y))
    predicted_std = float(np.std(p))
    return {
        "n_rows": int(len(y)), "coverage": float(len(y) / len(actual)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)) if len(y) > 1 else np.nan,
        "bias_pred_minus_actual": float(np.mean(p - y)),
        "overprediction_share": float(np.mean(p > y)),
        "underprediction_share": float(np.mean(p < y)),
        "actual_std": actual_std, "predicted_std": predicted_std,
        "std_ratio_pred_to_actual": predicted_std / actual_std if actual_std else np.nan,
    }


def select_blend_weights(validation: pd.DataFrame) -> dict:
    """Select global weights on Validation and record walk-forward stability."""
    candidates = []
    per_day = []
    per_baseline = {}
    baseline_columns = {"recent_daily": "baseline_recent_daily", "weekly": "baseline_weekly"}

    def best_for(frame: pd.DataFrame, baseline_col: str) -> tuple[float, float]:
        scored = []
        for alpha in BLEND_ALPHAS:
            prediction = alpha * frame["predicted_e0"].to_numpy(dtype=float) + (1 - alpha) * frame[baseline_col].to_numpy(dtype=float)
            mae = metric_values(frame["actual_smp"], pd.Series(prediction, index=frame.index))["mae"]
            scored.append((float(mae), -alpha, alpha))
        mae, _, alpha = min(scored)
        return float(alpha), float(mae)

    for baseline_name, baseline_col in baseline_columns.items():
        alpha, best_mae = best_for(validation, baseline_col)
        per_baseline[baseline_name] = {"alpha": alpha, "validation_mae": best_mae}
        for candidate_alpha in BLEND_ALPHAS:
            prediction = candidate_alpha * validation["predicted_e0"].to_numpy(dtype=float) + (1 - candidate_alpha) * validation[baseline_col].to_numpy(dtype=float)
            candidates.append({"baseline": baseline_name, "alpha": candidate_alpha,
                               "validation_mae": metric_values(validation["actual_smp"], pd.Series(prediction, index=validation.index))["mae"]})
        for forecast_day, group in validation.groupby("forecast_day", sort=True):
            day_alpha, day_mae = best_for(group, baseline_col)
            xgb_mae = metric_values(group["actual_smp"], group["predicted_e0"])["mae"]
            base_mae = metric_values(group["actual_smp"], group[baseline_col])["mae"]
            per_day.append({"baseline": baseline_name, "forecast_day": int(forecast_day), "alpha": day_alpha,
                            "validation_mae": day_mae, "xgb_only_mae": xgb_mae, "baseline_only_mae": base_mae,
                            "improvement_vs_xgb_pct": (xgb_mae - day_mae) / xgb_mae * 100 if xgb_mae else np.nan})

    chosen = min(candidates, key=lambda row: (row["validation_mae"], -row["alpha"], row["baseline"]))
    walk_forward = []
    month_values = sorted(validation["target_timestamp"].dt.strftime("%Y-%m").unique())
    month_series = validation["target_timestamp"].dt.strftime("%Y-%m")
    for baseline_name, baseline_col in baseline_columns.items():
        for forecast_day in [0, 1, 2, 3]:
            for month in month_values[1:]:
                prior = validation.loc[month_series < month]
                test = validation.loc[month_series == month]
                if forecast_day:
                    prior = prior.loc[prior["forecast_day"] == forecast_day]
                    test = test.loc[test["forecast_day"] == forecast_day]
                if prior.empty or test.empty:
                    continue
                alpha, _ = best_for(prior, baseline_col)
                pred = alpha * test["predicted_e0"].to_numpy(dtype=float) + (1 - alpha) * test[baseline_col].to_numpy(dtype=float)
                model_mae = metric_values(test["actual_smp"], test["predicted_e0"])["mae"]
                base_mae = metric_values(test["actual_smp"], test[baseline_col])["mae"]
                blend_mae = metric_values(test["actual_smp"], pd.Series(pred, index=test.index))["mae"]
                walk_forward.append({"baseline": baseline_name,
                                     "strategy": "global" if forecast_day == 0 else "forecast_day",
                                     "forecast_day": forecast_day if forecast_day else "all",
                                     "test_month": month, "training_months": int(prior["target_timestamp"].dt.strftime("%Y-%m").nunique()),
                                     "alpha": alpha, "blend_mae": blend_mae, "xgb_only_mae": model_mae,
                                     "baseline_only_mae": base_mae,
                                     "improvement_vs_xgb_pct": (model_mae - blend_mae) / model_mae * 100 if model_mae else np.nan})

    return {"baseline": chosen["baseline"], "alpha": chosen["alpha"],
            "validation_mae": chosen["validation_mae"], "by_baseline": per_baseline,
            "candidate_rows": candidates, "per_day_rows": per_day, "walk_forward_rows": walk_forward}


def _group_metrics(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(by, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(by, keys))
        row.update(metric_values(group["actual_smp"], group["predicted_smp"]))
        rows.append(row)
    return pd.DataFrame(rows)


def _baseline_scores(comparison: pd.DataFrame) -> pd.DataFrame:
    group_columns = [col for col in ["split", "forecast_day", "horizon", "target_month"] if col in comparison.columns]
    baseline = comparison.loc[comparison["experiment_id"].isin(["baseline_recent_daily", "baseline_weekly"]),
                              group_columns + ["experiment_id", "mae"]]
    daily = baseline.loc[baseline["experiment_id"] == "baseline_recent_daily", group_columns + ["mae"]].rename(columns={"mae": "baseline_recent_daily_mae"})
    weekly = baseline.loc[baseline["experiment_id"] == "baseline_weekly", group_columns + ["mae"]].rename(columns={"mae": "baseline_weekly_mae"})
    result = comparison.merge(daily, on=group_columns, how="left", validate="many_to_one").merge(weekly, on=group_columns, how="left", validate="many_to_one")
    result["improvement_vs_recent_daily_pct"] = (result["baseline_recent_daily_mae"] - result["mae"]) / result["baseline_recent_daily_mae"] * 100
    result["improvement_vs_weekly_pct"] = (result["baseline_weekly_mae"] - result["mae"]) / result["baseline_weekly_mae"] * 100
    return result


def _attach_change_size(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for split, group in frame.groupby("split", sort=True):
        actuals = group[["target_timestamp", "actual_smp"]].drop_duplicates("target_timestamp").sort_values("target_timestamp")
        actuals["absolute_hourly_change"] = actuals["actual_smp"].diff().abs()
        threshold = actuals["absolute_hourly_change"].quantile(0.90)
        actuals["sharp_change"] = actuals["absolute_hourly_change"] >= threshold
        actuals["high_price"] = actuals["actual_smp"] >= actuals["actual_smp"].quantile(0.90)
        actuals["low_price"] = actuals["actual_smp"] <= actuals["actual_smp"].quantile(0.10)
        tagged = group.merge(actuals[["target_timestamp", "absolute_hourly_change", "sharp_change", "high_price", "low_price"]],
                             on="target_timestamp", how="left", suffixes=("", "_slice"), validate="many_to_one")
        parts.append(tagged)
    return pd.concat(parts, ignore_index=True)


def create_data_drift_analysis() -> dict:
    manifest = json.loads((DATASET_DIR / "manifest.json").read_text(encoding="utf-8"))
    feature_manifest = pd.read_csv(DATASET_DIR / "feature_manifest.csv")
    feature_names = manifest["model_feature_columns"]
    feature_manifest["variable_name"] = feature_manifest["variable_name"].astype(str)
    feature_manifest["missing_indicator"] = feature_manifest["missing_indicator"].fillna("").astype(str)
    mapped = feature_manifest[
        feature_manifest["missing_indicator"].ne("")
        & ~feature_manifest["variable_name"].str.endswith("__is_missing")
        & ~feature_manifest["variable_name"].str.endswith("_missing")
    ].copy()
    flag_columns = [name for name in mapped["missing_indicator"].unique() if name in feature_names]
    lag_columns = [name for name in feature_names if name.startswith("smp_history_")
                   and not name.endswith("_missing") and not name.endswith("__is_missing")]
    read_columns = list(dict.fromkeys(["split", "forecast_origin", "target_timestamp"] + flag_columns + lag_columns))
    input_frame = pd.read_csv(DATASET_DIR / "forecast_inputs.csv", usecols=read_columns, low_memory=False,
                              parse_dates=["forecast_origin", "target_timestamp"])

    missing_rows = []
    for split, split_frame in input_frame.groupby("split", sort=True):
        flags = split_frame[flag_columns].apply(pd.to_numeric, errors="coerce") if flag_columns else pd.DataFrame(index=split_frame.index)
        if not flags.empty and not flags.isin([0, 1, np.nan]).all().all():
            raise ValueError("Common dataset missing indicators contain values other than 0/1")
        missing_rows.append({"split": split, "input_group": "all groups with a missing indicator",
                             "indicator_columns": len(flag_columns), "missing_indicator_rate": float(flags.stack().mean()),
                             "input_rows": len(split_frame)})
        for group_name, group_manifest in mapped.groupby("input_group", dropna=False, sort=True):
            group_flags = [name for name in group_manifest["missing_indicator"].unique() if name in flags.columns]
            if group_flags:
                missing_rows.append({"split": split, "input_group": str(group_name),
                                     "indicator_columns": len(group_flags),
                                     "missing_indicator_rate": float(flags[group_flags].stack().mean()),
                                     "input_rows": len(split_frame)})
    missing_by_group = pd.DataFrame(missing_rows)
    missing_by_group.to_csv(METRICS_DIR / "input_missing_by_group.csv", index=False)

    origin_features = input_frame.drop_duplicates(["split", "forecast_origin"])
    lag_rows = []
    for split, split_frame in origin_features.groupby("split", sort=True):
        for feature in lag_columns:
            values = pd.to_numeric(split_frame[feature], errors="coerce")
            missing_flag = f"{feature}__is_missing"
            flag = pd.to_numeric(split_frame[missing_flag], errors="coerce") if missing_flag in split_frame else pd.Series(np.nan, index=split_frame.index)
            lag_rows.append({"split": split, "feature": feature, "n_origins": len(split_frame),
                             "missing_rate": float(flag.mean()) if flag.notna().any() else np.nan,
                             "mean": float(values.mean()), "std": float(values.std(ddof=0)),
                             "median": float(values.median()), "p10": float(values.quantile(0.10)),
                             "p90": float(values.quantile(0.90))})
    lag_drift = pd.DataFrame(lag_rows)
    lag_drift.to_csv(METRICS_DIR / "smp_history_feature_drift.csv", index=False)

    target = pd.read_csv(DATASET_DIR / "evaluation_targets.csv", usecols=["split", "target_timestamp", "actual_smp"],
                         parse_dates=["target_timestamp"])
    target = target.drop_duplicates(["split", "target_timestamp"])
    price_rows = []
    for split, split_frame in target.groupby("split", sort=True):
        values = pd.to_numeric(split_frame["actual_smp"], errors="coerce").dropna()
        price_rows.append({"split": split, "unique_target_hours": len(values), "mean": float(values.mean()),
                           "std": float(values.std(ddof=0)), "min": float(values.min()),
                           "p10": float(values.quantile(0.10)), "median": float(values.median()),
                           "p90": float(values.quantile(0.90)), "p99": float(values.quantile(0.99)),
                           "max": float(values.max())})
    target_distribution = pd.DataFrame(price_rows)
    target_distribution.to_csv(METRICS_DIR / "target_price_distribution.csv", index=False)
    return {"missing_by_group": missing_by_group, "lag_drift": lag_drift,
            "target_distribution": target_distribution}


def create_figures(validation: pd.DataFrame, final_test: pd.DataFrame, metrics_day: pd.DataFrame,
                   metrics_horizon: pd.DataFrame, metrics_month: pd.DataFrame, config: dict) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    comparison = pd.concat([validation, final_test], ignore_index=True)
    selected = ["E0_xgb", "baseline_recent_daily", "baseline_weekly", "E1_selected_blend", "E2_train_plus_validation", "E3_retrained_blend"]
    labels = {"E0_xgb": "Existing XGBoost", "baseline_recent_daily": "Recent daily baseline",
              "baseline_weekly": "Weekly baseline", "E1_selected_blend": "E1 selected blend",
              "E2_train_plus_validation": "Train+Val refit", "E3_retrained_blend": "Refit + blend"}

    c = _group_metrics(comparison[comparison["experiment_id"].isin(selected)], ["split", "experiment_id"])
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(selected))
    width = 0.36
    for j, split in enumerate(["validation", "final_test"]):
        vals = [float(c.loc[(c["split"] == split) & (c["experiment_id"] == exp), "mae"].iloc[0])
                if ((c["split"] == split) & (c["experiment_id"] == exp)).any() else np.nan for exp in selected]
        ax.bar(x + (j - 0.5) * width, vals, width, label=split.replace("_", " ").title())
    ax.set_xticks(x, [labels[v] for v in selected], rotation=18, ha="right")
    ax.set_ylabel("MAE (KRW/kWh)")
    ax.set_title("Validation and Final Test model comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.5, 0.015, "E2/E3 are fit on Train+Validation; no in-sample Validation score is shown.", ha="center", fontsize=9)
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=160)
    plt.close(fig)

    plot_ids = ["E0_xgb", "E2_train_plus_validation", "E3_retrained_blend"]
    fig, ax = plt.subplots(figsize=(11, 6))
    for exp in plot_ids:
        for split in ["validation", "final_test"]:
            part = metrics_month[(metrics_month["split"] == split) & (metrics_month["experiment_id"] == exp)]
            if not part.empty:
                ax.plot(part["target_month"], part["mae"], marker="o", label=f"{labels[exp]} ({split.replace('_', ' ')})")
    ax.axvline("2025-07", color="black", linestyle="--", alpha=0.5)
    ax.set_ylabel("MAE (KRW/kWh)")
    ax.set_xlabel("Target month")
    ax.set_title("Monthly MAE (Validation and Final Test)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "monthly_mae.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for split in ["validation", "final_test"]:
        for exp in ["E0_xgb", "E2_train_plus_validation", "E3_retrained_blend"]:
            part = metrics_horizon[(metrics_horizon["split"] == split) & (metrics_horizon["experiment_id"] == exp)]
            if not part.empty:
                ax.plot(part["horizon"], part["mae"], label=f"{labels[exp]} ({split.replace('_', ' ')})", alpha=0.9)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("MAE (KRW/kWh)")
    ax.set_title("Horizon-specific MAE")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "horizon_mae.png", dpi=160)
    plt.close(fig)

    chosen_origin = pd.Timestamp("2025-07-01 23:00:00")
    sample = final_test[final_test["forecast_origin"] == chosen_origin]
    if sample.empty:
        chosen_origin = final_test["forecast_origin"].min()
        sample = final_test[final_test["forecast_origin"] == chosen_origin]
    fig, ax = plt.subplots(figsize=(12, 6))
    actual = sample[sample["experiment_id"] == "E0_xgb"].sort_values("target_timestamp")
    if actual.empty:
        actual = sample.drop_duplicates("target_timestamp").sort_values("target_timestamp")
    ax.plot(actual["target_timestamp"], actual["actual_smp"], color="black", marker=".", label="Actual SMP")
    for exp, label in [("E0_xgb", "Existing XGBoost"), ("E2_train_plus_validation", "Train+Val refit"),
                       ("E3_retrained_blend", "Refit + selected baseline blend")]:
        part = sample[sample["experiment_id"] == exp].sort_values("target_timestamp")
        if not part.empty:
            ax.plot(part["target_timestamp"], part["predicted_smp"], marker=".", label=label)
    for boundary in [24, 48]:
        ax.axvline(actual["target_timestamp"].iloc[boundary - 1], color="gray", linestyle="--", alpha=0.6)
    ax.set_ylabel("SMP (KRW/kWh)")
    ax.set_title(f"72-hour forecast comparison: {chosen_origin:%Y-%m-%d %H:%M}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "selected_72h_forecast.png", dpi=160)
    plt.close(fig)


def evaluate() -> dict:
    validation_path = PREDICTIONS_DIR / "experiments_validation.csv"
    test_path = PREDICTIONS_DIR / "final_test_predictions.csv"
    validation = pd.read_csv(validation_path, parse_dates=["forecast_origin", "observed_cutoff", "target_timestamp"])
    final_test = pd.read_csv(test_path, parse_dates=["forecast_origin", "observed_cutoff", "target_timestamp"])
    for split, frame, expected in [("validation", validation, 12888), ("final_test", final_test, 13104)]:
        if set(frame["split"].unique()) != {split}:
            raise ValueError(f"Unexpected split in {split} prediction file")
        if frame.groupby("experiment_id").size().nunique() != 1:
            raise ValueError(f"Experiment coverage differs within {split}")
        for experiment, group in frame.groupby("experiment_id"):
            if len(group) != expected or group["actual_smp"].isna().any() or group["predicted_smp"].isna().any():
                raise ValueError(f"Incomplete {split} coverage for {experiment}")
            if group.duplicated(["forecast_origin", "target_timestamp", "horizon"]).any():
                raise ValueError(f"Duplicate prediction row for {experiment} in {split}")

    all_predictions = pd.concat([validation, final_test], ignore_index=True)
    comparison = _group_metrics(all_predictions, ["split", "experiment_id", "model_name"])
    comparison = _baseline_scores(comparison)
    comparison.to_csv(METRICS_DIR / "experiment_comparison.csv", index=False)
    metrics_day = _group_metrics(all_predictions, ["split", "experiment_id", "model_name", "forecast_day"])
    metrics_day = _baseline_scores(metrics_day)
    metrics_day.to_csv(METRICS_DIR / "metrics_by_forecast_day.csv", index=False)
    metrics_horizon = _group_metrics(all_predictions, ["split", "experiment_id", "model_name", "horizon"])
    metrics_horizon = _baseline_scores(metrics_horizon)
    metrics_horizon.to_csv(METRICS_DIR / "metrics_by_horizon.csv", index=False)
    all_predictions["target_month"] = all_predictions["target_timestamp"].dt.strftime("%Y-%m")
    metrics_month = _group_metrics(all_predictions, ["split", "experiment_id", "model_name", "target_month"])
    metrics_month = _baseline_scores(metrics_month)
    metrics_month.to_csv(METRICS_DIR / "metrics_by_month.csv", index=False)
    metrics_month_by_day = _group_metrics(all_predictions, ["split", "experiment_id", "model_name", "target_month", "forecast_day"])
    metrics_month_by_day = _baseline_scores(metrics_month_by_day)
    metrics_month_by_day.to_csv(METRICS_DIR / "metrics_by_month_by_forecast_day.csv", index=False)

    tagged = _attach_change_size(all_predictions)
    bias = _group_metrics(tagged, ["split", "experiment_id", "model_name"])
    slice_rows = []
    for keys, group in tagged.groupby(["split", "experiment_id", "model_name"], sort=True):
        for label, mask in [("high_price_top_decile", group["high_price"]),
                            ("low_price_bottom_decile", group["low_price"]),
                            ("sharp_hourly_change_top_decile", group["sharp_change"])]:
            scored = metric_values(group.loc[mask, "actual_smp"], group.loc[mask, "predicted_smp"])
            slice_rows.append({"split": keys[0], "experiment_id": keys[1], "model_name": keys[2], "slice": label, **scored})
    slice_df = pd.DataFrame(slice_rows)
    bias.to_csv(METRICS_DIR / "bias_analysis.csv", index=False)
    slice_df.to_csv(METRICS_DIR / "event_slice_analysis.csv", index=False)

    drift = create_data_drift_analysis()
    config = json.loads((UPGRADE_DIR / "config" / "experiment_config.json").read_text(encoding="utf-8"))
    create_figures(validation, final_test, metrics_day, metrics_horizon, metrics_month, config)
    write_reports(config, comparison, metrics_day, bias, slice_df, drift)
    return {"comparison": comparison, "by_day": metrics_day, "bias": bias}


def _fmt(value: float) -> str:
    return "NA" if pd.isna(value) else f"{value:.4f}"


def write_reports(config: dict, comparison: pd.DataFrame, by_day: pd.DataFrame,
                  bias: pd.DataFrame, event_slices: pd.DataFrame, drift: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    selected = config["blend_selection"]
    selected_baseline = selected["baseline"]
    selected_alpha = selected["alpha"]
    walk_forward = pd.DataFrame(selected["walk_forward_rows"])
    per_day = pd.DataFrame(selected["per_day_rows"])
    selected_wf = walk_forward[walk_forward["baseline"] == selected_baseline]
    global_wf = selected_wf[selected_wf["strategy"] == "global"]
    day_wf = selected_wf[selected_wf["strategy"] == "forecast_day"]
    per_day_selected = per_day[per_day["baseline"] == selected_baseline]
    global_gain = float(global_wf["improvement_vs_xgb_pct"].mean()) if not global_wf.empty else np.nan
    day_gain = float(day_wf["improvement_vs_xgb_pct"].mean()) if not day_wf.empty else np.nan
    global_wins = int((global_wf["improvement_vs_xgb_pct"] > 0).sum()) if not global_wf.empty else 0
    day_wins = int((day_wf["improvement_vs_xgb_pct"] > 0).sum()) if not day_wf.empty else 0
    per_day_alpha_text = ", ".join(
        f"D+{int(row.forecast_day)}={row.alpha:.1f}" for row in per_day_selected.itertuples()
    )
    baseline_lines = [
        "# Baseline and blend comparison", "",
        "Blend weights were selected using Validation only. Alpha means `alpha * XGBoost + (1-alpha) * baseline`.",
        f"Selected baseline: `{selected_baseline}`; alpha: `{selected_alpha:.1f}`.",
        "The E0 row was recomputed from the existing prediction CSV on the same target rows; it was not retrained.",
        f"Full-Validation per-day optima for the selected baseline were {per_day_alpha_text}. Walk-forward validation over five later months improved over E0 in {global_wins}/{len(global_wf)} global folds (mean {global_gain:.2f}%) and {day_wins}/{len(day_wf)} per-day folds (mean {day_gain:.2f}%). The mean difference is small and the first fold regresses, so one global weight is retained.",
        "Candidate weights and all monthly folds are in `outputs/metrics/blend_candidates_validation.csv` and `outputs/metrics/blend_weight_walk_forward.csv`.", "",
        "| Split | Experiment | MAE | RMSE | R² | vs recent daily (%) | vs weekly (%) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.iterrows():
        baseline_lines.append(f"| {row['split']} | {row['experiment_id']} | {_fmt(row['mae'])} | {_fmt(row['rmse'])} | {_fmt(row['r2'])} | {_fmt(row['improvement_vs_recent_daily_pct'])} | {_fmt(row['improvement_vs_weekly_pct'])} |")
    (REPORTS_DIR / "baseline_comparison.md").write_text("\n".join(baseline_lines) + "\n", encoding="utf-8")
    pd.DataFrame(selected["walk_forward_rows"]).to_csv(METRICS_DIR / "blend_weight_walk_forward.csv", index=False)
    pd.DataFrame(selected["per_day_rows"]).to_csv(METRICS_DIR / "blend_weight_by_day_diagnostic.csv", index=False)

    e0_val = comparison[(comparison["split"] == "validation") & (comparison["experiment_id"] == "E0_xgb")].iloc[0]
    e0_test = comparison[(comparison["split"] == "final_test") & (comparison["experiment_id"] == "E0_xgb")].iloc[0]
    e2 = comparison[(comparison["split"] == "final_test") & (comparison["experiment_id"] == "E2_train_plus_validation")].iloc[0]
    e3 = comparison[(comparison["split"] == "final_test") & (comparison["experiment_id"] == "E3_retrained_blend")].iloc[0]
    e3_vs_e0_pct = (e0_test["mae"] - e3["mae"]) / e0_test["mae"] * 100
    e3_vs_daily_pct = e3["improvement_vs_recent_daily_pct"]
    retrain = [
        "# Train+Validation retraining", "",
        "The selected parameters were taken from the existing Validation-only candidate search. The model was refit once on Train plus Validation (2023-01-01 through 2025-06-30); Final Test labels were not used for fitting or selection.",
        f"Parameters: `{json.dumps(config['retrain_parameters'], sort_keys=True)}`.",
        f"Training rows: {config['retraining']['training_rows']}; features: {config['retraining']['feature_count']}; fit duration: {config['retraining']['fit_seconds']:.1f}s.",
        "The selected E1 blend weight was fixed before the refit and reused for E3. No post-refit blend weight was optimized on Final Test.", "",
        f"E0 Validation MAE: {e0_val['mae']:.4f}; E0 Final Test MAE: {e0_test['mae']:.4f}.",
        f"E2 refit Final Test MAE: {e2['mae']:.4f}; E3 refit-plus-blend Final Test MAE: {e3['mae']:.4f}.",
        "These scores use the same Final Test targets and the original E0 evaluation rows.", "",
        "The previously inspected Final Test period is reused, so this is a re-evaluation of a known period, not a blind future test.",
    ]
    (REPORTS_DIR / "retraining_report.md").write_text("\n".join(retrain) + "\n", encoding="utf-8")

    bias_lines = [
        "# Forecast-distance bias and error analysis", "",
        "Bias is `mean(predicted_smp - actual_smp)`: positive means overprediction; negative means underprediction.",
        "Prediction standard deviation is compared with actual standard deviation to show compression or excess spread.",
        "High/low-price and sharp-change slices are descriptive post-evaluation slices based on actual values; they are not used to choose the model or weights.", "",
        "| Split | Experiment | Bias | Overprediction share | Underprediction share | Actual SD | Predicted SD | SD ratio |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in bias.iterrows():
        bias_lines.append(f"| {row['split']} | {row['experiment_id']} | {_fmt(row['bias_pred_minus_actual'])} | {_fmt(row['overprediction_share'])} | {_fmt(row['underprediction_share'])} | {_fmt(row['actual_std'])} | {_fmt(row['predicted_std'])} | {_fmt(row['std_ratio_pred_to_actual'])} |")
    bias_lines.extend(["", "## D+1/D+2/D+3 MAE", "", "| Split | Experiment | Day | MAE | RMSE | R² | Bias |", "|---|---|---:|---:|---:|---:|---:|"])
    for _, row in by_day.iterrows():
        bias_lines.append(f"| {row['split']} | {row['experiment_id']} | D+{int(row['forecast_day'])} | {_fmt(row['mae'])} | {_fmt(row['rmse'])} | {_fmt(row['r2'])} | {_fmt(row['bias_pred_minus_actual'])} |")
    bias_lines.extend(["", "## Price and rapid-change slices", "", "See `outputs/metrics/event_slice_analysis.csv` for MAE/RMSE by split and experiment across the upper/lower actual-price deciles and the upper decile of absolute hour-to-hour actual-price changes."])
    final_slices = event_slices[(event_slices["split"] == "final_test") &
                                (event_slices["experiment_id"].isin(["E0_xgb", "E2_train_plus_validation", "E3_retrained_blend", "baseline_weekly"]))]
    bias_lines.extend(["", "| Final Test slice | Experiment | Rows | MAE | RMSE |", "|---|---|---:|---:|---:|"])
    for _, row in final_slices.iterrows():
        bias_lines.append(f"| {row['slice']} | {row['experiment_id']} | {int(row['n_rows'])} | {_fmt(row['mae'])} | {_fmt(row['rmse'])} |")
    (REPORTS_DIR / "bias_analysis.md").write_text("\n".join(bias_lines) + "\n", encoding="utf-8")

    target_distribution = drift["target_distribution"].set_index("split")
    missing_summary = drift["missing_by_group"].loc[
        drift["missing_by_group"]["input_group"] == "all groups with a missing indicator"
    ].set_index("split")
    latest_lag = drift["lag_drift"].loc[drift["lag_drift"]["feature"] == "smp_history_latest"].set_index("split")
    lag_missing = drift["lag_drift"].groupby("split")["missing_rate"].mean()
    target_mean_change = target_distribution.loc["final_test", "mean"] - target_distribution.loc["validation", "mean"]
    target_mean_change_pct = target_mean_change / target_distribution.loc["validation", "mean"] * 100
    val_missing = missing_summary.loc["validation", "missing_indicator_rate"]
    test_missing = missing_summary.loc["final_test", "missing_indicator_rate"]
    e0_months = pd.read_csv(METRICS_DIR / "metrics_by_month.csv")
    e0_test_months = e0_months[(e0_months["split"] == "final_test") & (e0_months["experiment_id"] == "E0_xgb")].sort_values("target_month")
    july_bias = float(e0_test_months.iloc[0]["bias_pred_minus_actual"])
    december_bias = float(e0_test_months.iloc[-1]["bias_pred_minus_actual"])
    group_rates = drift["missing_by_group"].loc[
        drift["missing_by_group"]["input_group"] != "all groups with a missing indicator"
    ].copy()
    group_rates["input_group"] = group_rates["input_group"].astype(str)
    group_rates = group_rates.set_index(["input_group", "split"])["missing_indicator_rate"]
    group_rate_text = "; ".join(
        f"group {group}: Train {group_rates.get((group, 'train'), np.nan):.2%}, Validation {group_rates.get((group, 'validation'), np.nan):.2%}, Final Test {group_rates.get((group, 'final_test'), np.nan):.2%}"
        for group in ["1", "6", "7", "11"] if (group, "validation") in group_rates.index or (group, "final_test") in group_rates.index
    )
    drift_lines = [
        "# Validation/Test shift diagnostics", "",
        "Target distribution uses unique target hours so overlapping 72-hour forecast blocks do not multiply the same SMP hour. Missing-input rates use the common dataset's pre-imputation missing flags; these are diagnostics only and do not change preprocessing.", "",
        "| Split | Unique target hours | SMP mean | SMP SD | P10 | Median | P90 | P99 | Max | Measured-input missing-flag rate | SMP-history lag missing rate | SMP-history latest mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ["train", "validation", "final_test"]:
        p = target_distribution.loc[split]
        m = missing_summary.loc[split]
        drift_lines.append(
            f"| {split} | {int(p['unique_target_hours'])} | {_fmt(p['mean'])} | {_fmt(p['std'])} | {_fmt(p['p10'])} | {_fmt(p['median'])} | {_fmt(p['p90'])} | {_fmt(p['p99'])} | {_fmt(p['max'])} | {_fmt(m['missing_indicator_rate'])} | {_fmt(lag_missing.loc[split])} | {_fmt(latest_lag.loc[split, 'mean'])} |"
        )
    drift_lines.extend([
        "", "Input-group missing-flag rates are in `outputs/metrics/input_missing_by_group.csv`; individual SMP history lag/rolling distributions are in `outputs/metrics/smp_history_feature_drift.csv`.",
        f"Validation-to-Final-Test unique-hour SMP mean changed by {target_mean_change:.2f} KRW/kWh ({target_mean_change_pct:.2f}%). The `smp_history_latest` mean changed from {latest_lag.loc['validation', 'mean']:.2f} to {latest_lag.loc['final_test', 'mean']:.2f} KRW/kWh, while its missing rate remained {latest_lag.loc['validation', 'missing_rate']:.2%}/{latest_lag.loc['final_test', 'missing_rate']:.2%}.",
        f"The overall pre-imputation indicator rate changed from {val_missing:.2%} to {test_missing:.2%}; group-level changes are in the CSV. E0 Final Test monthly bias rose from {july_bias:.2f} in July to {december_bias:.2f} KRW/kWh in December.",
        f"Selected group missing-flag rates: {group_rate_text}.",
        "This pattern is consistent with a falling SMP level and growing positive E0 bias; missingness changed only modestly overall and the recent SMP lag remained populated. These diagnostics do not establish a single cause.",
    ])
    (REPORTS_DIR / "validation_test_shift_analysis.md").write_text("\n".join(drift_lines) + "\n", encoding="utf-8")

    final = [
        "# Final XGBoost 72-hour upgrade report", "",
        f"Dataset: `{DATASET_VERSION}`. Forecast origin is D 23:00 Asia/Seoul; target is D+1 00:00 through D+3 23:00. All 12 common input groups, the common evaluation rows, and the 72-hour observed cutoff are unchanged.",
        f"The final-test period had already been examined in the prior experiment; these results are a re-evaluation and not a blind test. No Final Test metric was used to select hyperparameters, blend weights, or corrections.", "",
        f"Validation-selected blend: `{selected_baseline}`, alpha `{selected_alpha:.1f}`. Validation rows: {int(e0_val['n_rows'])}; Final Test rows: {int(e0_test['n_rows'])} per experiment.",
        f"Existing E0 Final Test MAE: {e0_test['mae']:.4f} KRW/kWh. Refit E2 MAE: {e2['mae']:.4f}; refit plus selected blend E3 MAE: {e3['mae']:.4f}, a {e3_vs_e0_pct:.2f}% reduction from E0.",
        f"E0 Final Test RMSE/R²: {e0_test['rmse']:.4f}/{e0_test['r2']:.4f}; E3 RMSE/R²: {e3['rmse']:.4f}/{e3['r2']:.4f}.",
        f"Recent-daily baseline MAE: {e0_test['baseline_recent_daily_mae']:.4f}; E3 improvement over it: {e3_vs_daily_pct:.2f}%. Weekly-baseline MAE on the same Final Test rows: {e0_test['baseline_weekly_mae']:.4f}; E3 improvement over it: {e3['improvement_vs_weekly_pct']:.2f}% (still worse than weekly).", "",
        "## Data and evaluation limits", "",
        "- Actual demand, generation, reserve, and weather publication times were not individually verified. The 72-hour cutoff remains a conservative study assumption, not a measured publication delay.",
        "- Domestic LNG is matched by target month under the conditional historical experiment; availability at forecast origin is unverified.",
        "- Original inputs, baseline predictions, fixed origins, and targets are read-only. No new APIs, input groups, or feature-selection experiments were introduced.",
        "- Validation/Test target-price distributions, input missingness, and SMP-history lag-feature distributions are compared in `reports/validation_test_shift_analysis.md`; these diagnostics are not causal proof.",
        "- No separate forecast-day models or bias correction were applied. The same per-day alpha was selected for all three days, while walk-forward per-day gain was not robust enough to justify separate settings.",
        "- Existing group-5 data coverage, source limitations, and imputation remain as documented in the common dataset quality report.",
        "- The existing model and its outputs remain unchanged; all upgrade artifacts are below this folder.", "",
        "## Files", "",
        "- Validation experiment predictions: `outputs/predictions/experiments_validation.csv`",
        "- Final Test predictions: `outputs/predictions/final_test_predictions.csv`",
        "- Metrics: `outputs/metrics/`",
        "- Figures: `outputs/figures/`",
        "- New Train+Validation model: `models/final_xgboost_model.json`",
        "- Blend, fit, and source metadata: `config/experiment_config.json`", "",
        "## Reproduction", "",
        "Run from repository root after restoring the bundled common dataset if needed:", "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe SMP_XGBOOST_2026\\upgrades\\upgrade_72h_20261119\\scripts\\run_upgrade.py",
        ".\\.venv\\Scripts\\python.exe SMP_XGBOOST_2026\\upgrades\\upgrade_72h_20261119\\scripts\\evaluate_upgrade.py",
        "```", "",
        "To expand the bundled compressed dataset first, run `python SMP_XGBOOST_2026/scripts/restore_shared_dataset.py`; it refuses to overwrite an existing dataset version.",
    ]
    (REPORTS_DIR / "final_upgrade_report.md").write_text("\n".join(final) + "\n", encoding="utf-8")


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    result = evaluate()
    print(result["comparison"].to_string(index=False))


if __name__ == "__main__":
    main()
