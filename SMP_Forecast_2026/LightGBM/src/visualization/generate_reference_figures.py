"""Create the two LightGBM-only Final Test figures styled after XGBoost outputs.

This utility reads saved LightGBM predictions and hourly metrics. It does not
load input data, fit a model, or create model-comparison figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREDICTIONS_PATH = PROJECT_ROOT / "LightGBM/outputs/predictions/lightgbm_day_ahead_predictions.csv"
HOURLY_METRICS_PATH = PROJECT_ROOT / "LightGBM/outputs/metrics/hourly_metrics.csv"
FIGURE_DIR = PROJECT_ROOT / "LightGBM/outputs/figures"


def generate_figures() -> dict[str, str]:
    if not PREDICTIONS_PATH.exists() or not HOURLY_METRICS_PATH.exists():
        raise FileNotFoundError("Saved LightGBM predictions and hourly metrics are required.")

    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=["target_timestamp", "target_date", "forecast_origin"],
        float_precision="round_trip",
    )
    test = predictions.loc[predictions.split == "final_test"].copy()
    test = test.sort_values("target_timestamp").reset_index(drop=True)
    if len(test) != 4416:
        raise ValueError(f"Expected 4,416 saved Final Test predictions, found {len(test)}.")
    if test.target_timestamp.min() != pd.Timestamp("2025-07-01 00:00:00"):
        raise ValueError("Saved LightGBM Final Test does not start on 2025-07-01 00:00 KST.")
    if test.target_timestamp.max() != pd.Timestamp("2025-12-31 23:00:00"):
        raise ValueError("Saved LightGBM Final Test does not end on 2025-12-31 23:00 KST.")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    first_date = test.target_date.min()
    week = test.loc[test.target_date < first_date + pd.Timedelta(days=7)]

    # Match the XGBoost example-week layout, using the LightGBM series color.
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(week.target_timestamp, week.actual_smp, marker="o", ms=2, color="#222222", label="Actual SMP")
    ax.plot(week.target_timestamp, week.lightgbm_smp, marker=".", ms=2, color="#16865b", label="LightGBM")
    ax.plot(week.target_timestamp, week.daily_baseline_smp, color="#e07a1f", alpha=0.8, label="Previous-day baseline")
    ax.set_title(f"Final Test example week — {first_date:%Y-%m-%d}")
    ax.set_xlabel("Target timestamp (Asia/Seoul)")
    ax.set_ylabel("SMP (KRW/kWh)")
    ax.grid(alpha=0.2)
    ax.legend(ncol=3)
    fig.tight_layout()
    example_path = FIGURE_DIR / "lightgbm_final_test_example_week.png"
    fig.savefig(example_path, dpi=160)
    plt.close(fig)

    hourly = pd.read_csv(HOURLY_METRICS_PATH)
    hourly = hourly.loc[hourly.split == "final_test"].sort_values("target_hour")
    if len(hourly) != 24 or hourly.target_hour.tolist() != list(range(24)):
        raise ValueError("Saved Final Test hourly metrics must contain target hours 0–23.")

    # Match the XGBoost hourly-MAE layout; only the LightGBM model is plotted.
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hourly.target_hour, hourly.lightgbm_mae, marker="o", color="#16865b", label="LightGBM MAE")
    ax.plot(hourly.target_hour, hourly.baseline_mae, marker="o", color="#e07a1f", label="Previous-day baseline MAE")
    ax.set_xticks(range(24))
    ax.set_xlabel("Target hour (KST)")
    ax.set_ylabel("MAE (KRW/kWh)")
    ax.set_title("Final Test error by target hour")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    hourly_path = FIGURE_DIR / "lightgbm_final_test_hourly_mae.png"
    fig.savefig(hourly_path, dpi=160)
    plt.close(fig)

    return {"example_week": str(example_path), "hourly_mae": str(hourly_path)}


if __name__ == "__main__":
    outputs = generate_figures()
    print(f"Created LightGBM-only figures from saved predictions for 2025-07-01 through 2025-12-31:")
    for name, path in outputs.items():
        print(f"{name}: {path}")
