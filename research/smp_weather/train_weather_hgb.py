"""Fit-only utilities for the separate SMP + KMA forecast HGB candidate.

This module writes only under research/smp_weather/models. It never touches
automation/models or the official V20 issuance path.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from research.smp_weather.weather_features import WEATHER_MODEL_FEATURES

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "research" / "smp_weather" / "models"
MIN_TRAIN_DAYS = 365
SOURCE = Path(os.environ.get("SMP_SOURCE", ROOT.parent / "gradproject")).resolve()
sys.path.insert(0, str(SOURCE))
from v11.store import connect, latest
from research.smp_weather.weather_features import WEATHER_MODEL_FEATURES, load_vintages, training_examples

WEATHER_PATH = ROOT / "automation/state/weather/forecast_vintages.jsonl"


def make_model() -> HistGradientBoostingRegressor:
    """Use the incumbent D+3 HGB hyperparameters, held fixed for first test."""
    return HistGradientBoostingRegressor(max_iter=250, max_leaf_nodes=31,
        learning_rate=0.05, early_stopping=False, l2_regularization=2.0,
        random_state=42)


def fit_weather_hgb(X: pd.DataFrame, residual: pd.Series, *, target_dates: list[str],
                    trained_through: str, output_dir: Path = MODEL_DIR) -> dict:
    """Fit and save a separate residual model; reject small or invalid samples."""
    output_dir = Path(output_dir)
    if tuple(X.columns) != tuple(WEATHER_MODEL_FEATURES):
        raise ValueError("Weather HGB feature schema does not match the candidate manifest")
    y = np.asarray(residual, dtype=float)
    if len(X) != len(y) or not len(y):
        raise ValueError("Weather HGB training matrix and residual labels are empty or misaligned")
    if len(set(target_dates)) < MIN_TRAIN_DAYS:
        raise ValueError(f"Weather HGB requires at least {MIN_TRAIN_DAYS} vintage-complete training days")
    if not np.isfinite(X.to_numpy(float)).all() or not np.isfinite(y).all():
        raise ValueError("Weather HGB training inputs contain non-finite values")
    model = make_model().fit(X.to_numpy(float), y)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "d3_smp_weather_hgb.joblib"
    import joblib
    joblib.dump({"model": model, "features": list(WEATHER_MODEL_FEATURES),
        "horizon_days": 3, "residual_target": "actual_smp - same_hour_3day_price",
        "anchor_weight": 0.05, "anchor_days": [7, 14, 21, 28],
        "trained_from": min(target_dates), "trained_through": trained_through,
        "training_days": len(set(target_dates)), "training_rows": len(y),
        "training_weather_vintage_verified": True,
        "model_family": "SMP + KMA issued forecast HGB residual + 5% seasonal anchor"}, path, compress=3)
    manifest = {"model": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "trained_from": min(target_dates), "trained_through": trained_through,
        "training_days": len(set(target_dates)), "training_rows": len(y),
        "horizon_days": 3, "anchor_weight": 0.05,
        "features": list(WEATHER_MODEL_FEATURES),
        "estimator": "HistGradientBoostingRegressor", "max_iter": 250,
        "max_leaf_nodes": 31, "learning_rate": 0.05,
        "early_stopping": False, "l2_regularization": 2.0, "random_state": 42,
        "scikit_learn": sklearn.__version__,
        "note": "Separate research candidate; never auto-promoted to SMP-only issuance."}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit the separate research-only SMP + weather D+3 HGB.")
    parser.add_argument("--train-before", required=True,
        help="Exclusive KST date cutoff; labels and weather inputs must precede each sample's own D-3 origin.")
    parser.add_argument("--weather-file", type=Path, default=WEATHER_PATH)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    db = connect(SOURCE / "artifacts/v11")
    try:
        prices = latest(db, "land").sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    finally:
        db.close()
    cutoff = pd.Timestamp(args.train_before, tz="Asia/Seoul")
    prices = prices[prices.timestamp < cutoff].copy()
    weather = load_vintages(args.weather_file)
    X, y = training_examples(prices, weather, args.train_before)
    dates = list(X.attrs.get("target_dates", []))
    if len(dates) < MIN_TRAIN_DAYS:
        print(json.dumps({"status": "blocked_insufficient_origin_eligible_weather_days",
            "complete_training_days": len(dates), "required_days": MIN_TRAIN_DAYS,
            "model_written": False, "weather_file": str(args.weather_file)}, ensure_ascii=False))
        return 2
    manifest = fit_weather_hgb(X, y, target_dates=dates,
        trained_through=max(dates), output_dir=args.output_dir)
    print(json.dumps({"status": "trained_research_candidate", **manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
