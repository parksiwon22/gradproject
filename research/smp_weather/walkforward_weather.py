"""Separate, vintage-safe 2026 D+3 walk-forward for SMP + weather HGB.

The incumbent 316-day artifact is read-only and supplies the exact comparison
cohort, baseline predictions, actual SMP, and EV conditions. This script writes
its candidate and comparison results to research/smp_weather/results only.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("SMP_SOURCE", ROOT.parent / "gradproject")).resolve()
sys.path.insert(0, str(SOURCE))
from v11.store import connect, latest

from automation.smp_auto.walkforward_d3_2026 import _metrics, KST
from research.smp_weather.train_weather_hgb import MIN_TRAIN_DAYS, make_model
from research.smp_weather.weather_features import (
    WEATHER_MODEL_FEATURES, index_vintages_by_target, load_vintages,
    target_feature_frame, training_examples,
)

BASELINE_ARTIFACT = ROOT / "docs/live/walkforward_d3_2026.json.gz"
DEFAULT_WEATHER_DATA = ROOT / "automation/state/weather/forecast_vintages.jsonl"
OUT = ROOT / "research/smp_weather/results/walkforward_d3_2026.json"
ANCHOR_WEIGHT = 0.05


def _load_saved_smp_only(path: Path = BASELINE_ARTIFACT) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("horizon_days") != 3 or len(data.get("records", [])) != 316:
        raise ValueError("preserved SMP-only D+3 artifact schema/cohort changed; refusing comparison")
    return data


def _read_prices(source: Path = SOURCE) -> pd.DataFrame:
    db = connect(source / "artifacts/v11")
    try:
        frame = latest(db, "land").sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    finally:
        db.close()
    return frame[np.isfinite(frame.smp.to_numpy(float))].copy()


def _record_metrics(prediction: np.ndarray, actual: np.ndarray,
                    baseline: np.ndarray) -> dict:
    m = _metrics(prediction, actual, baseline, include_v2g=True)
    actual, prediction = np.asarray(actual, float), np.asarray(prediction, float)
    peak_mask = actual >= 150.0
    low_n = min(6, len(actual))
    pred_low = set(np.argsort(prediction, kind="stable")[:low_n].tolist())
    actual_low = set(np.argsort(actual, kind="stable")[:low_n].tolist())
    m.update({
        "high_price_threshold_won_per_kwh": 150.0,
        "high_price_hours": int(peak_mask.sum()),
        "high_price_mae": float(np.mean(np.abs(prediction[peak_mask] - actual[peak_mask]))) if peak_mask.any() else None,
        "low_price_top6_overlap_rate": len(pred_low & actual_low) / low_n,
    })
    return m


def _aggregate(records: list[dict]) -> dict:
    if not records:
        return {"n_days": 0, "mae": None, "rmse": None, "mean_daily_g2v_regret_won": None,
                "mean_daily_v2g_regret_won": None}
    pred = np.concatenate([np.asarray(r["predictions"], float) for r in records])
    actual = np.concatenate([np.asarray(r["actual_smp"], float) for r in records])
    metrics = [r["metrics"] for r in records]
    return {
        "n_days": len(records), "first_target_date": records[0]["target_date"],
        "last_target_date": records[-1]["target_date"],
        "mae": float(np.mean(np.abs(pred - actual))),
        "rmse": float(np.sqrt(np.mean((pred - actual) ** 2))),
        "high_price_mae": float(np.mean([m["high_price_mae"] for m in metrics if m["high_price_mae"] is not None])) if any(m["high_price_mae"] is not None for m in metrics) else None,
        "low_price_top6_overlap_rate": float(np.mean([m["low_price_top6_overlap_rate"] for m in metrics])),
        "mean_daily_g2v_regret_won": float(np.mean([m["g2v_regret_won"] for m in metrics])),
        "mean_daily_v2g_regret_won": float(np.mean([m["v2g_regret_won"] for m in metrics])),
        "mean_v2g_gain_vs_g2v_won": float(np.mean([m["v2g_gain_vs_g2v_actual_won"] for m in metrics])),
        "ev_conditions": metrics[0]["ev_conditions"],
    }


def run(*, source: Path = SOURCE, weather_path: Path = DEFAULT_WEATHER_DATA,
        baseline_path: Path = BASELINE_ARTIFACT, out: Path = OUT) -> dict:
    baseline = _load_saved_smp_only(baseline_path)
    weather_records = load_vintages(weather_path)
    weather_index = index_vintages_by_target(weather_records)
    result = {
        "schema_version": 1,
        "status": "pending_weather_vintages",
        "cohort": "2026 D+3 saved time-ordered replay; paired dates only",
        "incumbent_artifact": str(baseline_path.relative_to(ROOT)).replace("\\", "/"),
        "weather_source": str(weather_path),
        "weather_record_count": len(weather_records),
        "model": {"family": "SMP + KMA issued forecast HGB residual + 5% seasonal anchor",
                  "estimator": "HistGradientBoostingRegressor", "max_iter": 250,
                  "max_leaf_nodes": 31, "learning_rate": 0.05,
                  "early_stopping": False, "l2_regularization": 2.0,
                  "random_state": 42, "anchor_weight": ANCHOR_WEIGHT,
                  "weather_variables": ["TMP", "REH", "WSD", "POP"],
                  "regions": ["Gangneung", "Seoul", "Daejeon", "Daegu", "Gwangju", "Busan"]},
        "training_vintage_rule": "For every training/evaluation target, each weather issue and actual first receipt must precede that target's D-3 23:00 origin.",
        "training_minimum_days": MIN_TRAIN_DAYS,
        "ev_conditions": baseline.get("ev_conditions"),
        "records": [], "exclusions": [],
        "evaluation_note": "No model score is computed unless complete, origin-eligible forecast vintages and pre-origin training labels exist.",
    }
    if not weather_records:
        result["status"] = "blocked_no_forecast_vintages"
        result["exclusions"] = [{"target_date": row["target_date"],
            "reason": "no archived KMA short-forecast vintage received by that D-3 origin"}
            for row in baseline["records"]]
        result["paired_comparison"] = {"smp_only_days": 316, "weather_hgb_days": 0,
                                       "common_days": 0, "metrics": None}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    prices = _read_prices(source)
    baseline_by_day = {r["target_date"]: r for r in baseline["records"]}
    checkpoints: dict[str, dict] = {}
    candidate = []
    for saved in baseline["records"]:
        day = saved["target_date"]
        try:
            test_frame, origin = target_feature_frame(prices, weather_index, day)
            week_key = origin.strftime("%G-W%V")
            if week_key not in checkpoints:
                monday = origin.normalize() - pd.Timedelta(days=origin.dayofweek)
                X_train, y_train = training_examples(prices, weather_index, monday.strftime("%Y-%m-%d"))
                training_days = int(len(X_train) / 24)
                if training_days < MIN_TRAIN_DAYS:
                    raise ValueError(f"only {training_days} complete vintage training days; requires {MIN_TRAIN_DAYS}")
                if not np.isfinite(X_train.to_numpy(float)).all() or not np.isfinite(y_train.to_numpy(float)).all():
                    raise ValueError("invalid training matrix")
                model = make_model().fit(X_train.to_numpy(float), y_train.to_numpy(float))
                model_sha = hashlib.sha256(pickle.dumps(model, protocol=4)).hexdigest()
                fold_dir = ROOT / "research/smp_weather/models/folds"
                fold_dir.mkdir(parents=True, exist_ok=True)
                fold_path = fold_dir / f"d3_weather_{week_key}.joblib"
                import joblib
                joblib.dump({"model": model, "features": list(WEATHER_MODEL_FEATURES),
                    "horizon_days": 3, "anchor_weight": ANCHOR_WEIGHT,
                    "trained_through": (monday - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    "trained_at_kst": monday.isoformat(), "training_days": training_days,
                    "training_rows": len(y_train), "model_sha256": model_sha,
                    "weather_vintage_rule": "issue and receipt both no later than each training target's D-3 origin",
                    "model_family": "research-only SMP + KMA forecast HGB"}, fold_path, compress=3)
                checkpoints[week_key] = {"model": model, "sha256": model_sha,
                    "train_rows": len(y_train), "train_days": training_days,
                    "trained_through": (monday - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    "trained_at_kst": monday.isoformat(),
                    "checkpoint": str(fold_path.relative_to(ROOT)).replace("\\", "/")}
            checkpoint = checkpoints[week_key]
            # The immutable SMP-only artifact supplies the common seasonal anchor.
            anchor = np.asarray(saved["baseline_predictions"], dtype=float)
            base = test_frame.daily.to_numpy(float) + checkpoint["model"].predict(
                test_frame.loc[:, WEATHER_MODEL_FEATURES].to_numpy(float))
            pred = 0.95 * base + ANCHOR_WEIGHT * anchor
            actual = np.asarray(saved["actual_smp"], dtype=float)
            if pred.shape != (24,) or not np.isfinite(pred).all():
                raise ValueError("candidate prediction is not a finite 24-hour vector")
            candidate.append({"target_date": day, "forecast_origin_kst": origin.isoformat(),
                "trained_through": checkpoint["trained_through"],
                "model_trained_at_kst": checkpoint["trained_at_kst"],
                "checkpoint": checkpoint["checkpoint"],
                "model_sha256": checkpoint["sha256"], "training_days": checkpoint["train_days"],
                "training_rows": checkpoint["train_rows"],
                "predictions": np.round(pred, 6).tolist(),
                "baseline_predictions": saved["predictions"], "actual_smp": saved["actual_smp"],
                "metrics": _record_metrics(pred, actual, np.asarray(saved["predictions"], float)),
                "cohort": "2026 historical replay; not operational issuance"})
        except Exception as exc:
            result["exclusions"].append({"target_date": day,
                "reason": str(exc) if isinstance(exc, ValueError) else type(exc).__name__})

    result["records"] = candidate
    result["status"] = "evaluated" if candidate else "blocked_insufficient_vintage_training_data"
    common = sorted(set(baseline_by_day) & {r["target_date"] for r in candidate})
    base_common = [baseline_by_day[d] for d in common]
    weather_common = [next(r for r in candidate if r["target_date"] == d) for d in common]
    base_scored = [{"target_date": b["target_date"], "predictions": b["predictions"],
        "actual_smp": b["actual_smp"], "metrics": _record_metrics(
            np.asarray(b["predictions"], float), np.asarray(b["actual_smp"], float),
            np.asarray(b["predictions"], float))} for b in base_common]
    result["paired_comparison"] = {
        "smp_only_days": len(baseline["records"]), "weather_hgb_days": len(candidate),
        "common_days": len(common), "common_target_dates": common,
        "smp_only": _aggregate(base_scored),
        "smp_weather_hgb": _aggregate(weather_common),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    summary = run()
    print(json.dumps({k: summary[k] for k in ("status", "weather_record_count", "paired_comparison", "exclusions")},
                     ensure_ascii=False, indent=2))
