"""Time-ordered D+3 research replay for 2026; never touches the issue gate.

Historical SMP values are filtered by their interval timestamp at every fold.
The source DB's first_seen vintage before 2026-10-15 is not verified, therefore
the resulting forecasts are explicitly retrospective replays, not operational
predictions.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("SMP_SOURCE", ROOT.parent / "gradproject")).resolve()
sys.path.insert(0, str(SOURCE))
from v11.store import connect, latest
from v17.features import FEATURES, build
try:
    from .publish_forecast import seasonal_anchor
except ImportError:  # direct script execution from the automation runner
    from publish_forecast import seasonal_anchor

KST = "Asia/Seoul"
OUT = ROOT / "docs" / "live" / "walkforward_d3_2026.json.gz"
EV = {"battery_kwh": 60.0, "soc_start_pct": 30.0, "soc_goal_pct": 80.0,
      "charge_kw": 7.0, "efficiency": 0.9, "start_hour": 0, "end_hour": 24}


def _vector(prices: pd.DataFrame, day: str) -> np.ndarray | None:
    rows = prices[prices.timestamp.dt.strftime("%Y-%m-%d") == day].sort_values("timestamp")
    if len(rows) != 24 or rows.timestamp.dt.hour.tolist() != list(range(24)):
        return None
    values = rows.smp.to_numpy(float)
    return values if np.isfinite(values).all() else None


def _charge_plan(prices: np.ndarray) -> tuple[list[int], float] | None:
    need = max(0.0, (EV["soc_goal_pct"] - EV["soc_start_pct"]) * EV["battery_kwh"] / 100.0)
    left, cost, hours = need, 0.0, []
    for hour in sorted(range(EV["start_hour"], EV["end_hour"]), key=lambda h: (float(prices[h]), h)):
        if left <= 1e-9:
            break
        grid_kwh = min(EV["charge_kw"], left / EV["efficiency"])
        cost += grid_kwh * float(prices[hour])
        left -= grid_kwh * EV["efficiency"]
        hours.append(hour)
    return (hours, float(cost)) if left <= 1e-6 else None


def _metrics(pred: np.ndarray, actual: np.ndarray, baseline: np.ndarray) -> dict:
    err = actual - pred
    pred_plan, base_plan, oracle = _charge_plan(pred), _charge_plan(baseline), _charge_plan(actual)
    if not pred_plan or not base_plan or not oracle:
        raise ValueError("EV 충전 에너지 조건을 충족할 수 없습니다.")
    peak_hour = int(np.argmax(actual))
    base_error = actual - baseline
    return {
        "n_hours": 24,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "hourly_signed_error": [round(float(x), 6) for x in (pred - actual)],
        "baseline_mae": float(np.mean(np.abs(base_error))),
        "baseline_rmse": float(np.sqrt(np.mean(base_error ** 2))),
        "peak_actual": float(actual[peak_hour]), "peak_actual_hour": peak_hour,
        "peak_prediction": float(pred[peak_hour]),
        "peak_price_error": float(pred[peak_hour] - actual[peak_hour]),
        "peak_time_error_hours": abs(int(np.argmax(pred)) - peak_hour),
        "g2v_regret_won": float(max(0.0, pred_plan[1] - oracle[1])),
        "baseline_g2v_regret_won": float(max(0.0, base_plan[1] - oracle[1])),
        "predicted_charge_hours": pred_plan[0], "baseline_charge_hours": base_plan[0],
        "oracle_charge_hours": oracle[0], "predicted_plan_cost_won": pred_plan[1],
        "baseline_plan_cost_won": base_plan[1], "oracle_plan_cost_won": oracle[1],
        "ev_conditions": EV,
    }


def validate_payload(payload: dict, atol: float = 1e-5) -> None:
    """Integrity checks used by the build and tests to re-derive saved scores."""
    seen = set()
    for row in payload.get("records", []):
        day = row["target_date"]
        if day in seen:
            raise AssertionError(f"duplicate target day in daily D+3 cohort: {day}")
        seen.add(day)
        pred = np.asarray(row["predictions"], dtype=float)
        actual = np.asarray(row["actual_smp"], dtype=float)
        baseline = np.asarray(row["baseline_predictions"], dtype=float)
        if row.get("hours_kst") != list(range(24)):
            raise AssertionError(f"hour labels are missing or misordered: {day}")
        if pred.shape != (24,) or actual.shape != (24,) or baseline.shape != (24,):
            raise AssertionError(f"non-24-hour vector: {day}")
        if not all(np.isfinite(x).all() for x in (pred, actual, baseline)):
            raise AssertionError(f"non-finite SMP vector: {day}")
        origin = pd.Timestamp(row["forecast_origin_kst"])
        trained_at = pd.Timestamp(row["model_trained_at_kst"])
        if pd.Timestamp(day, tz=KST) - origin.normalize() != pd.Timedelta(days=3):
            raise AssertionError(f"origin is not D+3: {day}")
        if pd.Timestamp(row["input_max_timestamp_kst"]) > origin:
            raise AssertionError(f"future SMP input timestamp: {day}")
        if trained_at > origin or pd.Timestamp(row["trained_through"], tz=KST) >= trained_at.normalize():
            raise AssertionError(f"model training cutoff is not before origin: {day}")
        e = actual - pred
        be = actual - baseline
        m = row["metrics"]
        if not np.isclose(float(np.mean(np.abs(e))), m["mae"], atol=atol):
            raise AssertionError(f"MAE mismatch: {day}")
        if not np.isclose(float(np.sqrt(np.mean(e ** 2))), m["rmse"], atol=atol):
            raise AssertionError(f"RMSE mismatch: {day}")
        if not np.allclose(-e, m["hourly_signed_error"], atol=atol):
            raise AssertionError(f"hourly error mismatch: {day}")
        recomputed = _metrics(pred, actual, baseline)
        for key in ("g2v_regret_won", "baseline_g2v_regret_won"):
            if not np.isclose(recomputed[key], m[key], atol=atol):
                raise AssertionError(f"G2V regret mismatch ({key}): {day}")
        if not np.isclose(float(np.mean(np.abs(be))), m["baseline_mae"], atol=atol):
            raise AssertionError(f"baseline MAE mismatch: {day}")
        if not np.isclose(float(np.sqrt(np.mean(be ** 2))), m["baseline_rmse"], atol=atol):
            raise AssertionError(f"baseline RMSE mismatch: {day}")


def _fit_checkpoint(prices: pd.DataFrame, train_asof: pd.Timestamp) -> dict:
    training_prices = prices[prices.timestamp <= train_asof].copy().sort_values("timestamp")
    training_prices = training_prices.drop_duplicates("timestamp", keep="last")
    train_frame = build(training_prices, days_ahead=3, include_labels=True)
    train_frame = train_frame[train_frame.target_date < train_asof.strftime("%Y-%m-%d")]
    if train_frame.empty:
        raise ValueError("예측 원점 이전 학습 표본이 없습니다.")
    train_end = train_frame.target_date.max()
    if pd.Timestamp(train_end, tz=KST) >= train_asof.normalize():
        raise AssertionError("학습 대상일이 모델 학습 기준일 당일 또는 이후입니다.")
    model = HistGradientBoostingRegressor(max_iter=250, max_leaf_nodes=31,
            learning_rate=0.05, early_stopping=False, l2_regularization=2.0,
            random_state=42)
    x_train = train_frame[FEATURES].to_numpy(float)
    y_train = (train_frame.actual_smp - train_frame.daily).to_numpy(float)
    if not np.isfinite(x_train).all() or not np.isfinite(y_train).all():
        raise ValueError("학습 데이터에 비정상 값이 있습니다.")
    model.fit(x_train, y_train)
    return {"model": model, "train_start": str(train_frame.target_date.min()),
            "train_end": str(train_end), "train_rows": int(len(train_frame)),
            "model_trained_at_kst": train_asof.isoformat(),
            "model_sha256": hashlib.sha256(pickle.dumps(model, protocol=4)).hexdigest()}


def _forecast_fold(prices: pd.DataFrame, target_day: str, checkpoint: dict | None = None) -> dict:
    target_date = pd.Timestamp(target_day, tz=KST)
    origin = target_date - pd.Timedelta(days=3) + pd.Timedelta(hours=23)
    # Training labels end on the calendar day before the forecast origin. The
    # frame is built from a truncated series so no later SMP can affect X or y.
    known = prices[prices.timestamp <= origin].copy().sort_values("timestamp")
    known = known.drop_duplicates("timestamp", keep="last")
    if checkpoint is None:
        checkpoint = _fit_checkpoint(known, origin.normalize())
    trained_at = pd.Timestamp(checkpoint["model_trained_at_kst"])
    if trained_at.tzinfo is None:
        trained_at = trained_at.tz_localize(KST)
    if trained_at > origin:
        raise AssertionError("모델 학습 기준시각이 예측 원점 이후입니다.")
    target_hours = pd.date_range(target_date, target_date + pd.Timedelta(hours=23), freq="h")
    pred_frame = build(known, targets=target_hours, days_ahead=3, include_labels=False)
    if len(pred_frame) != 24 or pred_frame.timestamp.dt.hour.tolist() != list(range(24)):
        raise ValueError("D+3 예측 피처가 24시간 완전하지 않습니다.")
    if (pred_frame.origin_timestamp > origin).any() or known.timestamp.max() > origin:
        raise AssertionError("예측 입력에 원점 이후 SMP가 포함됐습니다.")
    base = pred_frame.daily.to_numpy(float) + checkpoint["model"].predict(pred_frame[FEATURES].to_numpy(float))
    anchor = seasonal_anchor(known, target_hours)
    prediction = 0.95 * base + 0.05 * anchor
    if not np.isfinite(prediction).all():
        raise ValueError("예측값에 NaN 또는 무한대가 있습니다.")
    return {"prediction": prediction, "anchor": anchor,
            "train_start": checkpoint["train_start"], "train_end": checkpoint["train_end"],
            "train_rows": checkpoint["train_rows"],
            "model_trained_at_kst": checkpoint["model_trained_at_kst"],
            "input_max_timestamp": known.timestamp.max().isoformat(),
            "model_sha256": checkpoint["model_sha256"]}


def run(source: Path = SOURCE, out: Path = OUT, now: pd.Timestamp | None = None) -> dict:
    now = pd.Timestamp.now(tz=KST) if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize(KST)
    else:
        now = now.tz_convert(KST)
    db = connect(source / "artifacts" / "v11")
    try:
        prices = latest(db, "land").sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    finally:
        db.close()
    prices = prices[prices.smp.map(np.isfinite)].copy()
    # Stop at the last fully elapsed target date. Future official prices that
    # are already published are not treated as realized actuals.
    last_elapsed = (now.normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    dates = sorted(prices.timestamp.dt.strftime("%Y-%m-%d").unique())
    targets = [d for d in dates if d.startswith("2026-") and d <= last_elapsed]
    records, exclusions, checkpoints = [], [], {}
    for day in targets:
        actual = _vector(prices, day)
        if actual is None:
            exclusions.append({"target_date": day, "reason": "official SMP 24시간 자료가 없거나 시간 인덱스가 불완전함"})
            continue
        target_date = pd.Timestamp(day, tz=KST)
        origin = target_date - pd.Timedelta(days=3) + pd.Timedelta(hours=23)
        # Historical outcome prices are used only after predictions are fixed.
        asof = prices[prices.timestamp <= origin]
        try:
            # Fixed HGB settings; one checkpoint per ISO week, trained Monday
            # 00:00 KST on completed prior-day labels. It predates every 23:00
            # forecast origin in that week and avoids repeated weekly refits.
            week_key = origin.strftime("%G-W%V")
            if week_key not in checkpoints:
                train_asof = origin.normalize() - pd.Timedelta(days=origin.dayofweek)
                checkpoints[week_key] = _fit_checkpoint(asof, train_asof)
            result = _forecast_fold(asof, day, checkpoints[week_key])
            metrics = _metrics(result["prediction"], actual, result["anchor"])
        except Exception as exc:
            exclusions.append({"target_date": day, "origin_kst": origin.isoformat(),
                               "reason": f"{type(exc).__name__}: {exc}"})
            continue
        rows = prices[prices.timestamp.dt.strftime("%Y-%m-%d") == day]
        first_seen = pd.to_datetime(rows.first_seen, utc=True)
        records.append({
            "record_id": f"historical_replay|2026-D3|{day}|{result['model_sha256']}",
            "record_type": "historical_replay", "evaluation_cohort": "2026 D+3 time-ordered historical replay",
            "target_date": day, "original_horizon_days": 3, "horizon_days": 3,
            "hours_kst": list(range(24)),
            "forecast_origin_kst": origin.isoformat(), "generated_at_kst": None,
            "trained_from": result["train_start"], "trained_through": result["train_end"],
            "model_trained_at_kst": result["model_trained_at_kst"],
            "training_rows": result["train_rows"], "model_family": "HGB residual + 7/14/21/28-day seasonal anchor",
            "model_sha256": result["model_sha256"], "predictions": np.round(result["prediction"], 6).tolist(),
            "baseline_predictions": np.round(result["anchor"], 6).tolist(), "actual_smp": np.round(actual, 6).tolist(),
            "actual_label": "EPSIS land SMP · target day elapsed; settlement finality not independently verified",
            "actual_source": "EPSIS land SMP (saved source DB)",
            "actual_first_seen_min_utc": first_seen.min().isoformat(),
            "actual_first_seen_max_utc": first_seen.max().isoformat(),
            "input_max_timestamp_kst": result["input_max_timestamp"], "input_vintage_verified": False,
            "evaluation_type": "2026 timestamp-filtered historical replay; not an operational issue",
            "prospective_forecast_eligible": False, "metrics": metrics,
            "note": "Training labels and forecast features were timestamp-truncated at the forecast origin. Historical first_seen/vintage before 2026-10-15 is not verified; this is not evidence the data was received then.",
        })
    if records:
        errs = np.concatenate([np.asarray(r["metrics"]["hourly_signed_error"]) for r in records])
        baseline_errs = np.concatenate([np.asarray(r["baseline_predictions"]) - np.asarray(r["actual_smp"]) for r in records])
        summary = {"successful_days": len(records), "mae": float(np.mean(np.abs(errs))),
                   "rmse": float(np.sqrt(np.mean(errs ** 2))),
                   "baseline_mae": float(np.mean(np.abs(baseline_errs))),
                   "baseline_rmse": float(np.sqrt(np.mean(baseline_errs ** 2))),
                   "mean_daily_g2v_regret_won": float(np.mean([r["metrics"]["g2v_regret_won"] for r in records])),
                   "baseline_mean_daily_g2v_regret_won": float(np.mean([r["metrics"]["baseline_g2v_regret_won"] for r in records])),
                   "first_target_date": records[0]["target_date"], "last_target_date": records[-1]["target_date"]}
    else:
        summary = {"successful_days": 0, "mae": None, "rmse": None, "mean_daily_g2v_regret_won": None}
    payload = {"schema_version": 1, "evaluation_cohort": "2026 D+3 time-ordered historical replay",
               "horizon_days": 3, "model_family": "HGB residual + 7/14/21/28-day seasonal anchor",
               "algorithm": {"estimator": "HistGradientBoostingRegressor", "max_iter": 250,
                             "max_leaf_nodes": 31, "learning_rate": 0.05,
                             "early_stopping": False, "l2_regularization": 2.0,
                             "random_state": 42, "anchor_weight": 0.05},
               "source": "EPSIS land SMP saved source DB", "bulk_backfill_first_seen_utc": "2026-10-15T12:59:47Z",
               "historical_receipt_vintage_verified": False,
               "training_schedule": "fixed-parameter expanding-window HGB refit once per ISO week at Monday 00:00 KST; labels end before Monday",
               "software_versions": {"python": __import__("platform").python_version(),
                                     "scikit_learn": sklearn.__version__, "pandas": pd.__version__,
                                     "numpy": np.__version__},
               "prediction_availability_claim": "Historical timestamp-causal replay only; not verified operational availability",
               "actual_price_status": "last fully elapsed target date only; final settlement status not independently verified",
               "ev_conditions": EV, "summary": summary, "records": records,
               "excluded_target_count": len(exclusions), "exclusions": exclusions}
    validate_payload(payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    import gzip
    with gzip.open(out, "wt", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    return payload


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "excluded_target_count": result["excluded_target_count"],
                      "exclusions": result["exclusions"]}, ensure_ascii=False, indent=2))
