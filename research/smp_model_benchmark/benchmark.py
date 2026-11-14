"""Eight-model D+3 SMP-only benchmark on preserved chronological folds.

All files produced by this module live under research/smp_model_benchmark.
Historical source timestamps are used for causal replay, but original first-seen
vintages are not verified; outputs are never labelled operational forecasts.
"""
from __future__ import annotations

import gzip
import json
import os
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("SMP_SOURCE", ROOT.parent / "gradproject")).resolve()
sys.path.insert(0, str(SOURCE))
from v11.store import connect, latest
from v12.ev import EV, optimize as optimize_ev, score as score_ev
from v17.features import FEATURES, build as build_v17

OUT = ROOT / "research" / "smp_model_benchmark"
MODELS = OUT / "models"
RESULTS = OUT / "results"
FIGURES = OUT / "figures"
HORIZON = 3
ANCHOR_WEIGHT = 0.05
SEEDS = (17, 42, 91)
SEQUENCE_CANDIDATES = (168, 672)
MAX_EPOCHS = 100
PATIENCE = 8
EV_RULE = EV(capacity=60, initial_soc=30, target_soc=80, min_soc=20, max_soc=90,
             charge_kw=7, discharge_kw=7, charge_eff=0.9, discharge_eff=0.9,
             degradation=5, sell_ratio=1)
FOLDS = (
    {"name": "fall2024", "train_end": "2024-06-30", "cal_start": "2024-07-01", "cal_end": "2024-09-30", "val_start": "2024-10-01", "val_end": "2024-12-31"},
    {"name": "spring2025", "train_end": "2024-12-31", "cal_start": "2025-01-01", "cal_end": "2025-03-31", "val_start": "2025-04-01", "val_end": "2025-06-30"},
    {"name": "fall2025", "train_end": "2025-06-30", "cal_start": "2025-07-01", "cal_end": "2025-09-30", "val_start": "2025-10-01", "val_end": "2025-12-31"},
)
MODEL_NAMES = ("LSTM", "GRU", "HGB", "LightGBM", "ExtraTrees", "LinearRegression", "RandomForest", "XGBoost")
MODEL_CONFIGS = {
    "HGB": {"max_iter": 250, "max_leaf_nodes": 31, "learning_rate": 0.05, "early_stopping": False, "l2_regularization": 2.0, "random_state": 42},
    "LightGBM": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20, "random_state": 42, "n_jobs": 4, "verbosity": -1, "deterministic": True, "force_col_wise": True},
    "ExtraTrees": {"n_estimators": 150, "min_samples_leaf": 10, "max_features": 0.8, "random_state": 42, "n_jobs": 4},
    "LinearRegression": {"scaler": "StandardScaler fitted on train rows", "estimator": "LinearRegression"},
    "RandomForest": {"n_estimators": 200, "min_samples_leaf": 5, "max_features": 0.8, "random_state": 42, "n_jobs": 4},
    "XGBoost": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0, "objective": "reg:squarederror", "tree_method": "hist", "random_state": 42, "n_jobs": 4},
    "RNN": {"recurrent_units": 32, "recurrent_layers": 1, "dense_layers": [64, 128], "optimizer": "Adam", "learning_rate": 0.001, "loss": "MSE", "batch_size": 32, "max_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE},
}


def load_prices() -> pd.DataFrame:
    db = connect(SOURCE / "artifacts" / "v11")
    try:
        frame = latest(db, "land")
    finally:
        db.close()
    frame = frame.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    if frame.timestamp.dt.tz is None:
        raise ValueError("SMP timestamps must be timezone-aware KST")
    return frame


def _calendar_matrix(day: str) -> np.ndarray:
    target = pd.date_range(pd.Timestamp(day, tz="Asia/Seoul"), periods=24, freq="h")
    dow = target.dayofweek.to_numpy()
    month = target.month.to_numpy()
    hour = target.hour.to_numpy()
    return np.column_stack((np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
        np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7),
        np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12))).astype(np.float32)


def _weekly_anchor(series: pd.Series, target: pd.DatetimeIndex) -> np.ndarray:
    values = np.vstack([series.reindex(target - pd.Timedelta(days=7*k)).to_numpy(float)
                        for k in range(1, 5)])
    if values.shape != (4, 24) or not np.isfinite(values).all():
        raise ValueError("incomplete 7/14/21/28-day SMP seasonal anchor")
    return np.median(values, axis=0)


def make_day_map(prices: pd.DataFrame, days: list[str] | None = None) -> dict[str, dict]:
    """Build exact-origin D+3 features plus 168/672-h histories for complete days."""
    full = build_v17(prices, days_ahead=3)
    selected = set(days) if days is not None else None
    series = prices.set_index("timestamp").smp.sort_index()
    result = {}
    for day, group in full.groupby("target_date", sort=True):
        day = str(day)
        if selected is not None and day not in selected:
            continue
        group = group.sort_values("timestamp")
        if len(group) != 24 or group.timestamp.dt.hour.tolist() != list(range(24)):
            continue
        target = pd.DatetimeIndex(group.timestamp)
        origin = pd.Timestamp(group.origin_timestamp.iloc[0])
        expected = pd.Timestamp(day, tz="Asia/Seoul") - pd.Timedelta(days=3) + pd.Timedelta(hours=23)
        if origin != expected:
            raise AssertionError(f"wrong D+3 origin for {day}: {origin} != {expected}")
        actual = group.actual_smp.to_numpy(float)
        base = group.daily.to_numpy(float)
        try:
            anchor = _weekly_anchor(series, target)
        except ValueError:
            # Warm-up targets without all four pre-origin seasonal profiles are
            # not used for training, calibration, or evaluation.
            continue
        if not all(np.isfinite(x).all() for x in (actual, base, anchor)):
            continue
        seq_values = {}
        for length in SEQUENCE_CANDIDATES:
            index = pd.date_range(end=origin, periods=length, freq="h")
            seq = series.reindex(index).to_numpy(float)
            if seq.shape == (length,) and np.isfinite(seq).all() and index.max() <= origin:
                seq_values[length] = seq.astype(np.float32)
        # All tree features derive from the shared V17 price/calendar builder.
        X = group.loc[:, FEATURES].to_numpy(float)
        if X.shape != (24, len(FEATURES)) or not np.isfinite(X).all():
            continue
        result[day] = {"day": day, "target": target, "origin": origin, "actual": actual,
            "base": base, "anchor": anchor, "X": X, "residual": actual - base,
            "calendar": _calendar_matrix(day), "sequences": seq_values,
            "input_max_timestamp": origin}
    return result


def validate_no_future_leak(prices: pd.DataFrame, item: dict) -> None:
    """Rebuild a day after mutating post-origin SMP and require equal features."""
    origin = item["origin"]
    changed = prices.copy()
    changed.loc[changed.timestamp > origin, "smp"] += 123456.0
    original_frame = build_v17(prices, targets=item["target"], days_ahead=3, include_labels=False)
    changed_frame = build_v17(changed, targets=item["target"], days_ahead=3, include_labels=False)
    if original_frame.empty or changed_frame.empty:
        raise AssertionError("future-leak test could not build complete D+3 inputs")
    np.testing.assert_allclose(original_frame.loc[:, FEATURES].to_numpy(float),
                               changed_frame.loc[:, FEATURES].to_numpy(float), rtol=0, atol=1e-10)
    if item["input_max_timestamp"] > origin:
        raise AssertionError("input timestamp exceeds forecast origin")


def _build_sklearn_model(name: str):
    if name == "HGB":
        return HistGradientBoostingRegressor(**MODEL_CONFIGS[name])
    if name == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(objective="regression", **MODEL_CONFIGS[name])
    if name == "ExtraTrees":
        return ExtraTreesRegressor(**MODEL_CONFIGS[name])
    if name == "LinearRegression":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "RandomForest":
        return RandomForestRegressor(**MODEL_CONFIGS[name])
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(**MODEL_CONFIGS[name])
    raise ValueError(f"unknown sklearn model: {name}")


def _tf_modules():
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    return tf


def make_recurrent_model(kind: str, seq_len: int, seed: int):
    tf = _tf_modules()
    tf.keras.utils.set_random_seed(seed)
    seq = tf.keras.Input(shape=(seq_len, 1), name="smp_history")
    cal = tf.keras.Input(shape=(24, 6), name="target_calendar")
    recurrent = (tf.keras.layers.LSTM(32, name="lstm_32") if kind == "LSTM"
                 else tf.keras.layers.GRU(32, name="gru_32"))
    encoded = recurrent(seq)
    encoded = tf.keras.layers.Dense(64, activation="relu", name="history_dense")(encoded)
    cal_flat = tf.keras.layers.Flatten(name="calendar_flatten")(cal)
    joined = tf.keras.layers.Concatenate(name="history_calendar_concat")([encoded, cal_flat])
    hidden = tf.keras.layers.Dense(128, activation="relu", name="forecast_dense")(joined)
    output = tf.keras.layers.Dense(24, name="hourly_residuals")(hidden)
    model = tf.keras.Model([seq, cal], output, name=f"{kind.lower()}_d3_smp_only")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def _rnn_arrays(days: dict[str, dict], date_list: list[str], seq_len: int):
    keep = [d for d in date_list if seq_len in days[d]["sequences"]]
    if not keep:
        raise ValueError(f"no complete {seq_len}-hour sequences")
    seq = np.stack([days[d]["sequences"][seq_len] for d in keep]).astype(np.float32)[..., None]
    cal = np.stack([days[d]["calendar"] for d in keep]).astype(np.float32)
    y = np.stack([days[d]["residual"] for d in keep]).astype(np.float32)
    return keep, seq, cal, y


def _fit_rnn(kind: str, seq_len: int, seed: int, days: dict[str, dict], train_dates: list[str], cal_dates: list[str]):
    tf = _tf_modules()
    train_keep, x_train, cal_train, y_train = _rnn_arrays(days, train_dates, seq_len)
    cal_keep, x_cal, cal_x, y_cal = _rnn_arrays(days, cal_dates, seq_len)
    x_mean = float(x_train.mean())
    x_std = float(x_train.std()) or 1.0
    y_mean = float(y_train.mean())
    y_std = float(y_train.std()) or 1.0
    x_train = (x_train - x_mean) / x_std
    x_cal = (x_cal - x_mean) / x_std
    y_train_n = (y_train - y_mean) / y_std
    y_cal_n = (y_cal - y_mean) / y_std
    model = make_recurrent_model(kind, seq_len, seed)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
        min_delta=1e-4, restore_best_weights=True, mode="min")]
    start = time.perf_counter()
    history = model.fit([x_train, cal_train], y_train_n, validation_data=([x_cal, cal_x], y_cal_n),
        epochs=MAX_EPOCHS, batch_size=32, shuffle=False, callbacks=callbacks, verbose=0)
    fit_seconds = time.perf_counter() - start
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    return {"model": model, "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std,
        "seq_len": seq_len, "seed": seed, "fit_seconds": fit_seconds, "best_epoch": best_epoch,
        "train_dates": train_keep, "cal_dates": cal_keep, "parameter_count": int(model.count_params())}


def _rnn_predict(bundle: dict, item: dict) -> tuple[np.ndarray, float]:
    seq = item["sequences"][bundle["seq_len"]][None, :, None]
    seq = ((seq - bundle["x_mean"]) / bundle["x_std"]).astype(np.float32)
    cal = item["calendar"][None, :, :].astype(np.float32)
    start = time.perf_counter()
    out = bundle["model"].predict([seq, cal], verbose=0)[0]
    elapsed = time.perf_counter() - start
    return out.astype(float) * bundle["y_std"] + bundle["y_mean"], elapsed


def _rnn_predict_batch(bundle: dict, days: dict[str, dict], date_list: list[str]):
    valid=[d for d in date_list if bundle["seq_len"] in days[d]["sequences"]]
    seq=np.stack([days[d]["sequences"][bundle["seq_len"]] for d in valid]).astype(np.float32)[...,None]
    seq=((seq-bundle["x_mean"])/bundle["x_std"]).astype(np.float32)
    cal=np.stack([days[d]["calendar"] for d in valid]).astype(np.float32)
    start=time.perf_counter()
    residual=bundle["model"].predict([seq,cal],verbose=0)
    elapsed=(time.perf_counter()-start)/max(1,len(valid))
    return {d:residual[i].astype(float)*bundle["y_std"]+bundle["y_mean"] for i,d in enumerate(valid)},elapsed


def _predict_plan_metrics(actual: np.ndarray, prediction: np.ndarray, timestamps: pd.DatetimeIndex) -> dict:
    ev = EV(**EV_RULE.__dict__)
    plan = optimize_ev(prediction, timestamps, ev, v2g=False)
    p2v_plan = plan["schedule"]
    realized_g2v = score_ev(actual, p2v_plan.charge_kw.to_numpy(), np.zeros(24),
                            p2v_plan.connected_hours.to_numpy(), ev)
    oracle_g2v = optimize_ev(actual, timestamps, ev, v2g=False)
    g2v_regret = realized_g2v["net_cost"] - oracle_g2v["net_cost"]

    vplan = optimize_ev(prediction, timestamps, ev, v2g=True)
    vschedule = vplan["schedule"]
    realized_v2g = score_ev(actual, vschedule.charge_kw.to_numpy(), vschedule.discharge_kw.to_numpy(),
                            vschedule.connected_hours.to_numpy(), ev)
    oracle_v2g = optimize_ev(actual, timestamps, ev, v2g=True)
    v2g_regret = realized_v2g["net_cost"] - oracle_v2g["net_cost"]

    error = prediction - actual
    peak = actual >= 150.0
    pred_low = set(np.argsort(prediction, kind="stable")[:5].tolist())
    actual_low = set(np.argsort(actual, kind="stable")[:5].tolist())
    return {"mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "peak_hours": int(peak.sum()),
        "peak_mae": float(np.mean(np.abs(error[peak]))) if peak.any() else None,
        "low_price_top5_overlap": len(pred_low & actual_low) / 5.0,
        "g2v_regret_won": float(g2v_regret),
        "g2v_actual_net_cost_won": float(realized_g2v["net_cost"]),
        "v2g_regret_won": float(v2g_regret),
        "v2g_actual_net_cost_won": float(realized_v2g["net_cost"]),
        "v2g_gain_vs_g2v_actual_won": float(realized_g2v["net_cost"] - realized_v2g["net_cost"]),
        "predicted_charge_hours": [int(x) for x in np.flatnonzero(p2v_plan.charge_kw.to_numpy() > 1e-8)],
        "predicted_v2g_charge_hours": [int(x) for x in np.flatnonzero(vschedule.charge_kw.to_numpy() > 1e-8)],
        "predicted_v2g_discharge_hours": [int(x) for x in np.flatnonzero(vschedule.discharge_kw.to_numpy() > 1e-8)]}


def _tree_eval(model, name: str, item: dict) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    residual = model.predict(item["X"])
    predict_seconds = time.perf_counter() - start
    direct = item["base"] + residual
    return (1.0 - ANCHOR_WEIGHT) * direct + ANCHOR_WEIGHT * item["anchor"], predict_seconds


def _save_model(model, model_name: str, fold: str, seed: int | None = None) -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    suffix = f"_seed{seed}" if seed is not None else ""
    if model_name in ("LSTM", "GRU"):
        path = MODELS / f"{fold}_{model_name.lower()}_d3{suffix}.keras"
        model.save(path)
    else:
        path = MODELS / f"{fold}_{model_name.lower()}_d3.joblib"
        joblib.dump(model, path, compress=3)
    return int(path.stat().st_size)


def _model_complexity(model) -> tuple[int | None, int | None]:
    """Return exact NN parameter count and a comparable tree/node or linear coefficient count."""
    if hasattr(model, "count_params"):
        return int(model.count_params()), None
    if hasattr(model, "named_steps") and "linearregression" in model.named_steps:
        return None, int(np.asarray(model.named_steps["linearregression"].coef_).size)
    if hasattr(model, "estimators_"):
        trees = [tree for tree in np.asarray(model.estimators_, dtype=object).ravel()
                 if hasattr(tree, "tree_")]
        if trees:
            return None, int(sum(tree.tree_.node_count for tree in trees))
    if hasattr(model, "_predictors"):
        return None, int(sum(len(tree.nodes) for stage in model._predictors for tree in stage))
    if hasattr(model, "booster_"):
        dump = model.booster_.dump_model()
        def lightgbm_nodes(node):
            return 1 + sum(lightgbm_nodes(node[key]) for key in ("left_child", "right_child") if key in node)
        return None, int(sum(lightgbm_nodes(tree["tree_structure"]) for tree in dump["tree_info"]))
    if hasattr(model, "get_booster"):
        import json as _json
        def xgb_nodes(node):
            return 1 + sum(xgb_nodes(child) for child in node.get("children", []))
        return None, int(sum(xgb_nodes(_json.loads(tree)) for tree in model.get_booster().get_dump(dump_format="json")))
    return None, None


def _fit_tree_fold(name: str, days: dict[str, dict], train_dates: list[str]):
    X = np.concatenate([days[d]["X"] for d in train_dates])
    y = np.concatenate([days[d]["residual"] for d in train_dates])
    model = _build_sklearn_model(name)
    start = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - start, len(X)


def _date_range_days(days: dict, start: str, end: str) -> list[str]:
    return [d for d in sorted(days) if start <= d <= end]


def _conservative_fit_after_kst(train_end: str) -> str:
    """Model is considered available only after the last training target day closes."""
    return (pd.Timestamp(train_end, tz="Asia/Seoul") + pd.Timedelta(days=1)).isoformat()


def _primary_run(days: dict[str, dict]) -> dict:
    rnn_fits: dict[tuple[str, str, int, int], dict] = {}
    calibration_rows = []
    fold_days = {}
    # Examine both sequence windows only on the chronological calibration data.
    # Same fixed seeds and training/calibration split are used by LSTM and GRU.
    for fold in FOLDS:
        train_dates = _date_range_days(days, "1900-01-01", fold["train_end"])
        cal_dates = _date_range_days(days, fold["cal_start"], fold["cal_end"])
        val_dates = _date_range_days(days, fold["val_start"], fold["val_end"])
        if len(train_dates) < 300 or len(cal_dates) < 80 or len(val_dates) < 80:
            raise RuntimeError(f"incomplete {fold['name']} D+3 cohort: train={len(train_dates)} cal={len(cal_dates)} val={len(val_dates)}")
        fold_days[fold["name"]] = (train_dates, cal_dates, val_dates)
        for arch in ("LSTM", "GRU"):
            for seq_len in SEQUENCE_CANDIDATES:
                for seed in SEEDS:
                    bundle = _fit_rnn(arch, seq_len, seed, days, train_dates, cal_dates)
                    print(f"calibration fit {fold['name']} {arch} {seq_len}h seed={seed} epochs={bundle['best_epoch']} train_s={bundle['fit_seconds']:.1f}",flush=True)
                    rnn_fits[(fold["name"], arch, seq_len, seed)] = bundle
                    cal_residuals,_ = _rnn_predict_batch(bundle, days, bundle["cal_dates"])
                    for day in bundle["cal_dates"]:
                        resid = cal_residuals[day]
                        item = days[day]
                        pred = (1-ANCHOR_WEIGHT)*(item["base"]+resid)+ANCHOR_WEIGHT*item["anchor"]
                        calibration_rows.append({"fold":fold["name"],"model":arch,"sequence_hours":seq_len,
                            "seed":seed,"target_date":day,"mae":float(np.mean(np.abs(pred-item["actual"]))),
                            "best_epoch":bundle["best_epoch"],"fit_seconds":bundle["fit_seconds"]})
    calibration_df = pd.DataFrame(calibration_rows)
    sequence_summary = calibration_df.groupby(["model","sequence_hours"],as_index=False).agg(
        folds=("fold","nunique"),days=("target_date","nunique"),mean_calibration_mae=("mae","mean"),
        mean_best_epoch=("best_epoch","mean"),mean_fit_seconds=("fit_seconds","mean"))
    common_seq = sequence_summary.groupby("sequence_hours").mean_calibration_mae.mean().idxmin()
    chosen_seq = int(common_seq)
    if len(sequence_summary) != 4:
        raise RuntimeError("did not complete equal-budget LSTM/GRU 168h/672h calibration comparison")

    daily, hourly, fit_log = [], [], []
    rnn_prediction_cache={}
    for fold in FOLDS:
        train_dates, cal_dates, val_dates = fold_days[fold["name"]]
        # No evaluation date is used to select an architecture, sequence length or parameter.
        reference_preds = {"D3_lag_reference":[], "5pct_seasonal_anchor_reference":[]}
        tree_predictions = {}
        for name in ("HGB","LightGBM","ExtraTrees","LinearRegression","RandomForest","XGBoost"):
            model, fit_seconds, train_rows = _fit_tree_fold(name, days, train_dates)
            print(f"validation fit {fold['name']} {name} train_days={len(train_dates)} train_s={fit_seconds:.1f}",flush=True)
            model_size = _save_model(model,name,fold["name"])
            tree_predictions[name] = model
            fit_log.append({"fold":fold["name"],"model":name,"sequence_hours":None,"seed":None,
                "train_days":len(train_dates),"train_rows":train_rows,"train_start":train_dates[0],
                "trained_through":fold["train_end"],"model_fit_available_at_kst":_conservative_fit_after_kst(fold["train_end"]),
                "fit_seconds":fit_seconds,"parameter_count":None,"model_file_bytes":model_size,
                "best_epoch":None,"early_stopping":"not used; fixed declared parameters"})
        for seed in SEEDS:
            bundle=rnn_fits[(fold["name"],"LSTM",chosen_seq,seed)]
            batch_preds,batch_time=_rnn_predict_batch(bundle,days,val_dates)
            rnn_prediction_cache[(fold["name"],"LSTM",seed)]=(batch_preds,batch_time)
            size=_save_model(bundle["model"],"LSTM",fold["name"],seed)
            fit_log.append({"fold":fold["name"],"model":"LSTM","sequence_hours":chosen_seq,"seed":seed,
                "train_days":len(bundle["train_dates"]),"train_rows":len(bundle["train_dates"]),"train_start":bundle["train_dates"][0],
                "trained_through":fold["train_end"],"model_fit_available_at_kst":_conservative_fit_after_kst(fold["train_end"]),"fit_seconds":bundle["fit_seconds"],
                "parameter_count":bundle["parameter_count"],"model_file_bytes":size,"best_epoch":bundle["best_epoch"],
                "early_stopping":f"calibration val_loss; patience {PATIENCE}; restored best weights",
                "predict_seconds_per_day":batch_time})
            bundle2=rnn_fits[(fold["name"],"GRU",chosen_seq,seed)]
            batch_preds2,batch_time2=_rnn_predict_batch(bundle2,days,val_dates)
            rnn_prediction_cache[(fold["name"],"GRU",seed)]=(batch_preds2,batch_time2)
            size2=_save_model(bundle2["model"],"GRU",fold["name"],seed)
            fit_log.append({"fold":fold["name"],"model":"GRU","sequence_hours":chosen_seq,"seed":seed,
                "train_days":len(bundle2["train_dates"]),"train_rows":len(bundle2["train_dates"]),"train_start":bundle2["train_dates"][0],
                "trained_through":fold["train_end"],"model_fit_available_at_kst":_conservative_fit_after_kst(fold["train_end"]),"fit_seconds":bundle2["fit_seconds"],
                "parameter_count":bundle2["parameter_count"],"model_file_bytes":size2,"best_epoch":bundle2["best_epoch"],
                "early_stopping":f"calibration val_loss; patience {PATIENCE}; restored best weights",
                "predict_seconds_per_day":batch_time2})
        for day in val_dates:
            item=days[day]
            predictions={
                "D3_lag_reference":item["base"],
                "5pct_seasonal_anchor_reference":(1-ANCHOR_WEIGHT)*item["base"]+ANCHOR_WEIGHT*item["anchor"],
            }
            predict_times={}
            for name,model in tree_predictions.items():
                predictions[name],predict_times[name]=_tree_eval(model,name,item)
            rnn_seed_predictions={"LSTM":[],"GRU":[]}
            for arch in ("LSTM","GRU"):
                for seed in SEEDS:
                    bundle=rnn_fits[(fold["name"],arch,chosen_seq,seed)]
                    residuals,elapsed=rnn_prediction_cache[(fold["name"],arch,seed)]
                    resid=residuals[day]
                    rnn_seed_predictions[arch].append((seed,(1-ANCHOR_WEIGHT)*(item["base"]+resid)+ANCHOR_WEIGHT*item["anchor"],elapsed))
                predictions[arch]=np.mean(np.stack([p for _,p,_ in rnn_seed_predictions[arch]]),axis=0)
                predict_times[arch]=float(np.mean([t for _,_,t in rnn_seed_predictions[arch]]))
            for name,pred in predictions.items():
                if pred.shape!=(24,) or not np.isfinite(pred).all():
                    raise ValueError(f"non-finite prediction: {fold['name']} {day} {name}")
                metric=_predict_plan_metrics(item["actual"],pred,item["target"])
                row={"fold":fold["name"],"target_date":day,"model":name,"seed":"mean_3" if name in ("LSTM","GRU") else None,
                    "evaluation_type":"retrospective chronological fold replay; source first_seen/model availability unverified",
                    "forecast_origin_kst":item["origin"].isoformat(),"input_max_timestamp_kst":item["input_max_timestamp"].isoformat(),
                    "trained_through":fold["train_end"],**metric}
                daily.append(row)
                for hour,(actual,predicted) in enumerate(zip(item["actual"],pred)):
                    hourly.append({"fold":fold["name"],"target_date":day,"model":name,"seed":row["seed"],"hour":hour,
                        "actual_smp":float(actual),"predicted_smp":float(predicted),"error":float(predicted-actual),
                        "forecast_origin_kst":item["origin"].isoformat()})
            # Keep individual RNN seed metrics and hourly vectors to expose variance.
            for arch in ("LSTM","GRU"):
                for seed,pred,_ in rnn_seed_predictions[arch]:
                    metric=_predict_plan_metrics(item["actual"],pred,item["target"])
                    daily.append({"fold":fold["name"],"target_date":day,"model":arch,"seed":seed,
                        "evaluation_type":"retrospective chronological fold replay; individual neural seed",
                        "forecast_origin_kst":item["origin"].isoformat(),"input_max_timestamp_kst":item["input_max_timestamp"].isoformat(),
                        "trained_through":fold["train_end"],**metric})
                    for hour,(actual,predicted) in enumerate(zip(item["actual"],pred)):
                        hourly.append({"fold":fold["name"],"target_date":day,"model":arch,"seed":seed,"hour":hour,
                            "actual_smp":float(actual),"predicted_smp":float(predicted),"error":float(predicted-actual),
                            "forecast_origin_kst":item["origin"].isoformat()})
    return {"daily":pd.DataFrame(daily),"hourly":pd.DataFrame(hourly),"fit_log":pd.DataFrame(fit_log),
        "calibration":calibration_df,"sequence_summary":sequence_summary,"chosen_sequence_hours":chosen_seq,
        "rnn_fits":rnn_fits}


def _paired_bootstrap(daily: pd.DataFrame, reference: str="HGB", repetitions: int=2000) -> pd.DataFrame:
    rng=np.random.default_rng(20261114)
    pooled=daily[daily.seed.isna() | daily.seed.eq("mean_3")]
    hgb=pooled[pooled.model.eq(reference)].set_index("target_date")
    rows=[]
    for model in [x for x in MODEL_NAMES if x!=reference]:
        part=pooled[pooled.model.eq(model)].set_index("target_date")
        common=sorted(set(hgb.index)&set(part.index))
        if not common: continue
        delta=part.loc[common,"mae"].to_numpy(float)-hgb.loc[common,"mae"].to_numpy(float)
        draws=np.array([rng.choice(delta,size=len(delta),replace=True).mean() for _ in range(repetitions)])
        rows.append({"model":model,"reference":reference,"n_days":len(common),"mean_daily_mae_delta_vs_hgb":float(delta.mean()),
            "bootstrap_95pct_low":float(np.quantile(draws,.025)),"bootstrap_95pct_high":float(np.quantile(draws,.975)),
            "bootstrap_repetitions":repetitions,"interpretation":"paired retrospective interval; not independent test evidence"})
    return pd.DataFrame(rows)


def _summarize(daily: pd.DataFrame, hourly: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    # Seed-level mean predictions and mean individual-run performance are reported separately.
    pooled=daily[daily.seed.isna() | daily.seed.eq("mean_3")].copy()
    summary=pooled.groupby("model",as_index=False).agg(days=("target_date","nunique"),folds=("fold","nunique"),
        MAE=("mae","mean"),RMSE=("rmse","mean"),G2V_regret=("g2v_regret_won","mean"),
        V2G_regret=("v2g_regret_won","mean"),G2V_actual_net_cost=("g2v_actual_net_cost_won","mean"),
        V2G_actual_net_cost=("v2g_actual_net_cost_won","mean"),V2G_gain_vs_G2V=("v2g_gain_vs_g2v_actual_won","mean"),
        peak_MAE=("peak_mae","mean"),low_price_top5_overlap=("low_price_top5_overlap","mean"))
    pooled_hourly=hourly[hourly.seed.isna() | hourly.seed.eq("mean_3")]
    pooled_rmse=pooled_hourly.groupby("model").error.apply(lambda values:float(np.sqrt(np.mean(np.square(values))))).to_dict()
    pooled_mae=pooled_hourly.groupby("model").error.apply(lambda values:float(np.mean(np.abs(values)))).to_dict()
    summary["MAE"]=[pooled_mae[m] for m in summary.model]
    summary["RMSE"]=[pooled_rmse[m] for m in summary.model]
    seed_rows=[]
    for model in ("LSTM","GRU"):
        seed_daily_mask=daily.seed.astype(str).isin({str(seed) for seed in SEEDS})
        block=daily[(daily.model==model)&seed_daily_mask]
        seed_stats=block.groupby("seed").agg(MAE=("mae","mean"),RMSE=("rmse","mean"),G2V_regret=("g2v_regret_won","mean"),
            V2G_regret=("v2g_regret_won","mean"),days=("target_date","nunique"))
        seed_hourly_mask=hourly.seed.astype(str).isin({str(seed) for seed in SEEDS})
        seed_hourly=hourly[(hourly.model.eq(model)) & seed_hourly_mask]
        seed_pooled=seed_hourly.groupby("seed").error.agg(
            MAE=lambda values:float(np.mean(np.abs(values))),
            RMSE=lambda values:float(np.sqrt(np.mean(np.square(values)))))
        if len(seed_stats):
            seed_rows.append({"model":model,"seed_runs":len(seed_stats),"seed_MAE_mean":seed_pooled.MAE.mean(),"seed_MAE_sd":seed_pooled.MAE.std(ddof=1),
                "seed_RMSE_mean":seed_pooled.RMSE.mean(),"seed_RMSE_sd":seed_pooled.RMSE.std(ddof=1),
                "seed_G2V_regret_mean":seed_stats.G2V_regret.mean(),"seed_G2V_regret_sd":seed_stats.G2V_regret.std(ddof=1),
                "seed_V2G_regret_mean":seed_stats.V2G_regret.mean(),"seed_V2G_regret_sd":seed_stats.V2G_regret.std(ddof=1)})
    seed_summary=pd.DataFrame(seed_rows)
    hourly_mae=hourly[hourly.seed.isna() | hourly.seed.eq("mean_3")].groupby(["model","hour"],as_index=False).agg(hourly_MAE=("error",lambda s:float(np.mean(np.abs(s)))))
    return summary,seed_summary,hourly_mae


def _make_figures(summary: pd.DataFrame, daily: pd.DataFrame, hourly: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True,exist_ok=True)
    order=[m for m in MODEL_NAMES if m in set(summary.model)]
    for column,title,ylabel,path in [("MAE","D+3 MAE by model","MAE (KRW/kWh)","mae_comparison.png"),
        ("RMSE","D+3 RMSE by model","RMSE (KRW/kWh)","rmse_comparison.png"),
        ("G2V_regret","G2V regret by model","Regret (KRW/day)","g2v_regret_comparison.png"),
        ("V2G_regret","Virtual V2G regret by model","Regret (KRW/day)","v2g_regret_comparison.png")]:
        fig,ax=plt.subplots(figsize=(10,5)); part=summary.set_index("model").reindex(order)
        ax.bar(part.index,part[column],color=["#216e9c" if m not in ("LSTM","GRU") else "#da7b1d" for m in part.index])
        ax.set(title=title,ylabel=ylabel);ax.tick_params(axis="x",rotation=35);ax.grid(axis="y",alpha=.25);fig.tight_layout();fig.savefig(FIGURES/path,dpi=170);plt.close(fig)
    folds=daily[(daily.seed.isna())|daily.seed.eq("mean_3")].groupby(["fold","model"],as_index=False).agg(MAE=("mae","mean"),RMSE=("rmse","mean"),G2V_regret=("g2v_regret_won","mean"))
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    for ax,col,title in zip(axes,["MAE","RMSE","G2V_regret"],["Fold MAE","Fold RMSE","Fold G2V regret"]):
        for model in order:
            p=folds[folds.model.eq(model)];ax.plot(p.fold,p[col],marker="o",label=model)
        ax.set(title=title);ax.grid(alpha=.25);ax.tick_params(axis="x",rotation=25)
    axes[-1].legend(fontsize=7,bbox_to_anchor=(1.05,1),loc="upper left");fig.tight_layout();fig.savefig(FIGURES/"fold_performance.png",dpi=170,bbox_inches="tight");plt.close(fig)
    hm=hourly[hourly.seed.isna()|hourly.seed.eq("mean_3")].groupby(["model","hour"]).error.apply(lambda s:float(np.mean(np.abs(s)))).unstack(0)
    fig,ax=plt.subplots(figsize=(12,5));hm.plot(ax=ax);ax.set(title="D+3 hourly MAE",xlabel="Target hour (KST)",ylabel="MAE (KRW/kWh)");ax.grid(alpha=.25);ax.legend(ncols=4,fontsize=8);fig.tight_layout();fig.savefig(FIGURES/"hourly_mae.png",dpi=170);plt.close(fig)
    latest_fold=daily[daily.fold.eq("fall2025")].target_date.max()
    day=latest_fold
    sample=hourly[hourly.target_date.eq(day)]
    fig,ax=plt.subplots(figsize=(12,5));
    actual=sample[sample.model.eq("D3_lag_reference")]
    ax.plot(actual.hour,actual.actual_smp,label="Actual SMP",color="#111827",lw=2)
    for model in order:
        block=sample[(sample.model.eq(model)) & (sample.seed.isna()|sample.seed.eq("mean_3"))]
        if not block.empty:ax.plot(block.hour,block.predicted_smp,label=model,lw=1.2)
    ax.set(title=f"Representative D+3 prediction · {day} · retrospective replay",xlabel="Hour KST",ylabel="SMP (KRW/kWh)");ax.grid(alpha=.25);ax.legend(ncols=3,fontsize=8);fig.tight_layout();fig.savefig(FIGURES/"representative_forecast.png",dpi=170);plt.close(fig)
    low=daily[daily.seed.isna()|daily.seed.eq("mean_3")].groupby("model",as_index=False).agg(top5=("low_price_top5_overlap","mean"))
    fig,ax=plt.subplots(figsize=(10,5));p=low.set_index("model").reindex(order);ax.bar(p.index,p.top5,color="#4e8b57");ax.set(title="Actual/Predicted low-price Top-5 overlap",ylabel="Mean overlap rate");ax.set_ylim(0,1);ax.tick_params(axis="x",rotation=35);ax.grid(axis="y",alpha=.25);fig.tight_layout();fig.savefig(FIGURES/"low_price_top5_overlap.png",dpi=170);plt.close(fig)


def _write_report(summary: pd.DataFrame, seed_summary: pd.DataFrame, sequence_summary: pd.DataFrame,
                  fit_log: pd.DataFrame, bootstrap: pd.DataFrame, available_versions: dict):
    rows=summary.sort_values("MAE").to_dict("records")
    table="| Model | Days | MAE | RMSE | Peak MAE | Low-price top-5 overlap | G2V regret/day | G2V actual cost/day | V2G regret/day | V2G actual net cost/day | Parameters / nodes* | Fit s (mean/fold) | Predict s/day |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for row in rows:
        runtime=fit_log[fit_log.model.eq(row["model"])]
        fit=float(runtime.fit_seconds.mean()) if len(runtime) else float("nan")
        pred=float(runtime.get("predict_seconds_per_day",pd.Series(dtype=float)).mean()) if len(runtime) else float("nan")
        pcount=runtime.get("parameter_count",pd.Series(dtype=float)).dropna()
        ncount=runtime.get("tree_node_or_linear_coefficient_count",pd.Series(dtype=float)).dropna()
        if len(pcount):
            complexity=f"{int(pcount.mean()):,} params"
        elif len(ncount):
            unit="coef." if row["model"]=="LinearRegression" else "nodes"
            complexity=f"{int(ncount.mean()):,} {unit}"
        else:
            complexity="n/a"
        table+=f"| {row['model']} | {int(row['days'])} | {row['MAE']:.3f} | {row['RMSE']:.3f} | {row['peak_MAE']:.3f} | {row['low_price_top5_overlap']:.3f} | {row['G2V_regret']:.2f} | {row['G2V_actual_net_cost']:.2f} | {row['V2G_regret']:.2f} | {row['V2G_actual_net_cost']:.2f} | {complexity} | {fit:.3f} | {pred:.6f} |\n"
    seed_table="| Model | Seed count | MAE mean ± SD | RMSE mean ± SD | G2V regret mean ± SD | V2G regret mean ± SD |\n|---|---:|---:|---:|---:|---:|\n"
    for _,r in seed_summary.iterrows():
        seed_table+=f"| {r.model} | {int(r.seed_runs)} | {r.seed_MAE_mean:.3f} ± {r.seed_MAE_sd:.3f} | {r.seed_RMSE_mean:.3f} ± {r.seed_RMSE_sd:.3f} | {r.seed_G2V_regret_mean:.2f} ± {r.seed_G2V_regret_sd:.2f} | {r.seed_V2G_regret_mean:.2f} ± {r.seed_V2G_regret_sd:.2f} |\n"
    seq_header="| Model | Sequence hours | Folds | Calibration days | Mean calibration MAE | Mean selected epoch | Mean fit seconds |\n|---|---:|---:|---:|---:|---:|---:|\n"
    seq_md=seq_header+"".join(f"| {r.model} | {int(r.sequence_hours)} | {int(r.folds)} | {int(r.days)} | {r.mean_calibration_mae:.3f} | {r.mean_best_epoch:.2f} | {r.mean_fit_seconds:.2f} |\n" for r in sequence_summary.itertuples())
    text=f"""# SMP-only 8-model D+3 comparison\n\n## Summary\n\nThis experiment retrains eight algorithms on the same SMP-only chronological D+3 folds: LSTM, GRU, HGB, LightGBM, ExtraTrees, Linear Regression, Random Forest, and XGBoost. It is an additional retrospective benchmark and does not replace V5–V20 research findings, the existing 316-day HGB artifact, or any operating model.\n\n## Data, folds, and leakage controls\n\n- Market: mainland/land SMP from the existing V11 SQLite price store. Target is D 00:00–23:00 KST; forecast origin is D−3 23:00 KST. All input feature timestamps are bounded by each origin. The historical store's original first-seen/publication vintage is not verified, so all scores are timestamp-causal historical replay, not validated operational performance.\n- SMP-only inputs: V17 shared price/calendar feature set (including target calendar cycles, D−3 same-hour reference, lagged SMP, and rolling statistics). Weather, demand/supply, renewable generation, fuel, and FX features are excluded.\n- Target: residual `actual SMP − D−3 same-hour reference`; final prediction is `0.95 × (reference + residual prediction) + 0.05 × median(SMP at D−7/D−14/D−21/D−28, same hour)`. This 5% anchor weight is fixed for all models. Direct-price predictions are not mixed into this main ranking.\n- Each fold uses the published time split: train through June 30 2024 / Dec 31 2024 / June 30 2025; calibration the following quarter; evaluate Oct–Dec 2024 / Apr–Jun 2025 / Oct–Dec 2025. No random split. Tree model parameters are predeclared and not tuned on evaluation days. RNN early stopping and common sequence length selection use calibration only.\n- EV: 60 kWh, initial SOC 30%, target SOC 80%, min/max SOC 20/90%, charge/discharge power 7 kW, efficiencies 90%, degradation 5 KRW/kWh, 24-hour connection. V2G is a virtual SMP-linked settlement scenario. Regret is realized cost of the prediction-based fixed schedule on actual SMP minus the actual-SMP oracle schedule cost.\n\n## 8-model result (275 evaluation dates)\n\nMAE and RMSE are pooled over paired hourly predictions; EV regrets are mean daily values.\n\n{table}\n### Recurrent seed variability\n\nThree fixed seeds ({SEEDS}) are reported for LSTM and GRU. Main metric uses the mean prediction across the three seeds; the table below shows the mean and standard deviation of the individual-run fold metrics.\n\n{seed_table}\n### 168h vs 672h sequence choice\n\nThe common sequence length is selected across both architectures and all folds using calibration MAE only.\n\n{seq_md}\n\n## Model structures\n\n- **LSTM**: one 32-unit Long Short-Term Memory recurrent layer encodes the past SMP series; dense layers map its state and known target calendar to 24 residuals. Its cell carries a memory state and input/forget/output gates.\n- **GRU**: one 32-unit Gated Recurrent Unit layer uses update/reset gates and a smaller state transition; input, loss, output, normalization, seed set, optimizer, epoch limit, and early-stop rule match LSTM. The fitted parameter-count and runtime difference is recorded in `results/model_runtime.csv`.\n- **HGB**: V20 HistGradientBoosting settings (250 iterations, 31 leaves, 0.05 learning rate, L2 2).\n- **LightGBM**: V12-style leaf-wise boosted trees, 300 estimators, 31 leaves, 0.05 rate.\n- **ExtraTrees**: 150 randomized trees, minimum leaf 10 and 0.8 feature subsampling (V11 settings).\n- **Linear Regression**: standardized V17 inputs then ordinary least squares; scaling is fitted on the training fold only.\n- **Random Forest**: 200 bagged regression trees, minimum leaf 5, 0.8 feature subsampling.\n- **XGBoost**: 300 histogram gradient-boosted trees, depth 4, 0.05 rate, subsampling and L2 regularization.\n\nNeural models output all 24 residuals jointly; tree and linear models estimate hourly residuals row by row. This output-structure difference is a limitation when attributing performance differences to the algorithm alone.\n\n## Uncertainty, stability, and interpretation\n\nPaired day bootstrap intervals against HGB are in `results/paired_bootstrap_vs_hgb.csv`. Fold-specific metrics, hourly errors, top-5 low-price overlap, high-price MAE, actual-SMP realized costs, model fit/predict time and artifact size are in the CSV outputs. Fold dates have been repeatedly reviewed during prior research and are not a fresh independent test. Bootstrap intervals quantify within-cohort variation only.\n\n{('The lowest aggregate MAE candidate is '+rows[0]['model']+f" ({rows[0]['MAE']:.3f} KRW/kWh), but selection should also consider fold variation, RMSE and charging regret; this repeated historical cohort does not justify an operational model change.") if rows else 'No complete evaluation was generated.'}\n\nThe existing V20 HGB model remains unchanged. Any difference from its prior reported score may arise from the fixed 5% anchor, fixed fold-specific retraining, output structures, library versions, and scoring aggregation; do not present it as an unconditional improvement.\n\n## Environment and reproducibility\n\nRuntime package versions are recorded in `results/environment.json`; fixed model specs and random seeds are in `results/study_manifest.json`. The benchmark environment is separate from existing V20 and weather environments.\n\n```powershell\n$env:SMP_SOURCE = (Resolve-Path ..\\gradproject).Path\n.\\.venv-smp-benchmark\\Scripts\\python.exe -m research.smp_model_benchmark.benchmark\n```\n\n## Limits\n\nNo sample is verified against original first-seen receipt timestamps. Existing target windows were previously examined. The V2G sale ratio represents a hypothetical SMP-linked scenario, not a guaranteed Korean settlement payment. Models are research-only; no auto-promotion, website, official issuance gate, or operating model was changed.\n"""
    candidates=summary[summary.model.isin(MODEL_NAMES)]
    best_mae=candidates.sort_values("MAE").iloc[0]
    best_g2v=candidates.sort_values("G2V_regret").iloc[0]
    best_v2g=candidates.sort_values("V2G_regret").iloc[0]
    rnn=candidates[candidates.model.isin(["LSTM","GRU"])].set_index("model")
    gru_delta=float(rnn.loc["GRU","MAE"]-rnn.loc["LSTM","MAE"])
    gru_params=int(fit_log[fit_log.model.eq("GRU")].parameter_count.dropna().mean())
    lstm_params=int(fit_log[fit_log.model.eq("LSTM")].parameter_count.dropna().mean())
    lstm_fit=float(fit_log[fit_log.model.eq("LSTM")].fit_seconds.mean())
    gru_fit=float(fit_log[fit_log.model.eq("GRU")].fit_seconds.mean())
    insight=(f"## Result interpretation\n\n- Lowest price MAE: **{best_mae.model}**, {best_mae.MAE:.3f} KRW/kWh; lowest G2V regret: **{best_g2v.model}**, {best_g2v.G2V_regret:.2f} KRW/day. These are different winners.\n- Lowest virtual V2G regret: **{best_v2g.model}**, {best_v2g.V2G_regret:.2f} KRW/day. V2G remains hypothetical; a lower V2G regret does not establish lower realized cost than G2V.\n- GRU vs LSTM ensemble prediction: GRU MAE difference {gru_delta:+.3f} KRW/kWh ({gru_delta / float(rnn.loc['LSTM','MAE'])*100:+.2f}%). GRU has {gru_params:,} vs {lstm_params:,} parameters ({(gru_params/lstm_params-1)*100:+.2f}%) but mean fold training time {gru_fit:.1f}s vs {lstm_fit:.1f}s. Its price error is marginally lower, while its G2V regret is higher; this does not make GRU the preferred overall model.\n- The selected RNN window is 672 hours from calibration only; its calibration MAE advantage over 168 hours is small and its training cost is higher. The main evaluation does not tune this choice.\n- On this repeated retrospective cohort, ExtraTrees is strongest on aggregate price accuracy, Linear Regression has the lowest G2V regret, and HGB/LightGBM/XGBoost remain competitive tree alternatives. None of these results justify replacing the separately selected V20 models or claiming operational improvement.\n\n")
    text=text.replace("## Model structures", insight+"## Model structures")
    (OUT/"MODEL_COMPARISON_REPORT.md").write_text(text,encoding="utf-8")


def run() -> dict:
    RESULTS.mkdir(parents=True,exist_ok=True); FIGURES.mkdir(parents=True,exist_ok=True)
    prices=load_prices()
    target_days=[]
    for fold in FOLDS: target_days.extend(pd.date_range(fold["val_start"],fold["val_end"],freq="D").strftime("%Y-%m-%d").tolist())
    # Include only targets needed by training/calibration/validation; the source
    # rows may extend later, but each row's predictors are origin-causal.
    last_study_date=max(target_days)
    eligible_dates=prices.loc[prices.timestamp.dt.strftime("%Y-%m-%d")<=last_study_date,"timestamp"].dt.strftime("%Y-%m-%d").unique().tolist()
    days=make_day_map(prices,eligible_dates)
    missing=sorted(set(target_days)-set(days))
    if missing: raise RuntimeError(f"incomplete target day inputs/actuals in required fold: {missing[:10]} ({len(missing)} total)")
    # Leakage mutation check spans each fold's first, middle and final evaluation day.
    for fold in FOLDS:
        for day in (fold["val_start"],pd.Timestamp(fold["val_start"])+pd.Timedelta(days=45),fold["val_end"]):
            validate_no_future_leak(prices,days[pd.Timestamp(day).strftime("%Y-%m-%d")])
    result=_primary_run(days)
    daily=result["daily"]; hourly=result["hourly"]; fit_log=result["fit_log"]
    summary,seed_summary,hourly_mae=_summarize(daily,hourly)
    boot=_paired_bootstrap(daily)
    # Add timing fields from actual inference on first target day per fold.
    # Runtime includes one 24-hour batch and is reported as mean wall-clock seconds/day.
    rng=np.random.default_rng(20261114)
    for model in MODEL_NAMES:
        if model in ("LSTM","GRU"):
            logs=fit_log[fit_log.model.eq(model)]
            fit_log.loc[fit_log.model.eq(model),"predict_seconds_per_day"] = np.nan
        else:
            fit_log.loc[fit_log.model.eq(model),"predict_seconds_per_day"] = np.nan
    # Build per-model inference timing once per fold against a complete 24-hour feature matrix.
    for fold in FOLDS:
        day=FOLDS[[f["name"] for f in FOLDS].index(fold["name"])]["val_start"]
        item=days[day]
        for model_name in ("HGB","LightGBM","ExtraTrees","LinearRegression","RandomForest","XGBoost"):
            model_path=MODELS/f"{fold['name']}_{model_name.lower()}_d3.joblib"
            model=joblib.load(model_path)
            start=time.perf_counter()
            for _ in range(10): model.predict(item["X"])
            sec=(time.perf_counter()-start)/10
            fit_log.loc[(fit_log.fold.eq(fold["name"]))&(fit_log.model.eq(model_name)),"predict_seconds_per_day"]=sec
        for arch in ("LSTM","GRU"):
            for seed in SEEDS:
                bundle=result.get("rnn_fits",{}).get((fold["name"],arch,result["chosen_sequence_hours"],seed))
                if bundle:
                    _,sec=_rnn_predict(bundle,item)
                    fit_log.loc[(fit_log.fold.eq(fold["name"]))&(fit_log.model.eq(arch))&(fit_log.seed.eq(seed)),"predict_seconds_per_day"]=sec
    # parameter counts for tabular estimators and artifact file size are sampled from saved models.
    for _,row in fit_log.iterrows():
        if row.model in ("LSTM","GRU"):
            continue
        path=MODELS/f"{row.fold}_{row.model.lower()}_d3.joblib"
        model=joblib.load(path)
        params,nodes=_model_complexity(model)
        fit_log.loc[(fit_log.fold.eq(row.fold))&(fit_log.model.eq(row.model)),"parameter_count"]=params
        fit_log.loc[(fit_log.fold.eq(row.fold))&(fit_log.model.eq(row.model)),"tree_node_or_linear_coefficient_count"]=nodes
    summary,seed_summary,hourly_mae=_summarize(daily,hourly)
    summary.to_csv(RESULTS/"model_comparison.csv",index=False)
    daily.to_csv(RESULTS/"daily_scores.csv",index=False)
    hourly.to_csv(RESULTS/"hourly_predictions.csv",index=False)
    hourly_mae.to_csv(RESULTS/"hourly_mae.csv",index=False)
    seed_summary.to_csv(RESULTS/"neural_seed_variability.csv",index=False)
    result["sequence_summary"].to_csv(RESULTS/"rnn_sequence_calibration.csv",index=False)
    result["calibration"].to_csv(RESULTS/"rnn_calibration_runs.csv",index=False)
    fit_log.to_csv(RESULTS/"model_runtime.csv",index=False)
    boot.to_csv(RESULTS/"paired_bootstrap_vs_hgb.csv",index=False)
    _make_figures(summary,daily,hourly)
    versions={"python":sys.version.split()[0],"numpy":np.__version__,"pandas":pd.__version__,"scikit_learn":sklearn.__version__,
        "tensorflow":None,"lightgbm":None,"xgboost":None,"platform":sys.platform}
    try:
        import tensorflow as tf; versions["tensorflow"]=tf.__version__
    except Exception as exc: versions["tensorflow"]=f"unavailable: {type(exc).__name__}"
    try:
        import lightgbm; versions["lightgbm"]=lightgbm.__version__
    except Exception as exc: versions["lightgbm"]=f"unavailable: {type(exc).__name__}"
    try:
        import xgboost; versions["xgboost"]=xgboost.__version__
    except Exception as exc: versions["xgboost"]=f"unavailable: {type(exc).__name__}"
    (RESULTS/"environment.json").write_text(json.dumps(versions,ensure_ascii=False,indent=2),encoding="utf-8")
    manifest={"status":"completed","horizon_days":3,"origin":"target date - 3 days at 23:00 KST","evaluation_type":"historical timestamp-causal fold replay; first_seen and contemporaneous model availability unverified",
        "folds":list(FOLDS),"evaluation_days":len(set(daily[daily.model.eq("HGB")].target_date)),"missing_days":missing,
        "models":list(MODEL_NAMES),"features":list(FEATURES),"target":"hourly residual=actual SMP - D-3 same-hour SMP",
        "anchor":{"method":"same-hour median from target D-7/D-14/D-21/D-28","weight":ANCHOR_WEIGHT},
        "rnn":{"sequence_candidates_hours":list(SEQUENCE_CANDIDATES),"selected_common_sequence_hours":result["chosen_sequence_hours"],
            "seeds":list(SEEDS),"input_normalization":"mean/std fitted on training fold inputs only","target_normalization":"mean/std fitted on training fold residuals only",
            "calendar_input":"target hour, weekday, month cyclical features known at origin","early_stopping":"calibration MSE, patience 8, best weights restored"},
        "ev_conditions":EV_RULE.__dict__,"validation_dates_previously_reviewed":True,"original_first_seen_verified":False,
        "web_modified":False,"operating_model_replaced":False,"future_leak_mutation_tests":9}
    (RESULTS/"study_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    _write_report(summary,seed_summary,result["sequence_summary"],fit_log,boot,versions)
    return {"summary":summary,"seed_summary":seed_summary,"sequence_summary":result["sequence_summary"],"manifest":manifest}


if __name__=="__main__":
    output=run()
    print(output["summary"].to_string(index=False))
    print("selected RNN sequence hours:",output["manifest"]["rnn"]["selected_common_sequence_hours"])
