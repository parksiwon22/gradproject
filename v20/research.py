"""Gate-independent, immutable research forecasts and scorecards for V20.

Research records are never read by the official publisher's eligibility logic.
Every model input is sliced to the receipt cutoff captured before inference.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from v11.store import connect, latest
from v12.ev import EV
from v18.evaluation import compare_ev, price_metrics, reference
from v19.common import ROOT, read, write_csv, sha, now
from v19.models import predict_model
from v20.core import HOME, write_audit_once

RESEARCH = HOME / 'research'
MODEL = 'V17_joint'
MODEL_FILE = ROOT / 'artifacts/v17/study/winter2025.joblib'
TRAINED_THROUGH = '2025-06-30'


def _cutoff_prices(cutoff):
    db = connect()
    try:
        frame = latest(db, 'land', as_of=pd.Timestamp(cutoff).tz_convert('UTC').isoformat())
    finally:
        db.close()
    excluded_future = int((frame.timestamp > cutoff).sum())
    excluded_late_receipts = int((pd.to_datetime(frame.first_seen, utc=True) > cutoff.tz_convert('UTC')).sum())
    frame = frame[frame.timestamp <= cutoff].copy().sort_values('timestamp')
    frame.attrs['excluded_future_timestamp_rows'] = excluded_future
    frame.attrs['excluded_late_receipt_rows'] = excluded_late_receipts
    if frame.timestamp.duplicated().any():
        raise ValueError('Duplicate input timestamps')
    return frame


def _verified_model_hash():
    registry = read(ROOT / 'artifacts/v19/model_registry.json', {})
    meta = registry.get('entries', {}).get(MODEL)
    if not meta or not MODEL_FILE.exists() or sha(MODEL_FILE) != meta.get('sha256'):
        raise ValueError('Frozen V17 model hash mismatch or registry entry unavailable')
    return meta['sha256']


def _record_path(origin, horizon, model):
    return RESEARCH / 'forecasts' / f"{origin.strftime('%Y%m%dT%H%M')}_D{horizon}_{model}.json"


def baseline_prediction(prices, target, horizon):
    """Return a same-hour past price and its source timestamp for one horizon."""
    target = pd.DatetimeIndex(target)
    source_times = target - pd.Timedelta(days=horizon)
    series = prices.set_index('timestamp').smp
    values = series.reindex(source_times).to_numpy(dtype=float)
    return source_times, values


def generate(origin=None, clock=None):
    """Save D+1..D+3 research predictions, regardless of publication eligibility."""
    clock = pd.Timestamp.now(tz='Asia/Seoul') if clock is None else pd.Timestamp(clock).tz_convert('Asia/Seoul')
    issued = clock
    # The recorded forecast origin is the actual research issue instant. This
    # keeps the input cutoff and forecast time identical, including a late retry.
    gate_origin = pd.Timestamp(origin).tz_convert('Asia/Seoul') if origin is not None else issued
    origin = issued
    prices = _cutoff_prices(issued)
    if prices.empty:
        return dict(status='no_input_history', forecast_files=0, reference_origin=origin.isoformat())
    target_known = {d for d in prices.loc[prices.timestamp <= issued, 'timestamp'].dt.strftime('%Y-%m-%d')}
    entries = []
    try:
        model_hash = _verified_model_hash()
    except ValueError:
        model_hash = None
    for horizon in (1, 2, 3):
        day = (origin.normalize() + pd.Timedelta(days=horizon)).strftime('%Y-%m-%d')
        target = pd.date_range(pd.Timestamp(day, tz='Asia/Seoul'), periods=24, freq='h')
        if horizon == 1:
            # Transparent no-fit baseline: same hour from the latest available prior day.
            source_times, baseline_values = baseline_prediction(prices, target, horizon)
            model = 'same_hour_previous_day_baseline'
            predicted = baseline_values
            features = [{'timestamp': t.isoformat(), 'baseline_source_timestamp': source.isoformat(),
                         'baseline_source_smp': float(v)} for t, source, v in zip(target, source_times, predicted)]
        else:
            if model_hash is None:
                entries.append(dict(target_date=day, horizon_days=horizon, status='model_unavailable'))
                continue
            _, predicted, _, _ = predict_model(MODEL, prices, day, horizon)
            model = MODEL
            features = []
        if len(predicted) != 24 or not np.isfinite(predicted).all():
            entries.append(dict(target_date=day, horizon_days=horizon, model=model, status='incomplete_cutoff_history'))
            continue
        path = _record_path(origin, horizon, model)
        if path.exists():
            saved = read(path, {})
            entries.append(dict(target_date=day, horizon_days=horizon, model=model, status='already_saved', file=path.name,
                                sha256=sha(path), rows=len(saved.get('rows', []))))
            continue
        provenance = prices[['timestamp', 'first_seen', 'snapshot']].copy()
        max_receipt = pd.to_datetime(provenance.first_seen, utc=True).max()
        future_input_rows = int((prices.timestamp > issued).sum())
        late_receipts = int((pd.to_datetime(provenance.first_seen, utc=True) > issued.tz_convert('UTC')).sum())
        record = dict(schema=1, mode='RESEARCH_ONLY_NOT_OFFICIAL_ISSUANCE', model=model,
                      model_sha256=model_hash if horizon > 1 else 'deterministic_baseline_v1',
                      trained_through=TRAINED_THROUGH if horizon > 1 else None,
                      reference_origin=origin.isoformat(), issued_at=issued.isoformat(), input_cutoff=issued.isoformat(),
                      official_issue_window_origin=gate_origin.isoformat(),
                      target_date=day, horizon_days=horizon, target_prices_known_at_issue=day in target_known,
                      input_max_timestamp=prices.timestamp.max().isoformat(), input_max_first_seen=max_receipt.isoformat(),
                      future_input_rows=future_input_rows, late_receipt_rows=late_receipts,
                      future_timestamp_rows_excluded=prices.attrs.get('excluded_future_timestamp_rows', 0),
                      late_receipt_rows_excluded=prices.attrs.get('excluded_late_receipt_rows', 0),
                      source_snapshot_hashes=sorted(set(provenance.snapshot.astype(str))), features=features,
                      rows=[dict(timestamp=t.isoformat(), predicted_smp=float(v)) for t, v in zip(target, predicted)],
                      publication_gate_consulted=False, official_issuance=False)
        write_audit_once(path, record)
        entries.append(dict(target_date=day, horizon_days=horizon, model=model, status='created', file=path.name,
                            sha256=sha(path), rows=24))
    result = dict(at=now(), mode='RESEARCH_ONLY_NOT_OFFICIAL_ISSUANCE', reference_origin=origin.isoformat(),
                  input_cutoff=issued.isoformat(), records=entries, forecast_files=sum(e.get('rows') == 24 for e in entries),
                  official_gate_changed=False)
    status = RESEARCH / 'forecast_status.json'
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def inventory():
    records = []
    for path in sorted((RESEARCH / 'forecasts').glob('*.json')):
        row = read(path, {})
        if row.get('mode') != 'RESEARCH_ONLY_NOT_OFFICIAL_ISSUANCE':
            continue
        frame = pd.DataFrame(row.get('rows', []))
        if len(frame) != 24:
            continue
        frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True).dt.tz_convert('Asia/Seoul')
        records.append((path, row, frame))
    return records


def _baseline_backtest(prices, horizon):
    """Reconstruct weekly-spaced 2026 forecasts; labels never enter features."""
    rows = []
    s = prices.set_index('timestamp').smp.sort_index()
    days = sorted(d for d in set(prices.timestamp.dt.strftime('%Y-%m-%d')) if d >= '2026-01-01')[::7]
    for day in days:
        target = pd.date_range(pd.Timestamp(day, tz='Asia/Seoul'), periods=24, freq='h')
        origin = target[0].normalize() - pd.Timedelta(days=horizon) + pd.Timedelta(hours=23)
        if target.max() > prices.timestamp.max():
            continue
        actual = s.reindex(target).to_numpy(dtype=float)
        _, prediction = baseline_prediction(prices, target, horizon)
        if not np.isfinite(actual).all() or not np.isfinite(prediction).all():
            continue
        rows.append((origin, day, target, actual, prediction))
    return rows


def evaluate(backtest=True):
    """Refresh operational scorecards and a clearly separate historical replay."""
    RESEARCH.mkdir(parents=True, exist_ok=True)
    actual, conflicts = reference()
    actual = actual[['timestamp', 'smp']].drop_duplicates('timestamp').sort_values('timestamp')
    actual_index = actual.set_index('timestamp').smp
    rows, hourly, regrets = [], [], []
    for path, record, forecast in inventory():
        target = pd.date_range(pd.Timestamp(record['target_date'], tz='Asia/Seoul'), periods=24, freq='h')
        observed = actual_index.reindex(target).to_numpy(dtype=float)
        common = dict(evaluation_type='operational_research', file=path.name, forecast_sha256=sha(path),
                      model=record['model'], horizon_days=record['horizon_days'], target_date=record['target_date'],
                      reference_origin=record['reference_origin'], issued_at=record['issued_at'],
                      input_cutoff=record['input_cutoff'], target_prices_known_at_issue=record['target_prices_known_at_issue'],
                      future_input_rows=record['future_input_rows'], late_receipt_rows=record['late_receipt_rows'])
        if len(forecast) != 24 or not np.isfinite(observed).all():
            rows.append(dict(**common, status='awaiting_official_actuals', official_hours=int(np.isfinite(observed).sum())))
            continue
        predicted = forecast.predicted_smp.to_numpy(dtype=float)
        metrics = price_metrics(observed, predicted)
        ev = compare_ev(predicted, observed, target, ev=EV(degradation=5))
        rows.append(dict(**common, status='evaluated', official_hours=24, **metrics,
                         G2V_regret=ev['G2V_regret'], V2G_regret=ev['V2G_regret'],
                         G2V_actual_cost=ev['G2V_actual'], V2G_actual_cost=ev['V2G_actual'],
                         immediate_actual_cost=ev['immediate_actual'], uniform_actual_cost=ev['uniform_actual']))
        for hour, a, p in zip(target, observed, predicted):
            hourly.append(dict(**common, timestamp=hour.isoformat(), hour=hour.hour,
                               actual_smp=float(a), predicted_smp=float(p), error=float(p-a), absolute_error=float(abs(p-a))))
        regrets.append(dict(**common, G2V_regret=ev['G2V_regret'], V2G_regret=ev['V2G_regret'],
                            G2V_actual_cost=ev['G2V_actual'], V2G_actual_cost=ev['V2G_actual']))
    write_csv(RESEARCH / 'operational_metrics.csv', pd.DataFrame(rows))
    write_csv(RESEARCH / 'operational_hourly_errors.csv', pd.DataFrame(hourly))
    write_csv(RESEARCH / 'operational_ev_regret.csv', pd.DataFrame(regrets))

    back_rows, back_hourly = [], []
    backtest_ready = (RESEARCH / 'backtest_summary.csv').exists()
    if backtest and not backtest_ready:
        db = connect()
        try:
            history = latest(db, 'land')
        finally:
            db.close()
        try:
            model_available = _verified_model_hash() is not None
        except ValueError:
            model_available = False
        for horizon in (1, 2, 3):
            for origin, day, target, observed, baseline in _baseline_backtest(history, horizon):
                candidates = [('same_hour_previous_day_baseline', baseline)]
                if horizon in (2, 3) and model_available:
                    cutoff = target[0].normalize() - pd.Timedelta(days=horizon) + pd.Timedelta(hours=23)
                    available = history[history.timestamp <= cutoff]
                    if cutoff.normalize() > pd.Timestamp(TRAINED_THROUGH, tz='Asia/Seoul'):
                        try:
                            _, pred, _, _ = predict_model(MODEL, available, day, horizon)
                            candidates.append((MODEL, pred))
                        except (ValueError, KeyError, TypeError):
                            pass
                for model, predicted in candidates:
                    metrics = price_metrics(observed, predicted)
                    ev = compare_ev(predicted, observed, target, ev=EV(degradation=5))
                    common = dict(evaluation_type='historical_replay_backtest', model=model, horizon_days=horizon,
                                  target_date=day, reference_origin=origin.isoformat(),
                                  input_vintage_verified=False, trained_through=TRAINED_THROUGH if model == MODEL else None)
                    back_rows.append(dict(**common, status='evaluated', **metrics, G2V_regret=ev['G2V_regret'],
                                          V2G_regret=ev['V2G_regret'], G2V_actual_cost=ev['G2V_actual'],
                                          V2G_actual_cost=ev['V2G_actual']))
                    for hour, a, p in zip(target, observed, predicted):
                        back_hourly.append(dict(**common, timestamp=hour.isoformat(), hour=hour.hour,
                                                actual_smp=float(a), predicted_smp=float(p), error=float(p-a),
                                                absolute_error=float(abs(a-p))))
    if back_rows or (backtest and not backtest_ready):
        write_csv(RESEARCH / 'backtest_metrics.csv', pd.DataFrame(back_rows))
        write_csv(RESEARCH / 'backtest_hourly_errors.csv', pd.DataFrame(back_hourly))
    if back_rows:
        group = pd.DataFrame(back_rows).groupby(['model', 'horizon_days'], as_index=False).agg(
            days=('target_date', 'nunique'), MAE=('MAE', 'mean'), RMSE=('RMSE', 'mean'),
            G2V_regret=('G2V_regret', 'mean'), V2G_regret=('V2G_regret', 'mean'))
        write_csv(RESEARCH / 'backtest_summary.csv', group)
    if hourly:
        hframe = pd.DataFrame(hourly).groupby(['model', 'horizon_days', 'hour'], as_index=False).agg(
            samples=('timestamp', 'count'), MAE=('absolute_error', 'mean'), bias=('error', 'mean'))
        write_csv(RESEARCH / 'operational_hourly_summary.csv', hframe)
    historical_days = len({r['target_date'] for r in back_rows})
    if not historical_days and (RESEARCH / 'backtest_metrics.csv').exists():
        historical_days = int(pd.read_csv(RESEARCH / 'backtest_metrics.csv').target_date.nunique())
    status = dict(at=now(), operational_forecasts=len(rows), operational_evaluated=sum(r['status']=='evaluated' for r in rows),
                  historical_backtest_days=historical_days,
                  historical_backtest_vintage_verified=False, historical_backtest_sampling='fixed every-7th available target day from 2026-01-01', reference_conflicts=conflicts,
                  V5_results_preserved=True, cross_period_improvement_claim_allowed=False)
    (RESEARCH / 'evaluation_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    return status


if __name__ == '__main__':
    print(evaluate())
