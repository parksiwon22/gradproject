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
    future = frame[frame.timestamp > cutoff]
    excluded_future = int(len(future))
    excluded_late_receipts = int((pd.to_datetime(frame.first_seen, utc=True) > cutoff.tz_convert('UTC')).sum())
    future_dates = sorted(set(future.timestamp.dt.strftime('%Y-%m-%d')))
    future_counts = future.groupby(future.timestamp.dt.strftime('%Y-%m-%d')).size().to_dict()
    frame = frame[frame.timestamp <= cutoff].copy().sort_values('timestamp')
    frame.attrs['excluded_future_timestamp_rows'] = excluded_future
    frame.attrs['excluded_late_receipt_rows'] = excluded_late_receipts
    frame.attrs['available_future_timestamp_dates'] = future_dates
    frame.attrs['available_future_timestamp_counts'] = future_counts
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


def available_same_hour_baseline(prices, target, preferred_lag_days=1, max_lag_days=7):
    """Use the nearest available prior same-hour SMP, recording fallback lags."""
    target = pd.DatetimeIndex(target)
    series = prices.set_index('timestamp').smp
    source_times = []
    values = []
    lags = []
    for timestamp in target:
        found = False
        for lag in range(preferred_lag_days, max_lag_days + 1):
            source = timestamp - pd.Timedelta(days=lag)
            value = series.get(source, np.nan)
            if pd.notna(value) and source <= prices.timestamp.max():
                source_times.append(source)
                values.append(float(value))
                lags.append(lag)
                found = True
                break
        if not found:
            source_times.append(pd.NaT)
            values.append(np.nan)
            lags.append(None)
    return pd.DatetimeIndex(source_times), np.asarray(values, dtype=float), lags


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
    target_known = set(prices.attrs.get('available_future_timestamp_dates', []))
    entries = []
    try:
        model_hash = _verified_model_hash()
    except ValueError:
        model_hash = None
    for horizon in (1, 2, 3):
        day = (origin.normalize() + pd.Timedelta(days=horizon)).strftime('%Y-%m-%d')
        target = pd.date_range(pd.Timestamp(day, tz='Asia/Seoul'), periods=24, freq='h')
        if horizon == 1:
            # Prefer the previous day's same hour; if missing at issue, fall back
            # to the nearest earlier available day without filling from the future.
            source_times, baseline_values, baseline_lags = available_same_hour_baseline(prices, target)
            model = 'same_hour_previous_day_baseline'
            predicted = baseline_values
            features = [{'timestamp': t.isoformat(), 'baseline_source_timestamp': source.isoformat() if pd.notna(source) else None,
                         'baseline_source_smp': float(v) if np.isfinite(v) else None, 'baseline_lag_days': lag}
                        for t, source, v, lag in zip(target, source_times, predicted, baseline_lags)]
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
                      target_timestamp_rows_available_at_issue=int(prices.attrs.get('available_future_timestamp_counts', {}).get(day, 0)),
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
    days = sorted(d for d in set(prices.timestamp.dt.strftime('%Y-%m-%d'))
                  if '2026-01-01' <= d <= '2026-10-15')[::7]
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


def _v5_d1_reference():
    """Rescore preserved V5 final-test predictions under the V20 EV conditions."""
    source = ROOT / 'artifacts/results/v5_day_ahead_predictions.csv'
    if not source.exists():
        return pd.DataFrame()
    frame = pd.read_csv(source)
    models = {'profile_hgb_blend': 'V5_profile_hgb_blend',
              'daily_profile_baseline': 'V5_daily_profile_baseline'}
    common_days = None
    daily = {}
    for column, model in models.items():
        days = set(frame.loc[frame[column].notna(), 'target_date'].astype(str))
        common_days = days if common_days is None else common_days & days
    for day in sorted(common_days or []):
        block = frame[frame.target_date.astype(str).eq(day)].sort_values('timestamp')
        if len(block) != 24 or block.actual_smp.isna().any():
            continue
        times = pd.DatetimeIndex(pd.to_datetime(block.timestamp)).tz_localize('Asia/Seoul')
        actual = block.actual_smp.to_numpy(dtype=float)
        for column, model in models.items():
            predicted = block[column].to_numpy(dtype=float)
            metrics = price_metrics(actual, predicted)
            ev = compare_ev(predicted, actual, times, ev=EV(degradation=5))
            daily.setdefault(model, []).append(dict(target_date=day, **metrics,
                G2V_regret=ev['G2V_regret'], V2G_regret=ev['V2G_regret'],
                EV_conditions='full-day EV(degradation=5)'))
    rows = []
    for model, records in daily.items():
        d = pd.DataFrame(records)
        rows.append(dict(model=model, horizon_days=1, days=len(d),
            MAE=float(d.MAE.mean()), RMSE=float(d.RMSE.mean()),
            G2V_regret=float(d.G2V_regret.mean()), V2G_regret=float(d.V2G_regret.mean()),
            input_vintage_verified=False, comparison_period='V5 saved final test: 2023-10-01 through 2023-12-31',
            EV_conditions='same target dates; full-day EV(degradation=5)',
            comparison_basis='preserved V5 predictions; distinct period from V20 operational/replay results'))
    detail = [dict(model=model, **record, input_vintage_verified=False,
                   comparison_period='V5 saved final test: 2023-10-01 through 2023-12-31')
              for model, records in daily.items() for record in records]
    write_csv(RESEARCH / 'v5_d1_reference_metrics.csv', pd.DataFrame(detail))
    result = pd.DataFrame(rows)
    write_csv(RESEARCH / 'v5_d1_reference_summary.csv', result)
    return result


def evaluate(backtest=True):
    """Refresh operational scorecards and a clearly separate historical replay."""
    RESEARCH.mkdir(parents=True, exist_ok=True)
    actual, conflicts = reference()
    actual = actual[['timestamp', 'smp']].drop_duplicates('timestamp').sort_values('timestamp')
    actual_index = actual.set_index('timestamp').smp
    rows, hourly, regrets = [], [], []
    saved = inventory()
    # Retries are immutable evidence, but one forecast per origin/target/model
    # is the unit of evaluation. Keep the earliest issue as canonical.
    saved.sort(key=lambda item: (pd.Timestamp(item[1]['issued_at']), item[0].name))
    canonical, seen, duplicates = [], {}, []
    for path, record, forecast in saved:
        key = (record['reference_origin'][:10], record['target_date'], int(record['horizon_days']),
               record['model'], record.get('model_sha256'))
        digest = sha(path)
        vector_hash = __import__('hashlib').sha256(
            np.asarray(forecast.predicted_smp, dtype='<f8').tobytes()).hexdigest()
        if key in seen:
            prior_path, prior_record, prior_vector = seen[key]
            duplicates.append(dict(file=path.name, duplicate_of=prior_path.name, sha256=digest,
                                   target_date=record['target_date'], horizon_days=record['horizon_days'],
                                   model=record['model'], issued_at=record['issued_at'],
                                   canonical_issued_at=prior_record['issued_at'],
                                   identical_prediction=vector_hash == prior_vector,
                                   excluded_from_evaluation=True))
            continue
        seen[key] = (path, record, vector_hash)
        canonical.append((path, record, forecast))
    write_csv(RESEARCH / 'duplicate_forecasts.csv', pd.DataFrame(duplicates))
    for path, record, forecast in canonical:
        target = pd.date_range(pd.Timestamp(record['target_date'], tz='Asia/Seoul'), periods=24, freq='h')
        observed = actual_index.reindex(target).to_numpy(dtype=float)
        issued_cutoff = _cutoff_prices(pd.Timestamp(record['issued_at']))
        issue_counts = issued_cutoff.attrs.get('available_future_timestamp_counts', {})
        known_target_rows = int(issue_counts.get(record['target_date'], 0))
        known_target = known_target_rows > 0
        common = dict(evaluation_type='operational_research', file=path.name, forecast_sha256=sha(path),
                      model=record['model'], horizon_days=record['horizon_days'], target_date=record['target_date'],
                      reference_origin=record['reference_origin'], issued_at=record['issued_at'],
                      input_cutoff=record['input_cutoff'], target_prices_known_at_issue=known_target,
                      target_timestamp_rows_available_at_issue=known_target_rows,
                      saved_target_known_flag=record.get('target_prices_known_at_issue'),
                      future_input_rows=record['future_input_rows'], late_receipt_rows=record['late_receipt_rows'])
        if known_target:
            rows.append(dict(**common, status='excluded_target_prices_known_at_issue',
                             official_hours=int(np.isfinite(observed).sum())))
            continue
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
    backtest_ready = False
    if backtest:
        db = connect()
        try:
            history = latest(db, 'land')
        finally:
            db.close()
        try:
            model_available = _verified_model_hash() is not None
        except ValueError:
            model_available = False
        vintage_min_receipt = pd.to_datetime(history.first_seen, utc=True).min()
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
                                  input_vintage_verified=False,
                                  input_receipts_available_at_origin=False,
                                  input_first_seen_min=vintage_min_receipt.isoformat(),
                                  model_checkpoint_available_at_origin=False if model == MODEL else None,
                                  trained_through=TRAINED_THROUGH if model == MODEL else None)
                    if model == MODEL:
                        common['input_receipts_available_at_origin'] = bool(
                            (pd.to_datetime(available.first_seen, utc=True) <= cutoff.tz_convert('UTC')).all())
                    back_rows.append(dict(**common, status='evaluated', **metrics, G2V_regret=ev['G2V_regret'],
                                          V2G_regret=ev['V2G_regret'], G2V_actual_cost=ev['G2V_actual'],
                                          V2G_actual_cost=ev['V2G_actual']))
                    for hour, a, p in zip(target, observed, predicted):
                        back_hourly.append(dict(**common, timestamp=hour.isoformat(), hour=hour.hour,
                                                actual_smp=float(a), predicted_smp=float(p), error=float(p-a),
                                                absolute_error=float(abs(a-p))))
    if back_rows or (backtest and not backtest_ready):
        write_csv(RESEARCH / 'backtest_42_metrics.csv', pd.DataFrame(back_rows))
        write_csv(RESEARCH / 'backtest_42_hourly_errors.csv', pd.DataFrame(back_hourly))
    if hourly:
        hframe = pd.DataFrame(hourly).groupby(['model', 'horizon_days', 'hour'], as_index=False).agg(
            samples=('timestamp', 'count'), MAE=('absolute_error', 'mean'), bias=('error', 'mean'))
        write_csv(RESEARCH / 'operational_hourly_summary.csv', hframe)
    v5_summary = _v5_d1_reference()
    historical_days = len({r['target_date'] for r in back_rows})
    if not historical_days and (RESEARCH / 'backtest_42_metrics.csv').exists():
        historical_days = int(pd.read_csv(RESEARCH / 'backtest_42_metrics.csv').target_date.nunique())
    if back_rows:
        bframe = pd.DataFrame(back_rows)
        # Paired summaries use exactly the same weekly target dates per horizon.
        group = bframe.groupby(['model', 'horizon_days'], as_index=False).agg(
            days=('target_date', 'nunique'), MAE=('MAE', 'mean'), RMSE=('RMSE', 'mean'),
            G2V_regret=('G2V_regret', 'mean'), V2G_regret=('V2G_regret', 'mean'),
            input_vintage_verified=('input_vintage_verified', 'all'))
        group['comparison_period'] = '2026-01-01 through 2026-10-15; weekly common targets'
        group['EV_conditions'] = 'same full-day EV(degradation=5), identical targets within horizon'
        write_csv(RESEARCH / 'backtest_42_summary.csv', group)
    status = dict(at=now(), operational_forecasts=len(saved), operational_unique_forecasts=len(canonical),
                  operational_duplicates_excluded=len(duplicates),
                  operational_evaluated=sum(r['status']=='evaluated' for r in rows),
                  operational_known_target_excluded=sum(r['status']=='excluded_target_prices_known_at_issue' for r in rows),
                  historical_backtest_days=historical_days,
                  historical_backtest_vintage_verified=False, historical_backtest_sampling='42 fixed weekly target dates from 2026-01-01 through 2026-10-15; reported separately from legacy scorecards', reference_conflicts=conflicts,
                  V5_results_preserved=True, V5_rescored_shared_conditions=len(v5_summary) > 0,
                  cross_period_improvement_claim_allowed=False)
    (RESEARCH / 'evaluation_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    return status


if __name__ == '__main__':
    print(evaluate())
