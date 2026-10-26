"""Combine already-saved V5/V11-V20 forecast evaluations without retraining."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from v11.store import connect, latest
from v12.ev import EV
from v18.evaluation import compare_ev, price_metrics, reference
from v19.common import ROOT, write_csv
from v20.research import _cutoff_prices

RESEARCH = ROOT / 'artifacts/v20/research'
FINAL = ROOT / 'docs/final_project'


def _local_times(values):
    parsed = pd.to_datetime(values, utc=True, format='mixed')
    return parsed.dt.tz_convert('Asia/Seoul')


def _saved_evaluation_rows():
    """Use the latest existing V19 same-price score per saved forecast/day."""
    collected = []
    for path in (ROOT / 'artifacts/v19/evaluations').glob('*.json'):
        try:
            document = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        for row in document.get('rows', []):
            if row.get('status') != 'official_provided_price_comparison':
                continue
            item = dict(row)
            item['evaluation_at'] = document.get('at', '')
            collected.append(item)
    if not collected:
        return {}
    frame = pd.DataFrame(collected).sort_values('evaluation_at')
    frame = frame.drop_duplicates(['file', 'target_date'], keep='last')
    return {(str(r.file), str(r.target_date)): r._asdict() for r in frame.itertuples(index=False)}


def _known_rows_at_issue(day, issued, cache):
    key = pd.Timestamp(issued).isoformat()
    if key not in cache:
        cache[key] = _cutoff_prices(pd.Timestamp(issued))
    frame = cache[key]
    return int(frame.attrs.get('available_future_timestamp_counts', {}).get(day, 0))


def _baseline(prices, target, horizon):
    source = pd.DatetimeIndex(target) - pd.Timedelta(days=int(horizon))
    indexed = prices.set_index('timestamp').smp
    values = indexed.reindex(source).to_numpy(dtype=float)
    return source, values


def build_saved_evaluation():
    """Create a new, separate analysis set from saved forecasts and prior scores."""
    RESEARCH.mkdir(parents=True, exist_ok=True)
    # A price row for a future delivery day may be a published/indicative price,
    # not a realized outcome. Only score fully completed delivery dates.
    evaluation_cutoff_date = pd.Timestamp.now(tz='Asia/Seoul').normalize()
    actual, conflicts = reference()
    actual = actual[['timestamp', 'smp', 'source']].drop_duplicates('timestamp').sort_values('timestamp')
    actual_index = actual.set_index('timestamp').smp
    actual_source = actual.set_index('timestamp').source
    db = connect()
    try:
        latest_all = latest(db, 'land')
    finally:
        db.close()
    latest_all = latest_all[['timestamp', 'smp', 'first_seen']].drop_duplicates('timestamp')
    saved_scores = _saved_evaluation_rows()
    v20_metrics_path = RESEARCH / 'operational_metrics.csv'
    v20_scores = {}
    if v20_metrics_path.exists():
        metrics_frame = pd.read_csv(v20_metrics_path)
        v20_scores = {(str(r.file), str(r.target_date)): r._asdict()
                      for r in metrics_frame.itertuples(index=False)}
    cutoff_cache = {}
    summary_rows, hourly_rows, price_rows, manifest = [], [], [], []
    baseline_vectors = {}
    seen_forecasts = set()

    # Read the actual archived forecast vectors once. V19's existing scorecards
    # supply the saved-prediction MAE/RMSE/regret; only matched baselines are new.
    for version in range(11, 21):
        folder = ROOT / f'artifacts/v{version}/forecasts'
        folders = [folder]
        if version == 20:
            folders = [ROOT / 'artifacts/v20/research/forecasts']
        paths = sorted(path for folder in folders if folder.exists() for path in folder.glob('*.json'))
        for path in paths:
            try:
                record = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                continue
            raw_rows = record.get('price_rows') or record.get('rows') or []
            if not raw_rows:
                continue
            forecast = pd.DataFrame(raw_rows)
            if 'timestamp' not in forecast or 'predicted_smp' not in forecast:
                continue
            forecast['timestamp'] = _local_times(forecast.timestamp)
            issue_raw = record.get('issued_at') or record.get('reference_origin')
            origin_raw = record.get('reference_origin') or issue_raw
            issued = pd.Timestamp(issue_raw).tz_convert('Asia/Seoul')
            origin = pd.Timestamp(origin_raw).tz_convert('Asia/Seoul')
            model = record.get('model') or record.get('model_family') or path.stem
            for day, block in forecast.groupby(forecast.timestamp.dt.strftime('%Y-%m-%d')):
                block = block.sort_values('timestamp')
                horizon = int((pd.Timestamp(day, tz='Asia/Seoul') - origin.normalize()).days)
                file_key = (version, path.name, day, horizon)
                target = pd.date_range(pd.Timestamp(day, tz='Asia/Seoul'), periods=24, freq='h')
                pred = block.predicted_smp.to_numpy(dtype=float)
                known_count = _known_rows_at_issue(day, issued, cutoff_cache)
                known_flag = (known_count > 0 or record.get('official_smp_known_at_issue') is True or
                    record.get('official_known_at_issue') is True or record.get('mode') == 'official_known' or
                    any(pd.notna(x.get('official_smp_at_issue')) for x in raw_rows))
                issue_prices = cutoff_cache[pd.Timestamp(issued).isoformat()]
                source, baseline_pred = _baseline(issue_prices, target, horizon) if 1 <= horizon <= 3 else (pd.DatetimeIndex([]), np.array([]))
                baseline_ready = len(baseline_pred) == 24 and np.isfinite(baseline_pred).all()
                observed = actual_index.reindex(target).to_numpy(dtype=float)
                delivery_completed = pd.Timestamp(day, tz='Asia/Seoul') < evaluation_cutoff_date
                score = saved_scores.get((path.name, day))
                dedup_key = (version, origin.strftime('%Y-%m-%d'), day, horizon, model)
                duplicate_retry = dedup_key in seen_forecasts
                seen_forecasts.add(dedup_key)
                if duplicate_retry:
                    status = 'duplicate_retry_excluded'
                elif len(block) != 24 or block.timestamp.duplicated().any() or not np.isfinite(pred).all():
                    status = 'malformed_or_partial_saved_forecast'
                elif known_flag:
                    status = 'excluded_target_price_known_at_issue'
                elif not delivery_completed:
                    status = 'awaiting_completed_delivery_day'
                elif not np.isfinite(observed).all():
                    status = 'awaiting_actual_price'
                elif version == 20:
                    score = v20_scores.get((path.name, day))
                    status = ('evaluated_saved_forecast' if score and score.get('status') == 'evaluated'
                              else 'excluded_target_price_known_at_issue' if score and score.get('status') == 'excluded_target_prices_known_at_issue'
                              else 'awaiting_or_not_scored')
                elif score is None and len(block) == 24 and np.isfinite(observed).all():
                    status = 'evaluated_saved_forecast'
                else:
                    status = 'evaluated_saved_forecast'

                inputs = record.get('source_prices') or []
                if inputs:
                    receipts = pd.to_datetime([x.get('first_seen') for x in inputs], utc=True, errors='coerce')
                    input_vintage_verified = bool(receipts.notna().all() and (receipts <= issued.tz_convert('UTC')).all()
                                                  and all(pd.Timestamp(x['timestamp']) <= issued for x in inputs))
                elif record.get('input_max_timestamp') and record.get('input_max_first_seen'):
                    input_vintage_verified = (pd.Timestamp(record['input_max_timestamp']) <= issued and
                        pd.Timestamp(record['input_max_first_seen']).tz_convert('UTC') <= issued.tz_convert('UTC'))
                elif record.get('inputs_first_seen_max') or record.get('input_first_seen_max'):
                    receipt = pd.Timestamp(record.get('inputs_first_seen_max', record.get('input_first_seen_max')))
                    input_vintage_verified = receipt.tz_convert('UTC') <= issued.tz_convert('UTC')
                else:
                    input_vintage_verified = False
                model_locked = record.get('selection_locked_at') or record.get('model_registered_at')
                if model_locked:
                    model_available = pd.Timestamp(model_locked).tz_convert('UTC') <= issued.tz_convert('UTC')
                elif 'model_locked_after_reference_origin' in record:
                    model_available = not bool(record['model_locked_after_reference_origin'])
                else:
                    model_available = None
                manifest.append(dict(version=version, file=path.name, model=model, target_date=day,
                    horizon_days=horizon, issued_at=issued.isoformat(), reference_origin=origin.isoformat(),
                    issue_delay_minutes=(issued-origin).total_seconds()/60,
                    model_available_at_reference_origin=(pd.Timestamp(model_locked).tz_convert('UTC') <= origin.tz_convert('UTC')) if model_locked else (not bool(record.get('model_locked_after_reference_origin')) if 'model_locked_after_reference_origin' in record else None),
                    issue_known_target_rows=known_count, saved_known_flag=known_flag,
                    inputs_vintage_verified=input_vintage_verified, model_available_at_issue=model_available,
                    baseline_available_at_issue=baseline_ready, status=status,
                    evaluation_source='existing V19 score index' if version < 20 else 'existing V20 operational scorecard'))
                if status != 'evaluated_saved_forecast' or horizon not in (1, 2, 3):
                    continue

                if version == 20:
                    score_row = pd.read_csv(RESEARCH / 'operational_metrics.csv')
                    match = score_row[(score_row.file == path.name) & (score_row.target_date.astype(str) == day)]
                    if match.empty or match.iloc[0]['status'] != 'evaluated':
                        continue
                    score = match.iloc[0].to_dict()
                elif score is None:
                    metrics = price_metrics(observed, pred)
                    ev = compare_ev(pred, observed, target, ev=EV(degradation=5))
                    score = dict(**metrics, G2V_regret=ev['G2V_regret'], V2G_regret=ev['V2G_regret'])
                entry = dict(cohort='saved_issue_forecasts', version=version, model=model,
                    horizon_days=horizon, target_date=day, issued_at=issued.isoformat(),
                    status=status, evaluation_data='previously stored forecast vector',
                    actual_basis='official provided SMP; later final/corrected settlement status unavailable',
                    input_vintage_verified=input_vintage_verified, model_available_at_issue=model_available,
                    issue_delay_minutes=(issued-origin).total_seconds()/60,
                    model_available_at_reference_origin=(pd.Timestamp(model_locked).tz_convert('UTC') <= origin.tz_convert('UTC')) if model_locked else (not bool(record.get('model_locked_after_reference_origin')) if 'model_locked_after_reference_origin' in record else None),
                    target_prices_known_at_issue=False, comparison_group=f'D{horizon}:{day}',
                    MAE=float(score['MAE']), RMSE=float(score['RMSE']),
                    G2V_regret=float(score.get('G2V_regret', score.get('G2V_regret'))),
                    V2G_regret=float(score.get('V2G_regret', np.nan)))
                summary_rows.append(entry)
                forecast_id = f'V{version} · {model} · D+{horizon} · {day}'
                price_rows.extend(dict(forecast_id=forecast_id, version=version, model=model,
                    demo_cohort='saved_issue_forecasts', horizon_days=horizon, target_date=day, hour=t.hour, timestamp=t.isoformat(),
                    series='기준선 SMP' if 'baseline' in model else '예측 SMP', smp=float(p), evaluation_type='저장 예측')
                    for t, p in zip(target, pred))
                price_rows.extend(dict(forecast_id=forecast_id, version=version, model=model,
                    demo_cohort='saved_issue_forecasts', horizon_days=horizon, target_date=day, hour=t.hour, timestamp=t.isoformat(),
                    series='실제 SMP', smp=float(a), evaluation_type='공식 제공가격 비교')
                    for t, a in zip(target, observed))
                forecast_details = block.to_dict('records')
                for t, detail in zip(target, forecast_details):
                    for item in price_rows[-48:]:
                        if item.get('forecast_id') == forecast_id and item.get('timestamp') == t.isoformat() and item.get('series') != '실제 SMP':
                            item['risk_level'] = detail.get('high_price_risk', detail.get('risk_level'))
                            item['spike_probability'] = detail.get('spike_probability', detail.get('high_price_probability'))
                            item['price_weather'] = detail.get('price_weather', record.get('price_weather'))
                            item['weather_forecast_used'] = record.get('weather_forecast_used')
                for t, a, p in zip(target, observed, pred):
                    hourly_rows.append(dict(forecast_id=forecast_id, version=version, model=model,
                        horizon_days=horizon, target_date=day, hour=t.hour, actual_smp=float(a),
                        predicted_smp=float(p), error=float(p-a), absolute_error=float(abs(p-a)),
                        actual_source=actual_source.get(t, 'unknown')))

                pair_key = (day, horizon)
                if baseline_ready and pair_key not in baseline_vectors:
                    base_actual = actual_index.reindex(target).to_numpy(dtype=float)
                    if np.isfinite(base_actual).all():
                        bmetrics = price_metrics(base_actual, baseline_pred)
                        bev = compare_ev(baseline_pred, base_actual, target, ev=EV(degradation=5))
                        baseline_vectors[pair_key] = dict(cohort='saved_issue_forecasts', version='replayed',
                            model='same_hour_available_at_issue_baseline', horizon_days=horizon,
                            target_date=day, issued_at=issued.isoformat(), status='evaluated_replayed_baseline',
                            evaluation_data='reconstructed from issue-time available SMP',
                            actual_basis='official provided SMP; later final/corrected settlement status unavailable',
                            input_vintage_verified=True, model_available_at_issue=True,
                            target_prices_known_at_issue=False, comparison_group=f'D{horizon}:{day}',
                            MAE=bmetrics['MAE'], RMSE=bmetrics['RMSE'], G2V_regret=bev['G2V_regret'],
                            V2G_regret=bev['V2G_regret'])
                        if 'baseline' not in model:
                            price_rows.extend(dict(forecast_id=forecast_id, version='replayed',
                                demo_cohort='saved_issue_forecasts', model='same_hour_available_at_issue_baseline', horizon_days=horizon,
                                target_date=day, hour=t.hour, timestamp=t.isoformat(), series='기준선 SMP',
                                smp=float(p), evaluation_type='발급시점 입력 재현') for t, p in zip(target, baseline_pred))
                elif baseline_ready and 'baseline' not in model:
                    price_rows.extend(dict(forecast_id=forecast_id, version='replayed',
                        demo_cohort='saved_issue_forecasts', model='same_hour_available_at_issue_baseline', horizon_days=horizon,
                        target_date=day, hour=t.hour, timestamp=t.isoformat(), series='기준선 SMP',
                        smp=float(p), evaluation_type='발급시점 입력 재현') for t, p in zip(target, baseline_pred))
                elif not baseline_ready:
                    entry['paired_baseline_status'] = 'insufficient_issue_time_history'

    # Reuse the already-produced V5 92-day and V20 42-day summaries verbatim.
    v5_path = RESEARCH / 'v5_d1_reference_summary.csv'
    if v5_path.exists():
        v5 = pd.read_csv(v5_path)
        for row in v5.to_dict('records'):
            summary_rows.append(dict(cohort='V5_saved_92day_test', version=5,
                model=row['model'], horizon_days=int(row['horizon_days']), target_date='',
                days=int(row['days']), status='evaluated_reused_existing_summary',
                evaluation_data='existing V5 test predictions',
                actual_basis='V5 saved actual SMP; final correction status unverified',
                input_vintage_verified=False, model_available_at_issue=None,
                target_prices_known_at_issue=None, comparison_group='2023-10-01..2023-12-31; same 92 dates',
                MAE=row['MAE'], RMSE=row['RMSE'], G2V_regret=row['G2V_regret'],
                V2G_regret=row['V2G_regret']))
    v20_path = RESEARCH / 'backtest_42_summary.csv'
    if v20_path.exists():
        v20 = pd.read_csv(v20_path)
        for row in v20.to_dict('records'):
            summary_rows.append(dict(cohort='V20_42day_historical_replay', version=20,
                model=row['model'], horizon_days=int(row['horizon_days']), target_date='',
                days=int(row['days']), status='evaluated_reused_existing_summary',
                evaluation_data='existing V20 42-day replay',
                actual_basis='official price history; final correction status unverified',
                input_vintage_verified=bool(row['input_vintage_verified']), model_available_at_issue=False,
                target_prices_known_at_issue=None, comparison_group='2026-01-01..2026-10-15; same 42 weekly dates',
                MAE=row['MAE'], RMSE=row['RMSE'], G2V_regret=row['G2V_regret'],
                V2G_regret=row['V2G_regret']))

    # Include V5 and V20 replay vectors in the interactive plot data.
    v5_forecast = ROOT / 'artifacts/results/v5_day_ahead_predictions.csv'
    if v5_forecast.exists():
        frame = pd.read_csv(v5_forecast)
        for model, col in [('V5_profile_hgb_blend', 'profile_hgb_blend'),
                           ('V5_daily_profile_baseline', 'daily_profile_baseline')]:
            for day, block in frame.groupby('target_date'):
                block = block.sort_values('timestamp')
                forecast_id = f'{model} · D+1 · {day}'
                for row in block.itertuples():
                    hour = pd.Timestamp(row.timestamp).hour
                    price_rows.append(dict(forecast_id=forecast_id, version=5, model=model,
                        demo_cohort='V5_saved_92day_test', horizon_days=1, target_date=str(day), hour=hour, timestamp=str(row.timestamp),
                        series='기준선 SMP' if model == 'V5_daily_profile_baseline' else '예측 SMP',
                        smp=float(getattr(row, col)), risk_level=getattr(row, 'risk_level', None),
                        spike_probability=getattr(row, 'ramp_probability', None),
                        price_weather=getattr(row, 'price_weather', None), evaluation_type='보존 V5 테스트'))
                    price_rows.append(dict(forecast_id=forecast_id, version=5, model=model,
                        demo_cohort='V5_saved_92day_test', horizon_days=1, target_date=str(day), hour=hour, timestamp=str(row.timestamp),
                        series='실제 SMP', smp=float(row.actual_smp), evaluation_type='보존 V5 테스트'))
                    if model == 'V5_profile_hgb_blend':
                        price_rows.append(dict(forecast_id=forecast_id, version=5,
                            demo_cohort='V5_saved_92day_test', model='V5_daily_profile_baseline', horizon_days=1, target_date=str(day),
                            hour=hour, timestamp=str(row.timestamp), series='기준선 SMP',
                            smp=float(row.daily_profile_baseline), evaluation_type='동일 92일 V5 기준선'))
    v20_hourly = RESEARCH / 'backtest_42_hourly_errors.csv'
    if v20_hourly.exists():
        frame = pd.read_csv(v20_hourly)
        for row in frame.itertuples():
            forecast_id = f'V20 replay · {row.model} · D+{row.horizon_days} · {row.target_date}'
            is_baseline = row.model == 'same_hour_previous_day_baseline'
            price_rows.extend([
                dict(forecast_id=forecast_id, version=20, model=row.model, horizon_days=row.horizon_days,
                     demo_cohort='V20_42day_historical_replay', target_date=str(row.target_date), hour=row.hour, timestamp=row.timestamp,
                     series='기준선 SMP' if is_baseline else '예측 SMP', smp=row.predicted_smp, evaluation_type='V20 42일 과거 재현'),
                dict(forecast_id=forecast_id, version=20, model=row.model, horizon_days=row.horizon_days,
                     demo_cohort='V20_42day_historical_replay', target_date=str(row.target_date), hour=row.hour, timestamp=row.timestamp,
                     series='실제 SMP', smp=row.actual_smp, evaluation_type='V20 42일 과거 재현')])
            if row.model == 'V17_joint':
                baseline = frame[(frame.target_date.astype(str) == str(row.target_date)) &
                                 (frame.horizon_days == row.horizon_days) &
                                 (frame.model == 'same_hour_previous_day_baseline') & (frame.hour == row.hour)]
                if len(baseline):
                    price_rows.append(dict(forecast_id=forecast_id, version=20,
                        demo_cohort='V20_42day_historical_replay', model='same_hour_previous_day_baseline', horizon_days=row.horizon_days,
                        target_date=str(row.target_date), hour=row.hour, timestamp=row.timestamp,
                        series='기준선 SMP', smp=float(baseline.iloc[0].predicted_smp),
                        evaluation_type='V20 42일 과거 재현'))
    summary_rows.extend(baseline_vectors.values())

    write_csv(RESEARCH / 'final_saved_comparison_summary.csv', pd.DataFrame(summary_rows))
    write_csv(RESEARCH / 'final_saved_comparison_hourly.csv', pd.DataFrame(hourly_rows))
    write_csv(RESEARCH / 'final_saved_comparison_prices.csv', pd.DataFrame(price_rows))
    write_csv(RESEARCH / 'final_saved_comparison_manifest.csv', pd.DataFrame(manifest))

    # Same-date paired rows, including issue-time replay baselines.
    scored = pd.DataFrame(summary_rows)
    shared_rows = []
    archive = scored[(scored.cohort == 'saved_issue_forecasts') &
                     (scored.status == 'evaluated_saved_forecast')]
    if len(archive):
        pair_keys = set(zip(archive.target_date, archive.horizon_days))
        shared_rows = archive[[(day, h) in pair_keys for day, h in zip(archive.target_date, archive.horizon_days)]].to_dict('records')
        shared_rows += [x for x in baseline_vectors.values() if (x['target_date'], x['horizon_days']) in pair_keys]
    write_csv(RESEARCH / 'final_saved_comparison_shared_dates.csv', pd.DataFrame(shared_rows))
    if len(scored):
        archive = scored[scored.cohort.eq('saved_issue_forecasts')]
        aggregate = scored[~scored.cohort.eq('saved_issue_forecasts')].copy()
        archive_summary = archive.groupby(['cohort', 'version', 'model', 'horizon_days'],
            dropna=False, as_index=False).agg(days=('target_date', 'nunique'), MAE=('MAE', 'mean'),
            RMSE=('RMSE', 'mean'), G2V_regret=('G2V_regret', 'mean'), V2G_regret=('V2G_regret', 'mean'),
            input_vintage_verified=('input_vintage_verified', 'all'),
            model_available_at_issue=('model_available_at_issue', 'first'),
            model_available_at_reference_origin=('model_available_at_reference_origin', 'first'),
            issue_delay_minutes=('issue_delay_minutes', 'mean'))
        model_summary = pd.concat([archive_summary, aggregate], ignore_index=True, sort=False)
        write_csv(RESEARCH / 'final_saved_comparison_model_summary.csv', model_summary)
    _write_final_assets()
    return dict(saved_issue_forecasts=len(manifest), scored_saved_forecasts=sum(r.get('status') == 'evaluated_saved_forecast' for r in manifest),
                target_known_excluded=sum(r.get('status') == 'excluded_target_price_known_at_issue' for r in manifest),
                existing_evaluations_reused=len(saved_scores), baseline_pairs=len(baseline_vectors),
                output_rows=len(summary_rows), reference_conflicts=len(conflicts))


def _write_final_assets():
    """Export new, compact final-project files; source evaluation files stay untouched."""
    FINAL.mkdir(parents=True, exist_ok=True)
    summary_path = RESEARCH / 'final_saved_comparison_model_summary.csv'
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        write_csv(FINAL / 'model_performance_comparison.csv', summary)
        write_csv(FINAL / 'g2v_regret_comparison.csv', summary[
            [c for c in ['cohort', 'version', 'model', 'horizon_days', 'days', 'G2V_regret', 'V2G_regret'] if c in summary]])

    hourly = []
    v5_path = ROOT / 'artifacts/results/v5_day_ahead_predictions.csv'
    if v5_path.exists():
        v5 = pd.read_csv(v5_path)
        for name, column in [('V5_profile_hgb_blend', 'profile_hgb_blend'),
                             ('V5_daily_profile_baseline', 'daily_profile_baseline')]:
            v5['error'] = v5[column] - v5.actual_smp
            grouped = v5.groupby(v5.timestamp.astype(str).str[-5:-3]).error.agg(['count', 'mean'])
            for hour, row in grouped.iterrows():
                subset = v5[v5.timestamp.astype(str).str[-5:-3].eq(hour)]
                hourly.append(dict(cohort='V5_saved_92day_test', version=5, model=name,
                    horizon_days=1, hour=int(hour), samples=int(row['count']),
                    MAE=float(abs(subset.error).mean()), bias=float(row['mean'])))
    v20_path = RESEARCH / 'backtest_42_hourly_errors.csv'
    if v20_path.exists():
        v20 = pd.read_csv(v20_path)
        for (model, horizon, hour), group in v20.groupby(['model', 'horizon_days', 'hour']):
            hourly.append(dict(cohort='V20_42day_historical_replay', version=20, model=model,
                horizon_days=int(horizon), hour=int(hour), samples=len(group),
                MAE=float(group.absolute_error.mean()), bias=float(group.error.mean())))
    saved_hourly_path = RESEARCH / 'final_saved_comparison_hourly.csv'
    if saved_hourly_path.exists():
        saved = pd.read_csv(saved_hourly_path)
        for (version, model, horizon, hour), group in saved.groupby(['version', 'model', 'horizon_days', 'hour']):
            hourly.append(dict(cohort='saved_issue_forecasts', version=version, model=model,
                horizon_days=int(horizon), hour=int(hour), samples=len(group),
                MAE=float(group.absolute_error.mean()), bias=float(group.error.mean())))
    hourly_frame = pd.DataFrame(hourly)
    write_csv(FINAL / 'hourly_error_comparison.csv', hourly_frame)

    # A representative, actual saved D+2 forecast with its matched baseline.
    prices_path = RESEARCH / 'final_saved_comparison_prices.csv'
    if prices_path.exists():
        prices = pd.read_csv(prices_path)
        forecast_id = 'V12 · lightgbm_peak · D+2 · 2026-10-18'
        demo = prices[prices.forecast_id.astype(str).eq(forecast_id)].copy()
        if len(demo):
            write_csv(FINAL / 'representative_smp_series.csv', demo)
            fig, ax = plt.subplots(figsize=(10, 5.4), constrained_layout=True)
            labels = {'실제 SMP': 'Actual SMP', '예측 SMP': 'Saved V12 LightGBM forecast', '기준선 SMP': 'Issue-time same-hour baseline'}
            for series, block in demo.groupby('series'):
                ax.plot(block.hour, block.smp, marker='o', markersize=3, linewidth=1.8,
                        label=labels.get(series, series))
            ax.set(title='D+2 saved forecast vs official provided SMP · 2026-10-18',
                   xlabel='Hour (KST)', ylabel='SMP (KRW/kWh)', xticks=range(24))
            ax.grid(alpha=.25); ax.legend(); fig.savefig(FINAL / 'representative_smp_comparison.png', dpi=170); plt.close(fig)

    if len(hourly_frame):
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
        for h, ax in zip((1, 2, 3), axes):
            data = hourly_frame[hourly_frame.horizon_days.eq(h)]
            for (cohort, model), block in data.groupby(['cohort', 'model']):
                if cohort == 'saved_issue_forecasts' and len(block) < 24:
                    continue
                label = f'{cohort}: {model}'
                ax.plot(block.hour, block.MAE, marker='.', linewidth=1.2, label=label)
            ax.set(title=f'D+{h} hourly MAE by evaluation cohort', xlabel='Hour (KST)',
                   ylabel='MAE (KRW/kWh)', xticks=range(24)); ax.grid(alpha=.25)
            if ax.lines: ax.legend(fontsize=7, ncol=2)
        fig.savefig(FINAL / 'hourly_error_comparison.png', dpi=160); plt.close(fig)

    summary_path = RESEARCH / 'final_saved_comparison_model_summary.csv'
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        cohorts = [('V5_saved_92day_test', 'V5 saved 92-day test'),
                   ('V20_42day_historical_replay', 'V20 42-day replay'),
                   ('saved_issue_forecasts', 'Saved issue forecasts')]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
        for ax, (cohort, title) in zip(axes, cohorts):
            group = summary[summary.cohort.eq(cohort)].dropna(subset=['G2V_regret'])
            if len(group):
                labels = [f"D{int(r.horizon_days)} {r.model}" for r in group.itertuples()]
                ax.barh(labels, group.G2V_regret, color='#3976a8')
                ax.invert_yaxis(); ax.set_xlabel('G2V regret (KRW)')
            ax.set_title(title); ax.grid(axis='x', alpha=.25)
        fig.savefig(FINAL / 'g2v_regret_comparison.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    print(build_saved_evaluation())
