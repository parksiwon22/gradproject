"""Causal SMP horizon experiments using existing SMP history only.

All retrospective inputs are cut at each nominal 23:00 origin by timestamp.
The archive does not contain original publication vintages, so results are
explicitly research replays rather than verified operating forecasts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from v11.store import connect, latest
from v12.ev import EV
from v19.common import ROOT, write_csv

OUT = ROOT / 'docs' / 'smp_improvement'
EV_CONDITIONS = EV(degradation=5)
FOLDS = [
    dict(name='winter2024', calibration=('2024-07-01', '2024-09-30'), validation=('2024-10-01', '2024-12-31')),
    dict(name='spring2025', calibration=('2025-01-01', '2025-03-31'), validation=('2025-04-01', '2025-06-30')),
    dict(name='winter2025', calibration=('2025-07-01', '2025-09-30'), validation=('2025-10-01', '2025-12-31')),
]
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def load_prices():
    db = connect()
    try:
        frame = latest(db, 'land')
    finally:
        db.close()
    return frame.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)


def daily_vectors(prices, day, horizon):
    """Return labels and causal candidates for one full delivery day."""
    series = prices.set_index('timestamp').smp.sort_index()
    target = pd.date_range(pd.Timestamp(day, tz='Asia/Seoul'), periods=24, freq='h')
    origin = pd.Timestamp(day, tz='Asia/Seoul') - pd.Timedelta(days=horizon) + pd.Timedelta(hours=23)
    lag = series.reindex(target - pd.Timedelta(days=horizon)).to_numpy(dtype=float)
    weekly = np.vstack([series.reindex(target - pd.Timedelta(days=7 * k)).to_numpy(dtype=float)
                        for k in range(1, 5)]).T
    actual = series.reindex(target).to_numpy(dtype=float)
    source_times = [target - pd.Timedelta(days=horizon)] + [target - pd.Timedelta(days=7 * k) for k in range(1, 5)]
    max_input = max(pd.DatetimeIndex(x).max() for x in source_times)
    if max_input > origin:
        raise AssertionError(f'future input at {day} D+{horizon}: {max_input} > {origin}')
    if not (np.isfinite(actual).all() and np.isfinite(lag).all() and np.isfinite(weekly).all()):
        return None
    median4 = np.median(weekly, axis=1)
    median2 = np.median(weekly[:, :2], axis=1)
    median8_values = np.vstack([series.reindex(target - pd.Timedelta(days=7 * k)).to_numpy(dtype=float)
                                for k in range(1, 9)]).T
    mean4 = np.mean(weekly, axis=1)
    return dict(day=day, horizon=horizon, target=target, origin=origin, actual=actual,
                persistence=lag, median2=median2, median4=median4,
                median8=np.median(median8_values, axis=1), mean4=mean4,
                shift=float(np.mean(np.abs(lag - median4))))


def metrics(prediction, item):
    actual, target = item['actual'], item['target']
    # Full-day G2V optimum with a fixed energy request: rank hourly SMP,
    # allocate 7kW until 30kWh battery energy is supplied (33.333 grid kWh).
    # The fixed degradation term cancels between predicted and oracle schedules.
    grid_need = EV_CONDITIONS.capacity * (EV_CONDITIONS.target_soc - EV_CONDITIONS.initial_soc) / 100 / EV_CONDITIONS.charge_eff
    def allocate(prices):
        charge = np.zeros(24, dtype=float)
        remaining = grid_need
        for hour in np.argsort(prices, kind='stable'):
            amount = min(EV_CONDITIONS.charge_kw, remaining)
            charge[hour] = amount
            remaining -= amount
            if remaining <= 1e-10:
                break
        return charge
    selected = allocate(prediction)
    oracle = allocate(actual)
    regret = float(np.dot(actual, selected - oracle))
    peak = actual >= 150.0
    return dict(MAE=float(np.mean(np.abs(actual - prediction))),
                RMSE=float(np.sqrt(np.mean(np.square(actual - prediction)))), G2V_regret=float(regret),
                peak_hours=int(peak.sum()), peak_MAE=float(np.abs(actual[peak] - prediction[peak]).mean()) if peak.any() else np.nan)


def select_weight(calibration, anchor):
    baseline = [metrics(x['persistence'], x) for x in calibration]
    base_mae = float(np.mean([x['MAE'] for x in baseline]))
    base_regret = float(np.mean([x['G2V_regret'] for x in baseline]))
    candidates = []
    for weight in WEIGHTS:
        scored = [metrics((1 - weight) * x['persistence'] + weight * x[anchor], x) for x in calibration]
        mae = float(np.mean([x['MAE'] for x in scored]))
        rmse = float(np.sqrt(np.mean([x['RMSE'] ** 2 for x in scored])))
        regret = float(np.mean([x['G2V_regret'] for x in scored]))
        if mae <= base_mae + 1e-10 and regret <= base_regret + 1e-10:
            candidates.append((mae, regret, rmse, weight))
    return min(candidates)[-1] if candidates else 0.0


def select_anchor_weight(calibration, anchors):
    """Select a predeclared anchor/blend on calibration only, subject to both goals."""
    base = [metrics(x['persistence'], x) for x in calibration]
    base_mae = float(np.mean([x['MAE'] for x in base]))
    base_regret = float(np.mean([x['G2V_regret'] for x in base]))
    candidates = []
    for anchor in anchors:
        for weight in WEIGHTS:
            scored = [metrics((1 - weight) * x['persistence'] + weight * x[anchor], x)
                      for x in calibration]
            mae = float(np.mean([x['MAE'] for x in scored]))
            rmse = float(np.sqrt(np.mean([x['RMSE'] ** 2 for x in scored])))
            regret = float(np.mean([x['G2V_regret'] for x in scored]))
            if mae <= base_mae + 1e-10 and regret <= base_regret + 1e-10:
                candidates.append((mae, regret, rmse, anchor, weight))
    if not candidates:
        return 'persistence', 0.0
    selected = min(candidates)
    return selected[-2], selected[-1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    # Materialize only the declared replay/calibration windows. Each target uses
    # at most 8 weeks of earlier prices; the rest of the 1,396-day series is not
    # repeatedly reindexed in Python.
    study_days = [d.strftime('%Y-%m-%d') for d in pd.date_range('2024-07-01', '2025-12-31', freq='D')]
    study_days += [d.strftime('%Y-%m-%d') for d in pd.date_range('2026-07-01', '2026-09-30', freq='D')]
    frame_by_horizon = {h: {d: item for d in study_days if (item := daily_vectors(prices, d, h)) is not None}
                        for h in (1, 2, 3)}
    records, traces = [], []
    anchor_names = ('median2', 'median4', 'median8', 'mean4')
    for fold in FOLDS:
        cal_start, cal_end = fold['calibration']
        val_start, val_end = fold['validation']
        for horizon in (1, 2, 3):
            day_map = frame_by_horizon[horizon]
            calibration = [v for d, v in day_map.items() if cal_start <= d <= cal_end]
            validation = [v for d, v in day_map.items() if val_start <= d <= val_end]
            if len(validation) < 89 or len(calibration) < 89:
                raise RuntimeError(f'incomplete fold {fold["name"]} D+{horizon}: cal={len(calibration)}, val={len(validation)}')
            chosen = {anchor: select_weight(calibration, anchor) for anchor in anchor_names}
            selected_anchor, selected_weight = select_anchor_weight(calibration, anchor_names)
            # Predeclared primary hypothesis: median of the four previous same-weekday
            # prices, blended with the horizon-lag profile. Dynamic alpha is a second,
            # separate regime-response experiment, never fitted to validation labels.
            w4 = chosen['median4']
            shift_threshold = float(np.median([x['shift'] for x in calibration]))
            for model in ('persistence', 'weekday_median2', 'weekday_median4', 'weekday_median8',
                          'weekday_mean4', 'calibrated_median4_blend', 'calibration_selected_anchor_blend',
                          'adaptive_median4_blend'):
                by_day = []
                for item in validation:
                    if model == 'persistence':
                        prediction = item['persistence']
                    elif model == 'weekday_median2':
                        prediction = item['median2']
                    elif model == 'weekday_median4':
                        prediction = item['median4']
                    elif model == 'weekday_median8':
                        prediction = item['median8']
                    elif model == 'weekday_mean4':
                        prediction = item['mean4']
                    elif model == 'calibrated_median4_blend':
                        prediction = (1 - w4) * item['persistence'] + w4 * item['median4']
                    elif model == 'calibration_selected_anchor_blend':
                        prediction = ((1 - selected_weight) * item['persistence'] +
                                      selected_weight * item[selected_anchor])
                    else:
                        weight = w4 * min(1.0, shift_threshold / max(item['shift'], 1e-9))
                        prediction = (1 - weight) * item['persistence'] + weight * item['median4']
                    stat = metrics(prediction, item)
                    by_day.append(stat)
                    traces.extend(dict(fold=fold['name'], model=model, horizon_days=horizon,
                                       target_date=item['day'], hour=int(hour), timestamp=item['target'][hour].isoformat(),
                                       actual_smp=float(item['actual'][hour]), predicted_smp=float(prediction[hour]),
                                       error=float(prediction[hour] - item['actual'][hour])) for hour in range(24))
                predictions = []
                # Recompute prices for pooled hourly metrics; day-level regret is averaged.
                errors = []
                peak_abs = []
                for item in validation:
                    if model == 'persistence': prediction = item['persistence']
                    elif model == 'weekday_median2': prediction = item['median2']
                    elif model == 'weekday_median4': prediction = item['median4']
                    elif model == 'weekday_median8': prediction = item['median8']
                    elif model == 'weekday_mean4': prediction = item['mean4']
                    elif model == 'calibrated_median4_blend': prediction = (1 - w4) * item['persistence'] + w4 * item['median4']
                    elif model == 'calibration_selected_anchor_blend': prediction = ((1 - selected_weight) * item['persistence'] + selected_weight * item[selected_anchor])
                    else:
                        weight = w4 * min(1.0, shift_threshold / max(item['shift'], 1e-9))
                        prediction = (1 - weight) * item['persistence'] + weight * item['median4']
                    errors.extend(prediction - item['actual'])
                    peak = item['actual'] >= 150.0
                    peak_abs.extend(np.abs(prediction[peak] - item['actual'][peak]))
                    for hour in range(24):
                        row = hourly_by_key.setdefault((fold['name'], horizon, model, hour), [])
                        row.append(float(prediction[hour] - item['actual'][hour]))
                records.append(dict(fold=fold['name'], horizon_days=horizon, model=model, days=len(validation),
                    MAE=float(np.mean(np.abs(errors))), RMSE=float(np.sqrt(np.mean(np.square(errors)))),
                    G2V_regret=float(np.mean([x['G2V_regret'] for x in by_day])),
                    peak_hours=int(sum(x['peak_hours'] for x in by_day)),
                    peak_MAE=float(np.mean(peak_abs)) if peak_abs else np.nan,
                    calibration_weight_median2=chosen['median2'], calibration_weight_median4=w4,
                    calibration_weight_median8=chosen['median8'], calibration_weight_mean4=chosen['mean4'],
                    selected_anchor=selected_anchor, selected_anchor_weight=selected_weight,
                    calibration_days=len(calibration), validation_start=val_start, validation_end=val_end,
                    data_vintage_verified=False))

    summary = pd.DataFrame(records)
    write_csv(OUT / 'walk_forward_metrics.csv', summary)
    write_csv(OUT / 'walk_forward_predictions.csv', pd.DataFrame(traces))
    hourly_rows = []
    for (fold, horizon, model, hour), values in hourly_by_key.items():
        hourly_rows.append(dict(fold=fold, horizon_days=horizon, model=model, hour=hour,
                                MAE=float(np.mean(np.abs(values))), bias=float(np.mean(values)), samples=len(values)))
    write_csv(OUT / 'hourly_errors.csv', pd.DataFrame(hourly_rows))

    config = dict(schema=1, algorithm='causal_rolling_calibrated_weekday_anchor_blend',
        forecast_origin='nominal 23:00 Asia/Seoul; the caller must supply an as-of price snapshot',
        horizons=[1, 2, 3],
        calibration=dict(window_days=90, exclude_issue_date=True,
            anchors={'median2': [7, 14], 'median4': [7, 14, 21, 28],
                     'median8': list(range(7, 57, 7)), 'mean4': [7, 14, 21, 28]},
            weights=list(WEIGHTS),
            rule='On the 90 completed target days ending before the issue date, select anchor and weight minimizing average daily MAE subject to both MAE and G2V regret not exceeding horizon persistence; ties by regret then RMSE. If no alternative qualifies, use persistence.'),
        actual_threshold_for_peak=150.0,
        EV_conditions=dict(capacity_kwh=60, initial_soc_percent=30, target_soc_percent=80,
            charge_kw=7, efficiency=0.9, degradation=5, window='full delivery day'),
        input_policy='Filter input snapshot to first_seen<=origin when receipt metadata exists and timestamp<=origin; all calibration labels end before issue date; every feature lag is checked <= the nominal 23:00 origin.',
        availability='historical source first_seen timestamps are mainly 2026-10-15; original historical publication vintages cannot be verified',
        selection_note='All historical ranges have prior project exposure; walk-forward results are repeated retrospective validation, not an independent final test; do not replace V20 automatically',
        folds=FOLDS)
    (OUT / 'final_candidate.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'experiment_log.json').write_text(json.dumps(dict(
        hypothesis_1='Median of the previous four same-weekday hourly prices is a more robust seasonal anchor than one target-horizon lag when SMP levels shift.',
        hypothesis_2='Calibrate one fixed horizon-lag/seasonal-anchor blend on the immediately preceding quarter using MAE and G2V regret constraints; anchor candidates are fixed before scoring.',
        hypothesis_3='Reduce the seasonal-median weight on days when the recent profile sharply diverges from the seasonal median; threshold is the median divergence in calibration.',
        rejected_or_diagnostic=['2-week and 8-week medians', '4-week mean', 'adaptive blend'],
        prior_experiments='V14 recency and shape, V15 event and two-stage models, V17 joint/ranking, V19 separate D3 HGB reviewed; no retraining or prior artifact overwritten.',
        validation_caveat='2024Q4, 2025Q2, 2025Q4 were previously examined; use as replayed folds only.',
        results_rows=len(summary), final_model_promoted=False), ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved', len(summary), 'fold/model rows; rolling calibrated shadow recipe written.')


hourly_by_key = {}

if __name__ == '__main__':
    main()
