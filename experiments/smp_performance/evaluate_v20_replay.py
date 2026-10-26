"""Score a rolling-calibrated shadow candidate on the existing V20 42-day set.

This is a timestamp-causal replay only: original price first-seen vintages and
historical model availability are not established. Existing V20 files are read
without modification.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.smp_performance.run_study import daily_vectors, load_prices, metrics, select_anchor_weight
from v19.common import ROOT, write_csv

OUT = ROOT / 'docs' / 'smp_improvement'
SOURCE = ROOT / 'artifacts' / 'v20' / 'research' / 'backtest_42_hourly_errors.csv'
ANCHORS = ('median2', 'median4', 'median8', 'mean4')


def main():
    saved = pd.read_csv(SOURCE)
    prices = load_prices()
    cache = {}
    predictions, summaries = [], []
    for horizon in (1, 2, 3):
        cohort = saved[saved.horizon_days.eq(horizon)]
        days = sorted(cohort.target_date.astype(str).unique())
        for day in days:
            origin_day = pd.Timestamp(day) - pd.Timedelta(days=horizon)
            cal_end = origin_day - pd.Timedelta(days=1)
            cal_start = cal_end - pd.Timedelta(days=89)
            for d in pd.date_range(cal_start, cal_end, freq='D'):
                key = (horizon, d.strftime('%Y-%m-%d'))
                if key not in cache:
                    cache[key] = daily_vectors(prices, key[1], horizon)
            calibration = [cache[(horizon, d.strftime('%Y-%m-%d'))]
                           for d in pd.date_range(cal_start, cal_end, freq='D')
                           if cache[(horizon, d.strftime('%Y-%m-%d'))] is not None]
            if len(calibration) < 85:
                raise RuntimeError(f'not enough 90-day calibration history at {day} D+{horizon}')
            anchor, weight = select_anchor_weight(calibration, ANCHORS)
            item = daily_vectors(prices, day, horizon)
            if item is None:
                raise RuntimeError(f'incomplete SMP at target {day}')
            target_rows = cohort[cohort.target_date.astype(str).eq(day)]
            base_row = target_rows[target_rows.model.eq('same_hour_previous_day_baseline')].sort_values('hour')
            if len(base_row) != 24:
                raise RuntimeError(f'missing archived baseline for {day} D+{horizon}')
            actual = base_row.actual_smp.to_numpy(dtype=float)
            item['actual'] = actual
            base = base_row.predicted_smp.to_numpy(dtype=float)
            prediction = (1 - weight) * item['persistence'] + weight * item[anchor]
            v17 = target_rows[target_rows.model.eq('V17_joint')].sort_values('hour')
            model_vectors = [('same_hour_previous_day_baseline', base),
                             ('rolling_calibrated_anchor_shadow', prediction)]
            if len(v17) == 24:
                model_vectors.append(('V17_joint_saved_replay', v17.predicted_smp.to_numpy(dtype=float)))
            for model, vector in model_vectors:
                stat = metrics(vector, item)
                summaries.append(dict(horizon_days=horizon, target_date=day, model=model,
                    **stat, anchor=anchor if model == 'rolling_calibrated_anchor_shadow' else '',
                    anchor_weight=weight if model == 'rolling_calibrated_anchor_shadow' else np.nan,
                    calibration_start=cal_start.strftime('%Y-%m-%d'), calibration_end=cal_end.strftime('%Y-%m-%d'),
                    calibration_days=len(calibration), input_timestamp_cutoff_verified=True,
                    input_first_seen_vintage_verified=False, model_available_at_origin=False,
                    evaluation_type='existing V20 42-day timestamp-causal replay; not operational validation'))
                for hour, stamp in enumerate(item['target']):
                    predictions.append(dict(horizon_days=horizon, target_date=day, model=model,
                        hour=hour, timestamp=stamp.isoformat(), actual_smp=float(actual[hour]),
                        predicted_smp=float(vector[hour]), error=float(vector[hour] - actual[hour]),
                        anchor=anchor if model == 'rolling_calibrated_anchor_shadow' else '',
                        anchor_weight=weight if model == 'rolling_calibrated_anchor_shadow' else np.nan,
                        input_first_seen_vintage_verified=False,
                        evaluation_type='timestamp-causal replay; historical first_seen/model availability unverified'))
    detail = pd.DataFrame(summaries)
    write_csv(OUT / 'v20_42day_rolling_shadow_daily.csv', detail)
    write_csv(OUT / 'v20_42day_rolling_shadow_hourly.csv', pd.DataFrame(predictions))
    summary = detail.groupby(['horizon_days', 'model'], as_index=False).agg(
        days=('target_date', 'nunique'), MAE=('MAE', 'mean'),
        # Match the original V20 42-day scorecard: calculate hourly RMSE per
        # target day, then average the 42 daily RMSE values.
        RMSE=('RMSE', 'mean'),
        G2V_regret=('G2V_regret', 'mean'))
    write_csv(OUT / 'v20_42day_rolling_shadow_summary.csv', summary)

    hourly = pd.DataFrame(predictions)
    for horizon in (1, 2, 3):
        # Midpoint available target date makes a compact, reproducible demo plot.
        days = sorted(hourly.loc[hourly.horizon_days.eq(horizon), 'target_date'].unique())
        day = days[len(days) // 2]
        plot = hourly[(hourly.horizon_days == horizon) & (hourly.target_date == day)]
        fig, ax = plt.subplots(figsize=(11, 4))
        actual = plot[plot.model.eq('same_hour_previous_day_baseline')]
        if actual.empty:
            actual = plot[plot.model.eq('rolling_calibrated_anchor_shadow')]
        ax.plot(actual.hour, actual.actual_smp, label='SMP provided in V20 replay', color='#1f2937', linewidth=2)
        for model, label in [('same_hour_previous_day_baseline', 'Issue-time lag baseline'),
                             ('V17_joint_saved_replay', 'V17 saved replay'),
                             ('rolling_calibrated_anchor_shadow', 'Rolling seasonal shadow')]:
            row = plot[plot.model.eq(model)]
            if not row.empty:
                ax.plot(row.hour, row.predicted_smp, label=label, linewidth=1.5)
        ax.set(title=f'D+{horizon} SMP replay · {day} · vintage unverified',
               xlabel='Hour (KST)', ylabel='SMP (KRW/kWh)')
        ax.grid(alpha=.25); ax.legend(ncols=2); fig.tight_layout()
        fig.savefig(OUT / f'v20_42day_d{horizon}_comparison.png', dpi=170)
        plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
