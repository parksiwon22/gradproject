"""Render the committed, research-only summary tables and plots."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v19.common import ROOT, write_csv

OUT = ROOT / 'docs' / 'smp_improvement'


def main():
    frame = pd.read_csv(OUT / 'walk_forward_metrics.csv')
    selected = frame[frame.model.isin(['persistence', 'calibration_selected_anchor_blend'])]
    rows = []
    for (horizon, model), group in selected.groupby(['horizon_days', 'model']):
        weights = group.days.to_numpy(float)
        rows.append(dict(horizon_days=int(horizon), model=model, days=int(weights.sum()),
            MAE=float(np.average(group.MAE, weights=weights)),
            RMSE=float(np.sqrt(np.average(np.square(group.RMSE), weights=weights))),
            G2V_regret=float(np.average(group.G2V_regret, weights=weights)),
            peak_MAE=float(np.average(group.peak_MAE, weights=weights)),
            vintage_verified=False, evaluation='three known historical rolling validation folds'))
    summary = pd.DataFrame(rows)
    write_csv(OUT / 'walk_forward_summary.csv', summary)

    metrics = ['MAE', 'RMSE', 'G2V_regret']
    labels = ['MAE (KRW/kWh)', 'RMSE (KRW/kWh)', 'G2V regret (KRW/day)']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric, label in zip(axes, metrics, labels):
        piv = summary.pivot(index='horizon_days', columns='model', values=metric)
        piv = piv.reindex(index=[1, 2, 3], columns=['persistence', 'calibration_selected_anchor_blend'])
        piv.plot(kind='bar', ax=ax, color=['#9ca3af', '#2563eb'], rot=0)
        ax.set_title(label); ax.set_xlabel('Forecast horizon'); ax.set_xticklabels(['D+1', 'D+2', 'D+3'])
        ax.legend(['Horizon lag', 'Calibrated anchor'], fontsize=8); ax.grid(axis='y', alpha=.25)
    fig.suptitle('Known historical validation folds · vintage unverified', y=1.02)
    fig.tight_layout(); fig.savefig(OUT / 'walk_forward_metric_comparison.png', dpi=170, bbox_inches='tight'); plt.close(fig)

    hourly = pd.read_csv(OUT / 'hourly_errors.csv')
    hourly = hourly[hourly.model.isin(['persistence', 'calibration_selected_anchor_blend'])]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for horizon, ax in zip([1, 2, 3], axes):
        subset = hourly[hourly.horizon_days.eq(horizon)].groupby(['model', 'hour'], as_index=False).MAE.mean()
        for model, label, color in [('persistence', 'Horizon lag', '#9ca3af'),
                                    ('calibration_selected_anchor_blend', 'Calibrated anchor', '#2563eb')]:
            g = subset[subset.model.eq(model)].sort_values('hour')
            ax.plot(g.hour, g.MAE, marker='.', linewidth=1.5, label=label, color=color)
        ax.set_title(f'D+{horizon}'); ax.set_xlabel('Hour (KST)'); ax.grid(alpha=.25)
    axes[0].set_ylabel('Absolute error (KRW/kWh)'); axes[-1].legend(fontsize=8)
    fig.suptitle('Hourly validation error · timestamp-only replay', y=1.02)
    fig.tight_layout(); fig.savefig(OUT / 'hourly_error_comparison.png', dpi=170, bbox_inches='tight'); plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
