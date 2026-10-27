"""Render diagnostics from the saved ML study without refitting models."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v19.common import ROOT, write_csv


OUT = ROOT / 'docs' / 'smp_improvement_ml'


def main():
    hourly = pd.read_csv(OUT / 'hourly_predictions.csv')
    hourly['abs_error'] = hourly.error.abs()
    write_csv(OUT / 'hourly_error_by_model_hour.csv', hourly.groupby(['horizon_days','model','hour'],as_index=False).agg(
        samples=('error','size'), MAE=('abs_error','mean'), bias=('error','mean')))
    peak = hourly[hourly.actual_smp >= 150].groupby(['horizon_days','model'],as_index=False).agg(
        peak_hours=('error','size'), peak_MAE=('abs_error','mean'), peak_RMSE=('error',lambda x:float(np.sqrt(np.mean(x*x)))))
    write_csv(OUT / 'peak_error_summary.csv',peak)
    # Keep same representative days as the original study graph.
    for h in (1,2,3):
        part=hourly[(hourly.horizon_days==h)&(hourly.fold=='fall2025')]
        days=sorted(part.target_date.unique()); day=days[len(days)//2]
        p=part[part.target_date.eq(day)]
        actual=p[p.model.eq('persistence')]
        fig,ax=plt.subplots(figsize=(11,4)); ax.plot(actual.hour,actual.actual_smp,label='Actual SMP',lw=2,color='#111827')
        for model in ['persistence','hgb','ml_anchor_blend','rolling_anchor_shadow']:
            q=p[p.model.eq(model)]
            if not q.empty: ax.plot(q.hour,q.predicted_smp,label=model,lw=1.5)
        ax.set(title=f'D+{h} representative forecast · {day} · retrospective replay',xlabel='Hour (KST)',ylabel='SMP (KRW/kWh)')
        ax.grid(alpha=.25); ax.legend(ncols=2); fig.tight_layout(); fig.savefig(OUT/f'd{h}_representative_forecast.png',dpi=170); plt.close(fig)
    print(f'Wrote hourly/peak diagnostics from {len(hourly):,} saved rows')


if __name__ == '__main__':
    main()
