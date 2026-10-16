import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v16.common import ROOT,HOME,csv

def run():
    folder=ROOT/'docs/v16_figures';folder.mkdir(exist_ok=True)
    f=csv(ROOT/'docs/V16_SENSITIVITY.csv');f=f[f.cohort=='audit']
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    limit=max(1,abs(f.mean_gain).max())
    for ax,eff in zip(axes,[.8,.9,.95]):
        g=f[f.efficiency==eff].pivot(index='wear',columns='sell_ratio',values='mean_gain');im=ax.imshow(g,vmin=-limit,vmax=limit,cmap='RdBu')
        ax.set_xticks(range(3),g.columns);ax.set_yticks(range(3),g.index);ax.set_title(f'Efficiency each way {eff:.0%}');ax.set_xlabel('Assumed sale settlement ratio')
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{g.iloc[i,j]:.1f}',ha='center',va='center',color='black')
    axes[0].set_ylabel('Battery wear KRW / battery-throughput kWh');fig.suptitle('Seen2026H1: realized extra virtual KRW/day vs same-condition G2V');fig.tight_layout();fig.savefig(folder/'sensitivity.png',dpi=150);plt.close(fig)
    f=csv(ROOT/'docs/V16_INFORMATION_ORACLES.csv');f=f[f.cohort=='audit'].set_index('information').loc[['point_baseline','perfect_daily_mean','perfect_order_only','perfect_shape','perfect_prices']]
    ax=f[['G2V_regret','V2G_regret']].plot.bar(figsize=(11,4),rot=15);ax.set(ylabel='Virtual KRW/day regret',xlabel='',title='Information upper bounds: actual answers injected, NOT forecast models');ax.figure.tight_layout();ax.figure.savefig(folder/'information_value.png',dpi=150);plt.close(ax.figure)
    f=csv(ROOT/'docs/V16_DEFINITION_SCORES.csv');f=f[f.model.isin(['baseline','two_stage_shrink_0.5'])];g=f.groupby(['definition','model']).F1.mean().unstack()
    ax=g.plot.bar(figsize=(10,4),rot=15);ax.set(ylabel='Macro-fold F1',xlabel='',title='Different event definitions, reused frozen validation predictions');ax.figure.tight_layout();ax.figure.savefig(folder/'event_definitions.png',dpi=150);plt.close(ax.figure)
    files=sorted((HOME/'maintenance_study').glob('*_predictions.csv'))
    for file in files:
        f=csv(file);f.timestamp=pd.to_datetime(f.timestamp,utc=True).dt.tz_convert('Asia/Seoul');f['gain']=abs(f.price_only_common-f.actual_smp)-abs(f.price_plus_maintenance-f.actual_smp);gain=f.groupby('target_date').gain.mean();days=[gain.idxmax(),gain.idxmin()]
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,day,label in zip(axes,days,['Largest daily MAE reduction','Largest daily MAE increase']):
            g=f[f.target_date==day]
            for c in ['actual_smp','price_only_common','price_plus_maintenance']:ax.plot(g.timestamp.dt.hour,g[c],label=c)
            ax.set(title=day+' / '+label,xlabel='KST hour',ylabel='SMP KRW/kWh');ax.legend(fontsize=7)
        fig.suptitle('Retrospective maintenance study: original attachment vintage NOT verified',fontsize=10);fig.tight_layout();fig.savefig(folder/(file.stem+'.png'),dpi=150);plt.close(fig)
    print('V16 scientific figures saved')

if __name__=='__main__':run()
