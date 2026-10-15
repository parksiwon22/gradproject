"""Reuse frozen V12 dispatches; never fit/select on their audit outcomes."""
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v13.package import HOME
from v12.research import HOME as V12
from v12.ev import EV,baseline,optimize,score

def run():
    folder=HOME/'economics'; folder.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(V12/'analysis/ev_daily_backtest.csv')
    pairs=data.pivot(index=['day','model'],columns='mode',values=['estimated_net_cost','realized_net_cost','regret'])
    result=pd.DataFrame(index=pairs.index)
    result['predicted_v2g_saving']=pairs['estimated_net_cost']['G2V']-pairs['estimated_net_cost']['V2G']
    result['actual_v2g_saving']=pairs['realized_net_cost']['G2V']-pairs['realized_net_cost']['V2G']
    result['reversal']=(result.predicted_v2g_saving>1e-6)&(result.actual_v2g_saving < -1e-6)
    result.reset_index().to_csv(folder/'paired_dispatch.csv',index=False)
    result[result.reversal].sort_values('actual_v2g_saving').to_csv(folder/'v2g_reversals.csv')
    predictions=pd.read_csv(V12/'study/retrospective_audit_predictions.csv')
    # Reuse the same full-day connection and EV assumptions as V12 analysis.
    predictions.timestamp=pd.to_datetime(predictions.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
    actual=predictions.drop_duplicates('timestamp').sort_values('timestamp')
    ev=EV(degradation=5); baselines=[]
    for day,g in actual.groupby(actual.timestamp.dt.strftime('%Y-%m-%d')):
        for name,uniform in [('immediate',False),('uniform',True)]:
            baselines.append(dict(day=day,mode=name,realized_net_cost=baseline(g.actual_smp,g.timestamp,ev,uniform=uniform)['net_cost']))
    b=pd.DataFrame(baselines); b.to_csv(folder/'same_condition_baselines.csv',index=False)
    summary=data.groupby(['model','mode'])[['estimated_net_cost','realized_net_cost','regret']].mean().reset_index()
    summary.to_csv(folder/'model_economics.csv',index=False)
    counts=result.groupby('model').agg(paired_days=('reversal','size'),reversal_days=('reversal','sum'),mean_actual_v2g_saving=('actual_v2g_saving','mean'))
    counts.to_csv(folder/'reversal_summary.csv')
    worst=result[result.reversal].sort_values('actual_v2g_saving').reset_index().iloc[0]
    example=actual[actual.timestamp.dt.strftime('%Y-%m-%d')==worst.day].copy()
    plans={name:optimize(example[worst.model],example.timestamp,ev,v2g=flag) for name,flag in [('G2V',False),('V2G',True)]}
    example.to_csv(folder/'reversal_example_prices.csv',index=False)
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
    axes[0].plot(example.timestamp.dt.hour,example.actual_smp,label='Actual'); axes[0].plot(example.timestamp.dt.hour,example[worst.model],label='Frozen prediction'); axes[0].legend(); axes[0].set_ylabel('SMP KRW/kWh')
    for name,plan in plans.items():
        schedule=plan['schedule']; schedule.to_csv(folder/('reversal_example_'+name+'.csv'),index=False)
        axes[1].plot(example.timestamp.dt.hour,schedule.charge_kw-schedule.discharge_kw,label=name)
        axes[2].plot(example.timestamp.dt.hour,schedule.soc_end,label=name)
    axes[1].set_ylabel('Import minus export kW'); axes[2].set_ylabel('End SOC %'); axes[2].set_xlabel('KST hour'); axes[1].legend(); axes[2].legend()
    fig.suptitle(str(worst.day)+' / '+worst.model+' / retrospective simulation'); fig.tight_layout(); fig.savefig(folder/'reversal_example.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5)); ax.scatter(result.predicted_v2g_saving,result.actual_v2g_saving,s=12,alpha=.4)
    ax.axhline(0,color='black'); ax.axvline(0,color='black'); ax.set(xlabel='Predicted V2G saving vs G2V (virtual KRW)',ylabel='Realized saving (virtual KRW)',title='Frozen 2026 H1 audit: model-days are correlated')
    fig.tight_layout(); fig.savefig(folder/'predicted_vs_realized.png',dpi=160); plt.close(fig)
    report=dict(dispatch_rows=len(data),paired_model_days=len(result),independent_days=int(result.reset_index().day.nunique()),reversals=int(result.reversal.sum()),baseline_mean=b.groupby('mode').realized_net_cost.mean().to_dict(),selection_use=False,interpretation='Previously viewed retrospective audit, not new forward evidence. SMP-linked simulation, not retail tariff or real V2G revenue.')
    (folder/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__': run()
