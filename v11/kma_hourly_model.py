"""Fixed chronological study: delayed observations are NOT weather forecasts."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from v11.store import connect,latest,now
from v11.features import FEATURES,build
from v11.evaluate import metrics
from v11.exogenous_model import model,predict,read_date
from v11.exogenous_collect import EXO
from v11.kma_apihub import FOLDER
from v11.kma_hourly import STATIONS,FIELDS

def attach(base,observations):
    f=base.copy(); columns=[]
    for stn in STATIONS:
        source=observations[observations.station_id==stn].set_index('timestamp')
        for field in FIELDS:
            name=f'ASOS_{stn}_{field}_target_lag48h'
            f[name]=source[field].reindex(pd.DatetimeIndex(f.timestamp-pd.Timedelta(hours=48))).to_numpy()
            columns.append(name)
    return f,columns

def main():
    folder=FOLDER/'hourly_study'; folder.mkdir(parents=True,exist_ok=True)
    config=dict(at=now(),seed=42,family='hgb',tuning='none; reuse previous HGB parameters',
        train_targets=[['2023-01-01','2026-03-31']],
        validation_targets=['2026-04-01','2026-06-30'],test_targets=['2026-10-01','2026-10-15'],
        warning='October holdout is short, not a full-year benchmark. 2026Q3 is excluded from tuning and scoring. Historical release times unknown. Lagged observations are not issued weather forecasts. Demand forecast archive vintage unknown.',
        missing_policy='No interpolation. Common whole days with all 7 stations temperature/RH/wind present; other source missing values retained for HGB native missing handling.',
        requested_groups_C_D='Not run: archived issued-weather-forecast API authorization denied. No observed target-weather surrogate.',
        selection='no promotion or winner selection from held-out results')
    # Persist design before reading any new held-out scores.
    (folder/'design_before_evaluation.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    db=connect(); prices=latest(db,'land'); db.close()
    source=FOLDER/'observations_period_hourly.csv'; observations=read_date(source,'timestamp')
    data,weather=attach(build(prices),observations)
    demand=read_date(EXO/'demand_forecast_hourly.csv','timestamp')
    data=data.merge(demand[['timestamp','day_ahead_demand_mw']],on='timestamp',how='left')
    core=[c for c in weather if any(s in c for s in ('temperature_c','humidity_percent','wind_ms'))]
    data=data.dropna(subset=FEATURES+core+['day_ahead_demand_mw'])
    counts=data.groupby('target_date').size(); data=data[data.target_date.isin(counts[counts==24].index)].reset_index(drop=True)
    def between(start,end): return (data.target_date>=start)&(data.target_date<=end)
    train=data[np.logical_or.reduce([between(*r) for r in config['train_targets']])].reset_index(drop=True)
    val=data[between(*config['validation_targets'])].reset_index(drop=True)
    test=data[between(*config['test_targets'])].reset_index(drop=True)
    if any(len(f)<24 for f in (train,val,test)): raise ValueError('Insufficient collected whole days; continue collection before training')
    assert train.target_date.max()<val.target_date.min()<test.target_date.min()
    groups={'A_SMP':FEATURES,'B_SMP_archived_demand':FEATURES+['day_ahead_demand_mw'],
            'OBS_SMP_delayed_ASOS_NOT_FORECAST':FEATURES+weather,
            'OBS_SMP_demand_delayed_ASOS_NOT_FORECAST':FEATURES+['day_ahead_demand_mw']+weather}
    config.update(groups=groups,parameters=model('hgb').get_params(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  rows={name:len(f) for name,f in [('train',train),('validation',val),('test',test)]})
    (folder/'config_before_scores.json').write_text(json.dumps(config,indent=2,default=str),encoding='utf-8')
    data.to_csv(folder/'aligned_dataset.csv',index=False)
    rows=[]; output=test[['timestamp','target_date','actual_smp']].copy()
    for name,columns in groups.items():
        estimator=model('hgb'); estimator.fit(train[columns],train.actual_smp-train.daily)
        bundle=dict(model=estimator,features=columns,family='hgb',group=name,kind='DELAYED_OBSERVATIONS_RESEARCH_NOT_PRODUCTION',historical_weather_forecast_used=False,created_at=now(),trained_through=train.target_date.max())
        joblib.dump(bundle,folder/(name+'.joblib'))
        for split,f in [('validation',val),('test',test)]:
            pred=predict(bundle,f); rows.append(dict(group=name,split=split,**metrics(f,pred)))
            if split=='test': output[name]=pred
        print(name,'trained and evaluated',flush=True)
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False); output.to_csv(folder/'test_predictions.csv',index=False)
    make_plot(output,groups)
    print(pd.DataFrame(rows).query("split == 'test'")[['group','MAE','peak_MAE','peak_F1','EV_regret']].to_string(index=False),flush=True)

def make_plot(output,groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=output.copy(); output.timestamp=pd.to_datetime(output.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
    peakday=output.loc[output.actual_smp.idxmax(),'target_date']
    fig,axes=plt.subplots(2,1,figsize=(13,7))
    labels=['Actual SMP','SMP only','+ archived demand forecast','+ delayed ASOS observations','+ demand + delayed observations']
    for ax,chart,title in [(axes[0],output,'Full fixed October evaluation'),(axes[1],output[output.target_date==peakday],'Highest actual-price day: '+str(peakday))]:
        for col,label in zip(['actual_smp']+list(groups),labels): ax.plot(chart.timestamp,chart[col],label=label,linewidth=1.2)
        ax.axhline(150,color='gray',linestyle='--',linewidth=.8,label='Peak threshold 150')
        ax.set_ylabel('KRW/kWh'); ax.set_title(title); ax.legend(fontsize=7,ncol=3); ax.tick_params(axis='x',labelrotation=25)
    fig.tight_layout(); fig.savefig(FOLDER/'hourly_study/comparison.png',dpi=160); plt.close(fig)

if __name__=='__main__': main()
