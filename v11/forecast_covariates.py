"""Separate archived-demand and observed-weather-proxy research experiments."""
import json
import joblib
import pandas as pd
from v11.store import connect,latest,now
from v11.features import FEATURES,build
from v11.evaluate import metrics
from v11.exogenous_collect import EXO
from v11.exogenous_model import read_date,attach,complete,model,predict

def main():
    folder=EXO/'forecast_covariate_study'; folder.mkdir(exist_ok=True)
    db=connect(); raw=latest(db,'land'); db.close()
    base=build(raw); weather=read_date(EXO/'weather_daily_all.csv','date'); power=read_date(EXO/'power_daily_all.csv','date')
    archive=read_date(EXO/'demand_forecast_hourly.csv','timestamp')
    a,w,p,_=attach(base,weather,power)
    a=a.merge(archive[['timestamp','day_ahead_demand_mw']],on='timestamp',how='left')
    group={'SMP':FEATURES,'SMP_archived_demand_forecast':FEATURES+['day_ahead_demand_mw'],'SMP_archived_demand_forecast_lagged_weather':FEATURES+['day_ahead_demand_mw']+w,'SMP_archived_demand_forecast_lagged_weather_power':FEATURES+['day_ahead_demand_mw']+w+p}
    common=list(dict.fromkeys(sum(group.values(),[]))); data=complete(a,common)
    train=data[data.target_date<'2026-04-01']; validation=data[(data.target_date>='2026-04-01')&(data.target_date<'2026-07-01')].reset_index(drop=True); test=data[(data.target_date>='2026-07-01')&(data.target_date<'2026-10-01')].reset_index(drop=True)
    config=dict(at=now(),groups=group,seed=42,kind='retrospective archived-forecast-value study',warning='Historical original issue/vintage unknown. Values really are KPX forecasts, but not certified D-1 23h versions. 2026Q3 already inspected in earlier studies.',periods={'train':[train.target_date.min(),train.target_date.max()],'validation':[validation.target_date.min(),validation.target_date.max()],'test':[test.target_date.min(),test.target_date.max()]},row_counts={'train':len(train),'validation':len(validation),'test':len(test)})
    (folder/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    rows=[]; bundles={}; parameters={}
    for name,columns in group.items():
        for family in ('hgb','extra_trees'):
            estimator=model(family); estimator.fit(train[columns],train.actual_smp-train.daily)
            b=dict(model=estimator,features=columns,family=family,group=name,historical_vintage='unverified'); key=name+'__'+family; bundles[key]=b; parameters[key]=estimator.get_params()
            joblib.dump(b,folder/(key+'.joblib'))
            rows.append(dict(split='validation',group=name,family=family,**metrics(validation,predict(b,validation))))
            print(key,'trained',flush=True)
    v=pd.DataFrame(rows).sort_values(['MAE','peak_MAE','family']).groupby('group',sort=False).first().reset_index()
    (folder/'selection_before_test.json').write_text(json.dumps(dict(at=now(),selected=v[['group','family']].to_dict('records'),parameters=parameters,test_used_for_selection=False),indent=2,default=str),encoding='utf-8')
    predictions=test[['timestamp','target_date','actual_smp']].copy()
    for key,b in bundles.items():
        pred=predict(b,test); predictions[key]=pred; rows.append(dict(split='test',group=b['group'],family=b['family'],**metrics(test,pred)))
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False); predictions.to_csv(folder/'test_predictions.csv',index=False)
    # Explicitly post-observed training proxy. Live input is a real published
    # forecast, but this distribution shift has no historical validation yet.
    proxy=['temperature_mean_c','temperature_max_c','temperature_min_c','wind_mean_ms','humidity_mean_percent']
    latest_day=pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y-%m-%d')
    for ahead in (1,2):
        f=build(raw,days_ahead=ahead)
        for name in proxy: f['PROXY_'+name]=weather.set_index('date')[name].reindex(pd.DatetimeIndex(f.timestamp.dt.normalize())).to_numpy()
        columns=FEATURES+['PROXY_'+n for n in proxy]
        f=complete(f,columns); fit=f[f.target_date<latest_day]
        estimator=model('extra_trees'); estimator.fit(fit[columns],fit.actual_smp-fit.daily)
        b=dict(model=estimator,features=columns,family='extra_trees',kind='OBSERVED_WEATHER_TRAINING_PROXY_NOT_VALIDATED_ON_FORECASTS',days_ahead=ahead,trained_through=fit.target_date.max())
        joblib.dump(b,EXO/f'weather_forecast_proxy_{ahead}day.joblib')
    print('Archived forecasts and separately marked proxy study complete',flush=True)

if __name__=='__main__': main()
