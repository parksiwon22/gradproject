"""Paired external-variable ablations; original V11 model and files are untouched."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor
from v11.store import HOME,connect,latest,now
from v11.features import FEATURES,build
from v11.evaluate import metrics
from v11.exogenous_collect import EXO,ELEMENTS,POWER

def read_date(file,column):
    f=pd.read_csv(file); f[column]=pd.to_datetime(f[column],utc=True).dt.tz_convert('Asia/Seoul'); return f

def attach(frame,weather,power,hourly=None,days_ahead=1):
    out=frame.copy(); days=out.timestamp.dt.normalize()
    weather_columns=[]; power_columns=[]; hourly_columns=[]
    for prefix,source,columns in [('weather',weather,list(ELEMENTS.values())),('power',power,['daily_max_demand_mw','supply_capacity_mw','reserve_mw','reserve_rate_percent'])]:
        lookup=source.set_index('date')
        for lag in (1,7):
            dates=days-pd.Timedelta(days=days_ahead+lag)
            for column in columns:
                name=f'{prefix}_{column}_origin_daylag{lag}'
                out[name]=lookup[column].reindex(pd.DatetimeIndex(dates)).to_numpy()
                (weather_columns if prefix=='weather' else power_columns).append(name)
    if hourly is not None:
        series=hourly.set_index('timestamp').demand_mwh
        for lag in (72,168):
            name=f'demand_hourly_target_lag{lag}'
            out[name]=series.reindex(pd.DatetimeIndex(out.timestamp-pd.Timedelta(hours=lag))).to_numpy(); hourly_columns.append(name)
        for window in (24,168):
            name=f'demand_hourly_mean{window}_delayed48h'
            out[name]=series.rolling(window,min_periods=window).mean().reindex(pd.DatetimeIndex(out.origin_timestamp-pd.Timedelta(hours=48))).to_numpy(); hourly_columns.append(name)
    return out,weather_columns,power_columns,hourly_columns

def model(family):
    if family=='hgb': return HistGradientBoostingRegressor(max_iter=250,max_leaf_nodes=31,learning_rate=.05,random_state=42,early_stopping=False)
    return ExtraTreesRegressor(n_estimators=150,min_samples_leaf=10,max_features=.8,n_jobs=4,random_state=42)

def predict(bundle,f): return f.daily.to_numpy()+bundle['model'].predict(f[bundle['features']])

def complete(frame,columns):
    f=frame.dropna(subset=columns+FEATURES+['actual_smp','previous_actual_smp']).copy()
    counts=f.groupby('target_date').size()
    return f[f.target_date.isin(counts[counts==24].index)].reset_index(drop=True)

def run():
    folder=EXO/'experiments'; folder.mkdir(exist_ok=True)
    db=connect(); prices=latest(db,'land'); db.close()
    base=build(prices); weather=read_date(EXO/'weather_daily_all.csv','date'); power=read_date(EXO/'power_daily_all.csv','date'); hourly=read_date(EXO/'demand_hourly_all.csv','timestamp')
    a,w,p,h=attach(base,weather,power,hourly)
    a.to_csv(EXO/'aligned_hourly_dataset.csv',index=False)
    definitions={
        '2025_hourly_demand':dict(end='2025-07-01',valstart='2025-07-01',teststart='2025-10-01',testend='2026-01-01',groups={'SMP':FEATURES,'SMP_hourly_demand':FEATURES+h,'SMP_weather':FEATURES+w,'SMP_hourly_demand_weather':FEATURES+h+w}),
        '2026_daily_exogenous':dict(end='2026-04-01',valstart='2026-04-01',teststart='2026-07-01',testend='2026-10-01',groups={'SMP':FEATURES,'SMP_daily_power_reserve':FEATURES+p,'SMP_weather':FEATURES+w,'SMP_daily_power_weather_reserve':FEATURES+p+w})}
    config=dict(seed=42,at=now(),definitions=definitions,weather_resolution='daily Seoul108, not hourly observations',hourly_demand='national including Jeju, MWh hourly',daily_power='national peak-state MW and percent, not hourly demand forecast',cutoff='D-1 23h',historical_available_at='unknown: lagged observation replay; not certified historical deployability',selection='validation MAE minimum within each feature group; ties peak MAE',input_hashes={str(f.name):hashlib.sha256(f.read_bytes()).hexdigest() for f in [EXO/'weather_daily_all.csv',EXO/'power_daily_all.csv',EXO/'demand_hourly_all.csv']})
    (folder/'config_before_tests.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    rows=[]; experiments=[]; selections={}; bundles={}
    for study,c in definitions.items():
        common=list(dict.fromkeys(sum(c['groups'].values(),[])))
        f=complete(a,common)
        train=f[f.target_date<c['end']].reset_index(drop=True)
        validation=f[(f.target_date>=c['valstart'])&(f.target_date<c['teststart'])].reset_index(drop=True)
        test=f[(f.target_date>=c['teststart'])&(f.target_date<c['testend'])].reset_index(drop=True)
        if not all(len(x)>0 for x in [train,validation,test]): raise ValueError('empty paired split')
        assert train.target_date.max()<validation.target_date.min() and validation.target_date.max()<test.target_date.min()
        val_rows=[]
        for group,columns in c['groups'].items():
            for family in ('hgb','extra_trees'):
                key=study+'__'+group+'__'+family
                estimator=model(family); estimator.fit(train[columns],train.actual_smp-train.daily)
                bundle=dict(model=estimator,family=family,features=columns,group=group,study=study,days_ahead=1)
                bundles[key]=bundle; joblib.dump(bundle,folder/(key+'.joblib'))
                result=dict(study=study,split='validation',group=group,family=family,**metrics(validation,predict(bundle,validation)))
                val_rows.append(result); rows.append(result)
                experiments.append(dict(key=key,status='success',features=columns,parameters=estimator.get_params(),train_start=train.target_date.min(),train_end=train.target_date.max(),validation_start=validation.target_date.min(),validation_end=validation.target_date.max(),test_start=test.target_date.min(),test_end=test.target_date.max(),rows={'train':len(train),'validation':len(validation),'test':len(test)}))
                (folder/'experiments.json').write_text(json.dumps(experiments,indent=2,default=str),encoding='utf-8')
                print(key,'trained',flush=True)
        v=pd.DataFrame(val_rows).sort_values(['MAE','peak_MAE','family'])
        chosen=v.groupby('group',sort=False).first().reset_index()
        selections[study]=chosen[['group','family','MAE','peak_MAE','EV_regret']].to_dict('records')
        (folder/'selection_before_test.json').write_text(json.dumps(dict(at=now(),selections=selections,test_used_for_selection=False),indent=2),encoding='utf-8')
        output=test[['timestamp','target_date','actual_smp']].copy()
        for key,b in bundles.items():
            if b['study']!=study: continue
            prediction=predict(b,test)
            rows.append(dict(study=study,split='test',group=b['group'],family=b['family'],**metrics(test,prediction)))
            output[b['group']+'__'+b['family']]=prediction
        output.to_csv(folder/(study+'_test_predictions.csv'),index=False)
        pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
    # Separate post-observation research bound. Never an operational forecast model.
    oracle=a.copy()
    for column in list(ELEMENTS.values()): oracle['RESEARCH_target_actual_'+column]=weather.set_index('date')[column].reindex(pd.DatetimeIndex(oracle.timestamp.dt.normalize())).to_numpy()
    oracle_columns=[c for c in oracle if c.startswith('RESEARCH_target_actual_')]
    f=complete(oracle,oracle_columns); c=definitions['2026_daily_exogenous']
    train=f[f.target_date<c['end']]; validation=f[(f.target_date>=c['valstart'])&(f.target_date<c['teststart'])].reset_index(drop=True); test=f[(f.target_date>=c['teststart'])&(f.target_date<c['testend'])].reset_index(drop=True)
    estimator=model('extra_trees'); columns=FEATURES+oracle_columns; estimator.fit(train[columns],train.actual_smp-train.daily)
    oracle_bundle=dict(model=estimator,features=columns,family='extra_trees',group='RESEARCH_target_observed_weather',study='research_only',not_deployable=True)
    joblib.dump(oracle_bundle,folder/'RESEARCH_target_observed_weather.joblib')
    pd.DataFrame([dict(split=s,group='RESEARCH_target_observed_weather_NOT_FORECAST',**metrics(f,f.daily+estimator.predict(f[columns]))) for s,f in [('validation',validation),('test',test)]]).to_csv(folder/'research_oracle_metrics.csv',index=False)
    # Refit selected causal-input structure on latest complete historical days.
    chosen=next(r for r in selections['2026_daily_exogenous'] if r['group']=='SMP_daily_power_weather_reserve')
    reference=bundles['2026_daily_exogenous__'+chosen['group']+'__'+chosen['family']]
    eligible=complete(a,reference['features']); eligible=eligible[eligible.target_date<pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y-%m-%d')]
    current=model(chosen['family']); current.fit(eligible[reference['features']],eligible.actual_smp-eligible.daily)
    production=dict(**{k:v for k,v in reference.items() if k!='model'},model=current,created_at=now(),trained_through=eligible.target_date.max(),availability_status='live input first_seen checked separately; historical availability not certified',input_hashes=config['input_hashes'])
    joblib.dump(production,EXO/'production_exogenous.joblib')
    # Two-day supplemental horizon uses the same selected family/structure.
    data2=build(prices,days_ahead=2); a2,w2,p2,_=attach(data2,weather,power,days_ahead=2); train2=complete(a2,FEATURES+w2+p2); train2=train2[train2.target_date<=eligible.target_date.max()]
    two=model(chosen['family']); two.fit(train2[reference['features']],train2.actual_smp-train2.daily)
    production2={**production,'model':two,'days_ahead':2}; joblib.dump(production2,EXO/'production_exogenous_2day.joblib')
    print('Paired ablations complete; original production model preserved',flush=True)

if __name__=='__main__': run()
