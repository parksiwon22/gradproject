"""Official KPX price-only transfer evaluation, selected before 2024 labels."""
import json
import re
import hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from scripts.research_local import ROOT, OUT, SEED, metrics

def parse_kpx(text):
    rows=[]
    pattern=r'((?:\s*c\d+\s*=\s*textFormmat\("[^"]*",count\);\s*)+)gridData.push\(\{"Date":"(\d{4}/\d{2}/\d{2})"'
    for block,day in re.findall(pattern,text):
        values={int(h):float(v) for h,v in re.findall(r'c(\d+)\s*=\s*textFormmat\("([\d.\-]+)",count\)',block)}
        if not all(h in values for h in range(1,25)): raise ValueError('Incomplete official day')
        for h in range(1,25): rows.append({'timestamp':pd.Timestamp(day)+pd.Timedelta(hours=h-1),'smp':values[h]})
    if not rows: raise ValueError('Empty KPX response')
    frame=pd.DataFrame(rows).sort_values('timestamp').reset_index(drop=True)
    if frame.empty or frame.timestamp.duplicated().any(): raise ValueError('Empty or duplicate KPX response')
    return frame

def price_rows(raw):
    series=raw.set_index('timestamp').smp.sort_index()
    rows=[]
    for date in pd.date_range(series.index.min().normalize()+pd.Timedelta(days=8),series.index.max().normalize()):
        origin=date-pd.Timedelta(hours=1)
        if origin not in series.index: continue
        for h in range(24):
            t=date+pd.Timedelta(hours=h)
            if t not in series.index or t-pd.Timedelta(days=7) not in series.index: continue
            history=series.loc[origin-pd.Timedelta(hours=167):origin]
            daily=series[t-pd.Timedelta(hours=24)]; weekly=series[t-pd.Timedelta(hours=168)]
            row=dict(timestamp=t,target_date=str(date.date()),origin_timestamp=origin,actual_smp=series[t],previous_actual_smp=series[t-pd.Timedelta(hours=1)],daily=daily,weekly=weekly,last=series[origin],mean24=history.iloc[-24:].mean(),std24=history.iloc[-24:].std(),mean168=history.mean(),std168=history.std(),hour=h,dow=t.dayofweek,month=t.month,weekend=int(t.dayofweek>=5))
            for lag in (1,2,24,48,168): row['lag'+str(lag)]=series[origin-pd.Timedelta(hours=lag-1)]
            rows.append(row)
    return pd.DataFrame(rows)

def run():
    target=OUT/'price_only'; target.mkdir(exist_ok=True)
    raw=pd.read_csv(ROOT/'data/aligned_hourly_all_features_2023.csv',usecols=['timestamp','smp'],parse_dates=['timestamp'])
    data=price_rows(raw)
    excluded=['timestamp','target_date','origin_timestamp','actual_smp','previous_actual_smp']
    features=[c for c in data if c not in excluded]
    train=data[data.target_date<'2023-07-01'].reset_index(drop=True)
    val=data[(data.target_date>='2023-08-01')&(data.target_date<'2023-10-01')].reset_index(drop=True)
    # Fixed configurations; no 2024-dependent hyperparameters or refitting.
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor
    models={'hgb':HistGradientBoostingRegressor(max_iter=400,learning_rate=.05,random_state=SEED,early_stopping=False), 'peak_hgb':HistGradientBoostingRegressor(max_iter=400,learning_rate=.05,random_state=SEED,early_stopping=False), 'extra_trees':ExtraTreesRegressor(n_estimators=200,min_samples_leaf=10,max_features=.8,n_jobs=4,random_state=SEED), 'lightgbm':LGBMRegressor(n_estimators=400,learning_rate=.05,n_jobs=4,random_state=SEED,verbosity=-1), 'catboost':CatBoostRegressor(iterations=400,random_seed=SEED,verbose=False,thread_count=4,allow_writing_files=False), 'xgboost':XGBRegressor(n_estimators=400,learning_rate=.05,n_jobs=4,random_state=SEED)}
    rows=[]
    for name,model in models.items():
        kwargs={'sample_weight':np.where(train.actual_smp>=150,4.,1.)} if name=='peak_hgb' else {}
        model.fit(train[features],train.actual_smp-train.daily,**kwargs)
        rows.append(dict(split='validation',model=name,**metrics(val,val.daily+model.predict(val[features]),150)))
        joblib.dump(model,target/(name+'.joblib'))
    rows.append(dict(split='validation',model='fixed_profile_069',**metrics(val,.69*val.daily+.31*val.weekly,150)))
    validation=pd.DataFrame(rows)
    bound=float(validation.loc[validation.model=='fixed_profile_069','MAE'].iloc[0])*1.05
    winner=validation[validation.MAE<=bound].sort_values(['peak_MAE','MAE','model']).iloc[0].model
    selection={'winner':winner,'features':features,'seed':SEED,'train_period':[train.target_date.min(),train.target_date.max()],'validation_period':[val.target_date.min(),val.target_date.max()],'rule':'MAE <= 1.05 * fixed profile; minimize peak MAE then MAE','parameters':{n:m.get_params() for n,m in models.items()},'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'2024_labels_used_for_selection':False,'market_2023':'unverified; cross-market matching not certified','market_2024':'KPX mainland hourly SMP','original_V5_2024':'unavailable: original historical covariates missing; price-only models are distinct'}
    (target/'selection_before_2024.json').write_text(json.dumps(selection,indent=2,default=str),encoding='utf-8')
    # Access the official 2024 labels only after writing the frozen selection.
    payload=(OUT/'kpx_2024_raw.js').read_bytes()
    official=parse_kpx(payload.decode('utf-8'))
    expected=pd.date_range('2023-12-01','2024-12-31 23:00',freq='h')
    assert official.timestamp.tolist()==expected.tolist()
    official.to_csv(target/'kpx_official_hourly.csv',index=False)
    future=price_rows(official)
    future=future[future.target_date>='2024-01-01'].reset_index(drop=True)
    assert len(future)==8784
    forecast={'fixed_profile_069':.69*future.daily+.31*future.weekly}
    for name,model in models.items(): forecast[name]=future.daily+model.predict(future[features])
    output=future[excluded].copy()
    for n,p in forecast.items(): output[n]=np.asarray(p)
    output.to_csv(target/'2024_predictions.csv',index=False)
    for split,mask in [('2024_full',np.ones(len(future),dtype=bool)),('2024_Q4',future.target_date>='2024-10-01')]:
        f=future[mask].reset_index(drop=True)
        top5=float(f.actual_smp.quantile(.95)) # Descriptive only, never alert calibration.
        for name,p in forecast.items():
            p=np.asarray(p)[mask]
            result=metrics(f,p,150)
            result['descriptive_top5_MAE']=float(np.mean(abs(p[f.actual_smp>=top5]-f.actual_smp[f.actual_smp>=top5])))
            rows.append(dict(split=split,model=name,**result))
    pd.DataFrame(rows).to_csv(target/'metrics.csv',index=False)
    selection['official_payload_sha256']=hashlib.sha256(payload).hexdigest()
    selection['coverage']='2024 full 8784 hours; Q4 2208 hours; 2023-12 history downloaded'
    selection['Q4_status']='previous cloud results seen; held out of new selection, not fully unseen'
    selection['source']='https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpShdChart.do?menuId=040202'
    selection['mapping']='c1=01h ending interval -> start00:00; c24=24h -> start23:00; KST, historical final snapshot, publication unknown'
    (target/'evaluation_manifest.json').write_text(json.dumps(selection,indent=2,default=str),encoding='utf-8')
    import matplotlib.pyplot as plt
    for label,mask in [('2024_Q4',future.target_date>='2024-10-01'),('2024_first_week',future.target_date<'2024-01-08')]:
        fig,ax=plt.subplots(figsize=(14,4)); ax.plot(future.timestamp[mask],future.actual_smp[mask],label='Actual',lw=.8); ax.plot(future.timestamp[mask],np.asarray(forecast[winner])[mask],label=winner,lw=.8,alpha=.7); ax.set_ylabel('KRW/kWh'); ax.legend(); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(target/(label+'.png')); plt.close(fig)
    print('price-only validation-selected',winner,flush=True)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__': run()
