"""Locked chronological multi-year ablation, then separate production refit."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import hashlib
import importlib.metadata
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from v11.store import HOME,ROOT,connect,latest,now
from v11.features import FEATURES,build,predict
from v11.evaluate import metrics

def factories():
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor
    return {'hgb':lambda:HistGradientBoostingRegressor(max_iter=250,max_leaf_nodes=31,learning_rate=.05,random_state=42,early_stopping=False), 'extra_trees':lambda:ExtraTreesRegressor(n_estimators=150,min_samples_leaf=10,max_features=.8,n_jobs=4,random_state=42), 'catboost':lambda:CatBoostRegressor(iterations=300,depth=6,learning_rate=.05,random_seed=42,thread_count=4,verbose=False,allow_writing_files=False), 'lightgbm':lambda:LGBMRegressor(n_estimators=300,learning_rate=.05,num_leaves=31,random_state=42,n_jobs=4,verbosity=-1), 'xgboost':lambda:XGBRegressor(n_estimators=300,max_depth=6,learning_rate=.05,random_state=42,n_jobs=4), 'peak_hgb':lambda:HistGradientBoostingRegressor(max_iter=250,max_leaf_nodes=31,learning_rate=.05,random_state=42,early_stopping=False)}

def main():
    folder=HOME/'training'; folder.mkdir(exist_ok=True)
    db=connect(); raw=latest(db,'land'); db.close()
    data=build(raw)
    data.to_csv(folder/'dataset.csv',index=False)
    fingerprint=hashlib.sha256(raw[['timestamp','smp']].to_csv(index=False).encode()).hexdigest()
    validation=data[(data.target_date>='2026-04-01')&(data.target_date<'2026-07-01')].reset_index(drop=True)
    evaluation=data[(data.target_date>='2026-07-01')&(data.target_date<'2026-10-01')].reset_index(drop=True)
    assert len(validation)==91*24 and len(evaluation)==92*24
    stages={'2023_only':'2024-01-01','2023_2024':'2025-01-01','2023_2025':'2026-01-01','through_2026Q1':'2026-04-01'}
    fs=factories(); rows=[]; experiments=[]; bundles={}; valpred={}
    rule='validation MAE minimum; ties peak MAE; report peak/EV tradeoffs without test-dependent tuning'
    config=dict(created_at=now(),data_hash=fingerprint,features=FEATURES,seed=42,stages=stages,validation=['2026-04-01','2026-06-30'],test=['2026-07-01','2026-09-30'],rule=rule,peak_threshold=150,ramp_threshold=15,early_stopping='disabled for HGB; fixed iterations for other families',market='land',historical_availability='event-time replay only: original publication vintages unknown',packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','catboost','lightgbm','xgboost']})
    (folder/'config_before_evaluation.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    for stage,end in stages.items():
        train=data[data.target_date<end].reset_index(drop=True)
        assert train.target_date.max()<validation.target_date.min()
        for name,create in fs.items():
            key=stage+'__'+name; start=time.monotonic()
            try:
                model=create(); kwargs={'sample_weight':np.where(train.actual_smp>=150,4.,1.)} if name=='peak_hgb' else {}
                model.fit(train[FEATURES],train.actual_smp-train.daily,**kwargs)
                bundle=dict(family=name,model=model,stage=stage,data_hash=fingerprint,trained_through=train.target_date.max(),days_ahead=1)
                p=predict(bundle,validation); result=metrics(validation,p)
                rows.append(dict(split='validation',stage=stage,model=name,**result)); bundles[key]=bundle; valpred[key]=p
                joblib.dump(bundle,folder/(key+'.joblib'))
                experiments.append(dict(stage=stage,model=name,status='success',parameters=model.get_params(),train_start=train.target_date.min(),train_end=train.target_date.max(),train_rows=len(train),peak_weight=4 if name=='peak_hgb' else 1,seconds=time.monotonic()-start))
            except Exception as exc: experiments.append(dict(stage=stage,model=name,status='failed',reason=repr(exc)))
            (folder/'experiments.json').write_text(json.dumps(experiments,indent=2,default=str),encoding='utf-8')
            pd.DataFrame(rows).to_csv(folder/'validation_metrics.csv',index=False)
            print(key,experiments[-1]['status'],flush=True)
    profile=dict(family='profile',model=None,stage='fixed',days_ahead=1,data_hash=fingerprint)
    rows.append(dict(split='validation',stage='fixed',model='profile',**metrics(validation,predict(profile,validation))))
    bundles['fixed__profile']=profile
    rank=pd.DataFrame(rows).sort_values(['MAE','peak_MAE','model'])
    best=rank.iloc[0]; key=best.stage+'__'+best.model
    selected=bundles[key]
    selection=dict(stage=best.stage,family=best.model,rule=rule,locked_at=now(),data_hash=fingerprint,test_used_for_selection=False)
    (folder/'selection_before_test.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    # No test metric computed until the selection above has been written.
    output=evaluation[['timestamp','target_date','actual_smp']].copy()
    for name,bundle in bundles.items():
        p=predict(bundle,evaluation); output[name]=p
        rows.append(dict(split='test_2026Q3',stage=bundle['stage'],model=bundle['family'],**metrics(evaluation,p)))
    output.to_csv(folder/'test_predictions.csv',index=False)
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
    # Report a contemporary 2023 holdout separately; never call it V5 reproduction.
    historical=data[(data.target_date>='2023-10-01')&(data.target_date<'2024-01-01')].reset_index(drop=True)
    histtrain=data[data.target_date<'2023-09-01']
    model=fs['extra_trees'](); model.fit(histtrain[FEATURES],histtrain.actual_smp-histtrain.daily)
    historical_result=metrics(historical,historical.daily+model.predict(historical[FEATURES]))
    (folder/'2023_context.json').write_text(json.dumps(dict(note='exploratory price-only ExtraTrees; different features from V5',metrics=historical_result),indent=2),encoding='utf-8')
    # Deployment refit only after the historical selection and evaluation are fixed.
    issued=now(); eligible=data[data.timestamp<=pd.Timestamp(issued).tz_convert('Asia/Seoul')].copy()
    # Keep entire days together; exclude partial current/future target days from labels.
    eligible=eligible[eligible.target_date<pd.Timestamp(issued).tz_convert('Asia/Seoul').strftime('%Y-%m-%d')]
    if selected['family']!='profile':
        model=fs[selected['family']](); kwargs={'sample_weight':np.where(eligible.actual_smp>=150,4.,1.)} if selected['family']=='peak_hgb' else {}
        model.fit(eligible[FEATURES],eligible.actual_smp-eligible.daily,**kwargs)
    else: model=None
    validation_prediction=predict(selected,validation)
    # Symmetric validation residual band; coverage is descriptive, not guaranteed under drift.
    radius=float(np.quantile(abs(validation.actual_smp-validation_prediction),.9))
    production=dict(family=selected['family'],model=model,stage='production_refit',data_hash=fingerprint,trained_through=eligible.target_date.max(),days_ahead=1,created_at=issued,interval_radius90=radius,selection=selection)
    joblib.dump(production,HOME/'production.joblib')
    # Separate two-day horizon model for unobserved next-next-day targets.
    two=build(raw,days_ahead=2); two=two[two.target_date<=eligible.target_date.max()]
    model2=fs['extra_trees'](); model2.fit(two[FEATURES],two.actual_smp-two.daily)
    val2=two[(two.target_date>='2026-04-01')&(two.target_date<'2026-07-01')]
    # Fixed family, no family search at this horizon; in-sample band is prohibited.
    fit2=two[two.target_date<'2026-04-01']; calibration=fs['extra_trees'](); calibration.fit(fit2[FEATURES],fit2.actual_smp-fit2.daily)
    radius2=float(np.quantile(abs(val2.actual_smp-(val2.daily+calibration.predict(val2[FEATURES]))),.9))
    test2=build(raw,days_ahead=2); test2=test2[(test2.target_date>='2026-07-01')&(test2.target_date<'2026-10-01')].reset_index(drop=True)
    (folder/'two_day_metrics.json').write_text(json.dumps(dict(note='fixed ExtraTrees horizon2, train through2026Q1, 2026Q2 interval calibration',validation=metrics(val2.reset_index(drop=True),val2.daily+calibration.predict(val2[FEATURES])),test=metrics(test2,test2.daily+calibration.predict(test2[FEATURES]))),indent=2),encoding='utf-8')
    production2=dict(family='extra_trees',model=model2,stage='production_2day',data_hash=fingerprint,trained_through=eligible.target_date.max(),days_ahead=2,created_at=issued,interval_radius90=radius2,selection='fixed additional 2-day horizon; separate from original 23h policy')
    joblib.dump(production2,HOME/'production_2day.joblib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for label,mask in [('2026Q3',np.ones(len(evaluation),dtype=bool)),('2026Q3_first_week',np.arange(len(evaluation))<168)]:
        fig,ax=plt.subplots(figsize=(14,4)); ax.plot(evaluation.timestamp[mask],evaluation.actual_smp[mask],label='Official actual',lw=.8); ax.plot(evaluation.timestamp[mask],output[key][mask],label=key,lw=.8,alpha=.7); ax.set_ylabel('KRW/kWh'); ax.legend(); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(folder/(label+'.png')); plt.close(fig)
    print('Selected',selection,flush=True)

if __name__=='__main__': main()
