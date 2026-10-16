"""Training-fixed event thresholds and decision-related definitions, no new fit."""
import json
import joblib,numpy as np,pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from v11.store import ROOT,connect,latest
from v11.features import build as build11,predict as predict11
from v15.research import FOLDS,HOME as V15,predict as predict15
from v14.research import scored,ev_metrics
from v16.common import HOME,csv,write

def top6(values):
    a=np.asarray(values); out=np.zeros(len(a),dtype=bool)
    if np.ptp(a)>1e-9: out[np.argsort(-a,kind='stable')[:6]]=True
    return out

def thresholds(train):
    d=(train.actual_smp-train.previous_actual_smp).to_numpy()
    return dict(up=float(np.quantile(d[d>0],.9)),down=float(np.quantile(-d[d<0],.9)))

def masks(f,p,t):
    a=f.actual_smp.to_numpy(); p=np.asarray(p); actual_prev=f.previous_actual_smp.to_numpy()
    # Boundary hour has no prior-day predicted curve: exclude it for both curves.
    dp=np.full(len(f),np.nan); rel_a=np.zeros(len(f),bool); rel_p=np.zeros(len(f),bool)
    for ids in f.groupby('target_date',sort=False).indices.values():
        dp[ids[1:]]=np.diff(p[ids]); rel_a[ids]=top6(a[ids]); rel_p[ids]=top6(p[ids])
    da=a-actual_prev; valid=np.isfinite(dp)
    return dict(absolute150=(a>=150,p>=150,np.ones(len(f),bool)),known_uplift20=(a-f.daily>=20,p-f.daily>=20,np.ones(len(f),bool)),up_ramp=(da>=t['up'],dp>=t['up'],valid),down_ramp=(-da>=t['down'],-dp>=t['down'],valid),daily_top6=(rel_a,rel_p,np.ones(len(f),bool)))

def evaluate(f,p,t):
    rows=[]
    for name,(a,b,valid) in masks(f,p,t).items():
        a=np.asarray(a,dtype=bool)&valid; b=np.asarray(b,dtype=bool)&valid
        precision,recall,f1,_=precision_recall_fscore_support(a[valid],b[valid],average='binary',zero_division=0)
        rows.append(dict(definition=name,eligible_hours=int(valid.sum()),actual_event_hours=int(a.sum()),MAE=float(abs(np.asarray(p)-f.actual_smp).mean()),event_MAE=float(abs(np.asarray(p)[a]-f.actual_smp.to_numpy()[a]).mean()) if a.any() else None,event_bias=float((np.asarray(p)[a]-f.actual_smp.to_numpy()[a]).mean()) if a.any() else None,precision=precision,recall=recall,F1=f1,false_alarm_hours=int((~a&b&valid).sum())))
    return rows

def run():
    folder=HOME/'definitions';folder.mkdir(parents=True,exist_ok=True)
    matrix=csv(V15/'model_study/frozen_matrix.csv');matrix.timestamp=pd.to_datetime(matrix.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
    allrows=[]; dayrows=[]; thresholds_by_fold={}
    for fold in FOLDS:
        train=matrix[matrix.target_date<=fold['train_end']]; t=thresholds(train); thresholds_by_fold[fold['name']]=t
        base=csv(V15/'model_study'/(fold['name']+'_predictions.csv')); extra=csv(V15/'shrinkage_followup'/(fold['name']+'_predictions.csv'))
        f=matrix[(matrix.target_date>=fold['val_start'])&(matrix.target_date<=fold['val_end'])].reset_index(drop=True)
        assert np.array_equal(base.actual_smp,f.actual_smp),'Frozen labels differ'
        for name in ['baseline','event_features_lgb','relative_event_weight','relative_two_stage','event_analog','two_stage_shrink_0.25','two_stage_shrink_0.5']:
            p=(base if name in base else extra)[name].to_numpy()
            allrows.extend(dict(fold=fold['name'],model=name,**r) for r in evaluate(f,p,t))
            path=(V15/('shrinkage_followup' if name.startswith('two_stage_shrink') else 'model_study')/(fold['name']+'_'+name+'_ev.csv'))
            # Followup EV daily rows were not saved previously; no fabricate.
            daily=csv(path).set_index('day') if path.exists() else None
            for definition,(a,b,valid) in masks(f,p,t).items():
                for day,ids in f.groupby('target_date',sort=False).indices.items():
                    r=dict(fold=fold['name'],model=name,day=day,definition=definition,event_hours=int(np.asarray(a)[ids].sum()),missed_hours=int((np.asarray(a)[ids]&~np.asarray(b)[ids]&valid[ids]).sum()))
                    if daily is not None and day in daily.index:
                        for k in ['G2V_regret','V2G_regret']:r[k]=float(daily.loc[day,k])
                    dayrows.append(r)
    write(folder/'training_thresholds.json',thresholds_by_fold); pd.DataFrame(allrows).to_csv(ROOT/'docs/V16_DEFINITION_SCORES.csv',index=False)
    pd.DataFrame(dayrows).to_csv(folder/'definition_daily.csv',index=False)
    associations=[]
    d=pd.DataFrame(dayrows)
    for (model,definition),g in d.groupby(['model','definition']):
        for k in ['G2V_regret','V2G_regret']:
            valid=g.dropna(subset=[k]); x=valid.missed_hours
            associations.append(dict(model=model,definition=definition,metric=k,days=len(valid),missed_hour_rank_correlation=float(x.corr(valid[k],method='spearman')) if len(valid) and x.nunique()>1 else None,mean_regret_when_event=float(valid.loc[valid.event_hours>0,k].mean()),mean_regret_without_event=float(valid.loc[valid.event_hours==0,k].mean()),causal_claim=False))
    pd.DataFrame(associations).to_csv(ROOT/'docs/V16_EVENT_EV_ASSOCIATION.csv',index=False)
    # Same hyperparameters and training calendar end, old fitted artifacts only.
    db=connect();raw=latest(db,'land');db.close(); horizon=[];predictions=[]
    d1=joblib.load(ROOT/'artifacts/v11/training/2023_2024__hgb.joblib');d2=joblib.load(V15/'model_study/spring2025_baseline.joblib')
    assert d1['model'].get_params()==d2['model'].get_params()
    for cohort,start,end in [('validation','2025-04-01','2025-06-30'),('seasonal_stress','2025-07-01','2025-09-30')]:
        times=pd.date_range(start,pd.Timestamp(end)+pd.Timedelta(hours=23),freq='h',tz='Asia/Seoul')
        for ahead,bundle in [(1,d1),(2,d2)]:
            f=build11(raw,targets=times,days_ahead=ahead);p=predict11(bundle,f) if ahead==1 else predict15(bundle,f); ev,_=ev_metrics(f,p)
            horizon.append(dict(cohort=cohort,days_ahead=ahead,train_end='2024-12-31',hyperparameters_identical=True,availability='event-time replay, original vintages not certified',**scored(f,p,150),**ev))
            out=f[['timestamp','target_date','actual_smp']].copy();out['predicted_smp']=p;out['days_ahead']=ahead;predictions.append(out)
    pd.DataFrame(horizon).to_csv(ROOT/'docs/V16_HORIZON_COMPARISON.csv',index=False);pd.concat(predictions).to_csv(folder/'horizon_predictions.csv',index=False)
    print('Definition scores and matched HGB horizon comparisons saved; new model fits0',flush=True)

if __name__=='__main__':run()
