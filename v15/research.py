"""Preregistered expanding-window validation; failed trials never become candidates."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import hashlib,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from lightgbm import LGBMRegressor,LGBMClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from v11.store import ROOT,connect,latest,now
from v11.features import FEATURES
from v12.research import predict as oldpredict
from v14.research import ev_metrics,scored
from v15.features import build,event_labels
HOME=ROOT/'artifacts/v15'
FOLDS=[dict(name='winter2024',train_end='2024-06-30',cal_start='2024-07-01',cal_end='2024-09-30',val_start='2024-10-01',val_end='2024-12-31'),dict(name='spring2025',train_end='2024-12-31',cal_start='2025-01-01',cal_end='2025-03-31',val_start='2025-04-01',val_end='2025-06-30'),dict(name='winter2025',train_end='2025-06-30',cal_start='2025-07-01',cal_end='2025-09-30',val_start='2025-10-01',val_end='2025-12-31')]
NAMES=['event_features_lgb','relative_event_weight','relative_two_stage','event_analog']
HYPOTHESES={'event_features_lgb':'Known pre-origin ramp frequency and volatility may distinguish plateau from surprise transitions','relative_event_weight':'Regime-normalized uplift labels avoid overemphasizing the2023 high-price level','relative_two_stage':'Calibrated relative-event occurrence plus conditional severity separates rare jumps from ordinary hours','event_analog':'Local analog prediction tests whether similar known information actually identifies price jumps'}

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2,default=str)); os.replace(tmp,path)

def lgb(): return LGBMRegressor(n_estimators=160,num_leaves=20,min_child_samples=35,learning_rate=.04,n_jobs=4,random_state=42,verbosity=-1)

def predict(b,f):
    if b['kind']=='blend': return (1-b['weight'])*predict(b['base'],f)+b['weight']*predict(b['event'],f)
    if b['kind']=='old': return oldpredict(b['bundle'],f)
    if b['kind']=='baseline': return f.daily.to_numpy()+b['model'].predict(f[FEATURES])
    if b['kind']=='two_stage':
        raw=np.clip(b['classifier'].predict_proba(f[b['features']])[:,1],1e-6,1-1e-6); logits=np.log(raw/(1-raw)).reshape(-1,1)
        p=b['calibrator'].predict_proba(logits)[:,1] if b['calibrator'] is not None else np.full(len(f),b['prior'])
        base=predict(b['anchor'],f); severity=f.daily.to_numpy()+b['expert'].predict(f[b['features']])
        return base+p*np.maximum(0,severity-base)
    x=f[b['features']]
    if b['kind']=='analog': x=b['scaler'].transform(x)
    r=b['model'].predict(x)
    return f.daily.to_numpy()+r*(f.std168.to_numpy()+5 if b['kind']=='normalized' else 1)

def baseline(train,fold):
    if fold['name']=='winter2025': return dict(kind='old',bundle=joblib.load(ROOT/'artifacts/v12/study/hgb_none.joblib'))
    m=HistGradientBoostingRegressor(max_iter=250,max_leaf_nodes=31,learning_rate=.05,early_stopping=False,random_state=42); m.fit(train[FEATURES],train.actual_smp-train.daily); return dict(kind='baseline',model=m)

def fit(name,train,cal,cols,anchor):
    y=(train.actual_smp-train.daily).to_numpy(); x=train[cols]; b=dict(features=cols,hypothesis=HYPOTHESES[name],train_end=str(train.target_date.max()))
    if name=='relative_two_stage':
        labels=event_labels(train); classifier=LGBMClassifier(n_estimators=140,num_leaves=15,min_child_samples=40,class_weight='balanced',n_jobs=4,random_state=42,verbosity=-1); classifier.fit(x,labels)
        prob=np.clip(classifier.predict_proba(cal[cols])[:,1],1e-6,1-1e-6); labels_cal=event_labels(cal); calibrator=None
        if len(np.unique(labels_cal))==2: calibrator=LogisticRegression(random_state=42).fit(np.log(prob/(1-prob)).reshape(-1,1),labels_cal)
        expert=lgb(); expert.fit(x[labels],y[labels]); b.update(kind='two_stage',classifier=classifier,calibrator=calibrator,prior=float(np.mean(labels_cal)),expert=expert,anchor=anchor); return b
    if name=='event_analog':
        cols=['hour_sin','hour_cos','dow_sin','dow_cos','daily','mean168','std168','origin_jump6','up15_fraction168','down15_fraction168','profile_rank']; b['features']=cols
        scaler=StandardScaler().fit(train[cols]); model=KNeighborsRegressor(n_neighbors=48,weights='distance',n_jobs=4); model.fit(scaler.transform(train[cols]),y); b.update(kind='analog',scaler=scaler,model=model); return b
    model=lgb()
    if name=='relative_event_weight':
        weights=1+4*event_labels(train)+2*((train.actual_smp-train.previous_actual_smp)<=-15).to_numpy(); model.fit(x,y/(train.std168.to_numpy()+5),sample_weight=weights); b.update(kind='normalized',model=model)
    else: model.fit(x,y); b.update(kind='residual',model=model)
    return b

def run():
    folder=HOME/'model_study'; folder.mkdir(parents=True,exist_ok=True)
    design=dict(at=now(),folds=FOLDS,seed=42,new_hypotheses=HYPOTHESES,features='V14 price33 plus9 known pre-origin event features',event_target='actual minus D-2 same-hour known price >=20KRW/kWh; fixed before scores',qualification=dict(mean_MAE_ratio_max=1.03,mean_peak_MAE_ratio_max=.98,mean_G2V_regret_ratio_max=1.02,mean_V2G_regret_ratio_max=1.0,mean_absolute_peak_F1_not_worse=True,max_fold_MAE_ratio=1.2),all_historical_periods_previously_viewed=True,year2026_used_for_diagnosis_only=True,max_threads=4,checkpoint_not_candidate=True)
    manifest=folder/'design_before_scores.json'
    if not manifest.exists(): write(manifest,design)
    matrix=folder/'frozen_matrix.csv'
    if matrix.exists(): f=pd.read_csv(matrix, float_precision='round_trip'); f.timestamp=pd.to_datetime(f.timestamp,utc=True).dt.tz_convert('Asia/Seoul'); cols=json.loads((folder/'features.json').read_text())
    else:
        db=connect(); raw=latest(db,'land'); db.close(); f,cols=build(raw); f.to_csv(matrix,index=False); write(folder/'features.json',cols)
    rows=[]
    for fold in FOLDS:
        train=f[f.target_date<=fold['train_end']].reset_index(drop=True); cal=f[(f.target_date>=fold['cal_start'])&(f.target_date<=fold['cal_end'])].reset_index(drop=True); val=f[(f.target_date>=fold['val_start'])&(f.target_date<=fold['val_end'])].reset_index(drop=True)
        basepath=folder/(fold['name']+'_baseline.joblib')
        anchor=joblib.load(basepath) if basepath.exists() else baseline(train,fold)
        if not basepath.exists(): joblib.dump(anchor,basepath)
        out=val[['timestamp','target_date','actual_smp','previous_actual_smp','daily']].copy()
        for name in ['baseline']+NAMES:
            checkpoint=folder/(fold['name']+'_'+name+'.joblib'); resultpath=folder/(fold['name']+'_'+name+'.json'); started=time.monotonic()
            try:
                b=anchor if name=='baseline' else joblib.load(checkpoint) if checkpoint.exists() else fit(name,train,cal,cols,anchor)
                if not checkpoint.exists(): joblib.dump(b,checkpoint)
                p=predict(b,val); out[name]=p
                if resultpath.exists(): result=json.loads(resultpath.read_text())
                else:
                    ev,daily=ev_metrics(val,p); daily.to_csv(folder/(fold['name']+'_'+name+'_ev.csv'),index=False)
                    result=dict(name=name,fold=fold['name'],train_end=fold['train_end'],calibration=[fold['cal_start'],fold['cal_end']],validation=[fold['val_start'],fold['val_end']],status='success',**scored(val,p,float(train.actual_smp.quantile(.95))),**ev,relative_event_F1=f1_score(event_labels(val),p-val.daily.to_numpy()>=20,zero_division=0),seconds=time.monotonic()-started)
                    write(resultpath,result)
                rows.append(result); print(fold['name'],name,'MAE',round(result['MAE'],3),'peak',round(result['peak_MAE'],2),'V2G regret',round(result['V2G_regret'],2),flush=True)
            except Exception as exc: rows.append(dict(name=name,fold=fold['name'],status='failed',reason=type(exc).__name__+': '+str(exc))); print('FAILED',fold['name'],name,type(exc).__name__,flush=True)
            pd.DataFrame(rows).to_csv(folder/'cross_validation.csv',index=False)
        out.to_csv(folder/(fold['name']+'_predictions.csv'),index=False)
    summary=[]
    for name in NAMES:
        a=pd.DataFrame(rows); c=a[(a.name==name)&a.status.eq('success')].set_index('fold'); b=a[(a.name=='baseline')&a.status.eq('success')].set_index('fold')
        qualified=len(c)==3; record=dict(name=name)
        for key in ['MAE','peak_MAE','G2V_regret','V2G_regret']: record[key+'_ratio']=float((c[key]/b[key]).mean()) if len(c)==3 else None
        qualified=bool(qualified and record['MAE_ratio']<=1.03 and record['peak_MAE_ratio']<=.98 and record['G2V_regret_ratio']<=1.02 and record['V2G_regret_ratio']<=1 and (c.MAE/b.MAE).max()<=1.2 and c.peak_F1.mean()>=b.peak_F1.mean())
        record.update(qualified=qualified,not_promoted=True); summary.append(record)
        if qualified:
            dest=HOME/'candidates'; dest.mkdir(exist_ok=True); bundle=joblib.load(folder/('winter2025_'+name+'.joblib')); joblib.dump(bundle,dest/(name+'.joblib'))
    write(folder/'qualification.json',dict(at=now(),results=summary,qualified=[r['name'] for r in summary if r['qualified']],operating_model_replaced=False)); print('Qualification:',summary,flush=True)

if __name__=='__main__': run()
