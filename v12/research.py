"""Bounded hypothesis-driven peak study. Previously seen periods are audits."""
import hashlib
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,f1_score
from v11.store import ROOT,connect,latest,now
from v11.features import FEATURES,build
from v11.train import factories
from v11.evaluate import metrics

HOME=ROOT/'artifacts/v12'

def risk_predict(bundle,f):
    raw=bundle['classifier'].predict_proba(f[FEATURES])[:,1]
    if bundle.get('calibrator') is None: return np.full(len(f),bundle['prior'])
    logits=np.log(np.clip(raw,1e-6,1-1e-6)/(1-np.clip(raw,1e-6,1-1e-6))).reshape(-1,1)
    return bundle['calibrator'].predict_proba(logits)[:,1]

def predict(bundle,f):
    kind=bundle['kind']; baseline=f.daily.to_numpy()
    if kind=='ensemble': return sum(w*predict(b,f) for w,b in zip(bundle['weights'],bundle['members']))
    if kind=='profile': return .69*f.daily.to_numpy()+.31*f.weekly.to_numpy()
    if kind=='ramp':
        delta=bundle['model'].predict(f[FEATURES]); anchor=predict(bundle['anchor'],f)
        out=np.empty(len(f))
        for ids in f.groupby('target_date',sort=False).indices.values(): out[ids]=anchor[ids[0]]+np.r_[0,np.cumsum(delta[ids[1:]])]
        return out
    if kind=='daily_peak':
        out=predict(bundle['anchor'],f); daily=daily_features(f)
        maxima=bundle['maximum'].predict(daily); hours=np.clip(bundle['hour'].predict(daily),0,23)
        for j,ids in enumerate(f.groupby('target_date',sort=True).indices.values()): out[ids]+=np.maximum(0,maxima[j]-out[ids].max())*np.exp(-.5*((np.arange(24)-hours[j])/2)**2)
        return out
    if kind=='two_stage':
        prob=risk_predict(bundle['risk'],f); expert=bundle['expert'].predict(f[FEATURES]); base=predict(bundle['anchor'],f)
        if bundle['gate']=='hard': prob=(prob>=bundle['threshold']).astype(float)
        if bundle['gate']=='risk_warning_only': return base
        return (1-prob)*base+prob*expert
    if kind=='relative_peak':
        prob=risk_predict(bundle['risk'],f)
        if bundle['hard']: prob=(prob>=bundle['threshold']).astype(float)
        base=predict(bundle['anchor'],f); severity=bundle['expert'].predict(f[FEATURES])
        return base+bundle['strength']*prob*np.maximum(0,severity-(base-baseline))
    out=bundle['model'].predict(f[FEATURES])+(f.origin_price.to_numpy() if kind=='origin_delta' else baseline)
    if kind=='residual_correction': out=out+np.where(risk_predict(bundle['risk'],f)>=bundle['threshold'],bundle['correction'],0)
    return out

def daily_features(f):
    return f.groupby('target_date',sort=True)[FEATURES].agg(['mean','min','max']).to_numpy()

def scored(f,p,threshold):
    result=metrics(f,p); tail=metrics(f,p,threshold=threshold)
    result.update(training_q95_peak_MAE=tail['peak_MAE'],training_q95_F1=tail['peak_F1'],training_q95_hours=tail['peak_hours'],false_alarm_rate=float(np.mean((np.asarray(p)>=150)&(f.actual_smp.to_numpy()<150))))
    return result

def run():
    folder=HOME/'study'; folder.mkdir(parents=True,exist_ok=True)
    design=dict(at=now(),origin='D-2 23:00 KST; candidate earlier horizon, not certified universally unpublished',seed=42,
        train=['2023-01-01','2025-06-30'],calibration=['2025-07-01','2025-09-30'],selection=['2025-10-01','2025-12-31'],
        retrospective_audit=['2026-01-01','2026-06-30'],known_failure_diagnosis=['2026-10-01','2026-10-15'],
        warning='All historical periods may have been inspected in V11. Audit is not an unseen final test. Only newly issued immutable forward records can establish prospective performance. 2026Q3 excluded.',
        budget='at most30 registered candidates; CPU4 threads; no deep learning without a validated tabular improvement/data justification',
        selection_rule='rank by mean normalized MAE, peak MAE, EV regret and 1-F1; only validation-improved candidates audited; always keep baseline. No selection by 2026 or October labels.',
        historical_availability='final observed price snapshots, original historical publication versions unknown',features=FEATURES)
    (folder/'design_before_scores.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
    db=connect(); raw=latest(db,'land'); db.close(); data=build(raw,days_ahead=2)
    def period(a,b): return data[(data.target_date>=a)&(data.target_date<=b)].reset_index(drop=True)
    train=period(*design['train']); cal=period(*design['calibration']); val=period(*design['selection'])
    q95=float(train.actual_smp.quantile(.95)); threshold=max(150.,q95)
    design.update(rows={'train':len(train),'calibration':len(cal),'selection':len(val)},q95=q95,risk_threshold=150,source_hash=hashlib.sha256(raw[['timestamp','smp']].to_csv(index=False).encode()).hexdigest())
    (folder/'design_before_scores.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
    bundles={}; rows=[]; experiments=[]; predictions={}; timer=time.monotonic()
    def register(name,bundle,hypothesis,started=None):
        p=predict(bundle,val); record=dict(name=name,split='selection',**scored(val,p,q95))
        bundles[name]=bundle; predictions[name]=p; rows.append(record)
        experiments.append(dict(name=name,status='success',hypothesis=hypothesis,seconds=time.monotonic()-(started or timer),kind=bundle['kind'],parameters=bundle['model'].get_params() if 'model' in bundle else {},features=FEATURES))
        joblib.dump(bundle,folder/(name+'.joblib'))
        (folder/'experiments.json').write_text(json.dumps(experiments,indent=2,default=str),encoding='utf-8')
        pd.DataFrame(rows).to_csv(folder/'selection_metrics.csv',index=False)
        print(name,'MAE',round(record['MAE'],4),'peak',record['peak_MAE'],'F1',round(record['peak_F1'],3),flush=True)
    fs=factories()
    for family in ('hgb','extra_trees','catboost','lightgbm','xgboost'):
        for weighting in ('none','peak','ramp'):
            started=time.monotonic(); name=family+'_'+weighting
            try:
                estimator=fs[family](); weights=np.ones(len(train))
                if weighting=='peak': weights=np.where(train.actual_smp>=150,6.,1.)
                if weighting=='ramp': weights=np.where(abs(train.actual_smp-train.previous_actual_smp)>=15,5.,1.)
                estimator.fit(train[FEATURES],train.actual_smp-train.daily,sample_weight=weights)
                register(name,dict(kind='residual',model=estimator),{'none':'family inductive bias baseline','peak':'emphasize rare high-price underprediction','ramp':'emphasize transitions rather than constant high-price plateaus'}[weighting],started)
            except Exception as exc:
                experiments.append(dict(name=name,status='failed',reason=type(exc).__name__+': '+str(exc),seconds=time.monotonic()-started))
                (folder/'experiments.json').write_text(json.dumps(experiments,indent=2,default=str),encoding='utf-8')
    register('fixed_profile',dict(kind='profile'),'fixed profile guards against overfitting in tree learners')
    for quantile in (.75,.9):
        estimator=HistGradientBoostingRegressor(loss='quantile',quantile=quantile,max_iter=250,max_leaf_nodes=31,learning_rate=.05,random_state=42,early_stopping=False)
        estimator.fit(train[FEATURES],train.actual_smp-train.daily)
        register('quantile_'+str(quantile),dict(kind='residual',model=estimator),'upper conditional price scenario may reduce downside peak bias at false-alarm cost')
    classifier=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,learning_rate=.05,random_state=42,early_stopping=False)
    classifier.fit(train[FEATURES],train.actual_smp>=150)
    r=classifier.predict_proba(cal[FEATURES])[:,1]; labels=(cal.actual_smp>=150).astype(int)
    calibrator=None
    if labels.nunique()==2: calibrator=LogisticRegression().fit(np.log(np.clip(r,1e-6,1-1e-6)/(1-np.clip(r,1e-6,1-1e-6))).reshape(-1,1),labels)
    risk=dict(classifier=classifier,calibrator=calibrator,prior=float((labels.sum()+1)/(len(labels)+2)))
    joblib.dump(risk,folder/'risk.joblib')
    cp=risk_predict(risk,cal); gates=[.05,.1,.2,.35,.5]
    gate=max(gates,key=lambda x:f1_score(labels,cp>=x,zero_division=0))
    expert=fs['extra_trees'](); peak=train[train.actual_smp>=150]; expert.fit(peak[FEATURES],peak.actual_smp)
    anchor=bundles['hgb_none']
    for mode in ('soft','hard','risk_warning_only'):
        register('two_stage_'+mode,dict(kind='two_stage',risk=risk,expert=expert,anchor=anchor,gate=mode,threshold=gate),'calibrated occurrence probability separates peak risk from conditional peak price')
    basecal=predict(anchor,cal); active=cp>=gate
    correction=float(np.clip(np.mean((cal.actual_smp.to_numpy()-basecal)[active]),0,40)) if active.any() else 0.
    register('risk_residual_correction',dict(kind='residual_correction',model=anchor['model'],risk=risk,threshold=gate,correction=correction),'out-of-training calibration residual corrects risk-specific underprediction')
    estimator=fs['extra_trees'](); estimator.fit(train[FEATURES],train.actual_smp-train.origin_price)
    register('origin_delta',dict(kind='origin_delta',model=estimator),'predict change from known origin price rather than daily profile')
    estimator=fs['hgb'](); estimator.fit(train[FEATURES],train.actual_smp-train.previous_actual_smp)
    register('recursive_ramp',dict(kind='ramp',model=estimator,anchor=anchor),'forecast changes and integrate from a predicted first hour; no actual target-hour anchor')
    maximum=fs['hgb'](); hour=fs['hgb'](); daily=daily_features(train)
    maximum.fit(daily,train.groupby('target_date',sort=True).actual_smp.max())
    hour.fit(daily,train.groupby('target_date',sort=True).actual_smp.apply(lambda s:float(s.to_numpy().argmax())))
    register('daily_peak_shape',dict(kind='daily_peak',maximum=maximum,hour=hour,anchor=anchor),'daily maximum and timing prediction adds a localized peak shape')
    ranking=pd.DataFrame(rows).sort_values('MAE'); chosen=ranking.name.head(3).tolist()
    # Ensemble membership and weights come only from selection-period scores.
    register('validation_ensemble',dict(kind='ensemble',weights=[1/3]*3,members=[bundles[x] for x in chosen]),'combine validation-selected error patterns')
    register('base_quantile_blend',dict(kind='ensemble',weights=[.7,.3],members=[anchor,bundles['quantile_0.9']]),'limited upper-tail adjustment bounds false-alarm cost')
    baseline=next(r for r in rows if r['name']=='hgb_none')
    for r in rows:
        r['selection_score']=np.mean([r['MAE']/baseline['MAE'],(r['peak_MAE'] or 0)/(baseline['peak_MAE'] or 1),r['EV_regret']/max(baseline['EV_regret'],1e-9),1-r['peak_F1']])
        r['validation_improved']=bool(r['MAE']<=baseline['MAE'] and (r['peak_MAE'] or 0)<=(baseline['peak_MAE'] or 0) and r['EV_regret']<=baseline['EV_regret'] or r['peak_F1']>baseline['peak_F1'] and r['MAE']<=baseline['MAE']*1.1 and r['EV_regret']<=baseline['EV_regret'])
    ranking=pd.DataFrame(rows).sort_values(['selection_score','MAE']); top=ranking[ranking.validation_improved].name.head(3).tolist()
    selection=dict(at=now(),top3=top,baseline='hgb_none',gate_from_calibration=gate,calibrated_validation_Brier=brier_score_loss(val.actual_smp>=150,risk_predict(risk,val)),not_promoted=True,remaining_top_diagnostic=ranking.name.head(3).tolist())
    (folder/'selection_before_audit.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    pd.DataFrame(rows).to_csv(folder/'selection_metrics.csv',index=False)
    valout=val[['timestamp','target_date','actual_smp','previous_actual_smp']].copy()
    for n,p in predictions.items(): valout[n]=p
    valout.to_csv(folder/'selection_predictions.csv',index=False)
    auditrows=[]
    for periodname,limits in [('retrospective_audit',design['retrospective_audit']),('known_failure_diagnosis',design['known_failure_diagnosis'])]:
        frame=period(*limits); out=frame[['timestamp','target_date','actual_smp','previous_actual_smp']].copy()
        for name in list(dict.fromkeys(['hgb_none']+top)):
            p=predict(bundles[name],frame); out[name]=p; auditrows.append(dict(name=name,split=periodname,**scored(frame,p,q95)))
        out['peak_probability']=risk_predict(risk,frame); out.to_csv(folder/(periodname+'_predictions.csv'),index=False)
    pd.DataFrame(auditrows).to_csv(folder/'audit_metrics.csv',index=False)
    # Standalone fixed baseline stays separate from the existing V11 operating model.
    joblib.dump(dict(bundle=anchor,risk=risk,features=FEATURES,trained_through=train.target_date.max(),data_hash=design['source_hash'],horizon_days=2,kind='V12_RESEARCH_BASELINE_NOT_V11_REPLACEMENT'),HOME/'forward_model.joblib')
    print('Finished registered candidates:',len(rows),'seconds',round(time.monotonic()-timer,1),'top validation-qualified',top,flush=True)

if __name__=='__main__': run()
