"""Checkpointed new hypotheses; historical validation is not new independent evidence."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse,hashlib,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from lightgbm import LGBMRegressor,LGBMRanker
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from v11.store import ROOT,connect,latest,now
from v11.evaluate import metrics
from v12.research import HOME as V12,predict as oldpredict
from v12.ev import EV,optimize,baseline,score
from v14.features import augment
HOME=ROOT/'artifacts/v14'

def write(path,obj):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8'); os.replace(tmp,path)

def xvalues(bundle,f):
    x=f[bundle['features']].replace([np.inf,-np.inf],np.nan)
    return bundle['imputer'].transform(x) if bundle.get('imputer') is not None else x

def predict(bundle,f):
    if bundle['kind']=='old': return oldpredict(bundle['bundle'],f)
    if bundle['kind']=='blend': return sum(w*predict(b,f) for w,b in zip(bundle['weights'],bundle['members']))
    if bundle['kind']=='regime':
        low=bundle['low'].predict(xvalues(bundle,f)); high=bundle['high'].predict(xvalues(bundle,f))
        return f.daily.to_numpy()+np.where(f.mean168.to_numpy()<bundle['threshold'],low,high)
    if bundle['kind']=='gru': return gru_predict(bundle,f)
    p=bundle['model'].predict(xvalues(bundle,f))
    if bundle['kind']=='rank':
        out=np.empty(len(f)); levels=predict(bundle['anchor'],f)
        for ids in f.groupby('target_date',sort=False).indices.values(): out[ids[np.argsort(p[ids],kind='stable')]]=np.sort(levels[ids])
        return out
    if bundle['kind']=='relative': return f.profile_mean.to_numpy()+p*(f.profile_std.to_numpy()+5)
    if bundle['kind']=='absolute': return p
    return f.daily.to_numpy()+p

def ev_metrics(f,p,policy='nominal'):
    ev=EV(degradation=5); rows=[]
    for day,ids in f.groupby('target_date',sort=True).indices.items():
        g=f.iloc[ids]; a=g.actual_smp.to_numpy(); pred=np.asarray(p)[ids]; times=pd.DatetimeIndex(g.timestamp)
        gplan=optimize(pred,times,ev)
        planner=EV(degradation=5,sell_ratio=.85 if policy=='export_haircut' else 1)
        vplan=optimize(pred,times,planner,v2g=True)
        estimated={}; realized={}; regrets={}
        if policy=='benefit_gate':
            if gplan['net_cost']-vplan['net_cost']<100: vplan=gplan
        for name,plan,v2g in [('G2V',gplan,False),('V2G',vplan,True)]:
            s=plan['schedule']; dt=s.connected_hours.to_numpy()
            estimated[name]=score(pred,s.charge_kw.to_numpy(),s.discharge_kw.to_numpy(),dt,ev)['net_cost']
            realized[name]=score(a,s.charge_kw.to_numpy(),s.discharge_kw.to_numpy(),dt,ev)['net_cost']
            key=(day,v2g)
            if key not in ORACLES: ORACLES[key]=optimize(a,times,ev,v2g=v2g)['net_cost']
            regrets[name]=realized[name]-ORACLES[key]
        immediate=baseline(a,times,ev)['net_cost']
        rows.append(dict(day=day,G2V_expected=estimated['G2V'],V2G_expected=estimated['V2G'],G2V_actual=realized['G2V'],V2G_actual=realized['V2G'],G2V_regret=regrets['G2V'],V2G_regret=regrets['V2G'],actual_saving_vs_immediate_G2V=immediate-realized['G2V'],actual_saving_vs_immediate_V2G=immediate-realized['V2G'],wrong_charge_kwh=float(np.sum(gplan['schedule'].charge_kw.to_numpy()[a>np.quantile(a,.75)])),reversal=estimated['G2V']-estimated['V2G']>1e-6 and realized['G2V']-realized['V2G']< -1e-6))
    result=pd.DataFrame(rows); summary=result.drop(columns='day').mean().to_dict(); summary['reversal_count']=int(result.reversal.sum()); summary['independent_days']=len(result)
    return summary,result
ORACLES={}

def scored(f,p,threshold):
    result=metrics(f,p); extra=metrics(f,p,threshold=threshold)
    neg=f.actual_smp.to_numpy()<150
    result.update(false_alarm_rate=result['false_alarms']/max(1,int(neg.sum())),training_q95_peak_MAE=extra['peak_MAE'],training_q95_F1=extra['peak_F1'])
    return result

def specifications():
    specs=[]
    def add(name,family,group='price',kind='residual',**kw): specs.append(dict(name=name,family=family,group=group,kind=kind,**kw))
    add('lgb_recency180','lgb',decay=180); add('lgb_recency365','lgb',decay=365)
    add('lgb_recent730','lgb',recent=730); add('hgb_recent365','hgb',recent=365)
    add('hgb_recent730','hgb',recent=730); add('et_relative_shape','et',kind='relative')
    add('lgb_relative_shape','lgb',kind='relative'); add('lgb_hour_rank','rank',kind='rank')
    add('xgb_pseudohuber','xgb'); add('cat_mae_recent','cat',recent=730)
    add('lgb_downturn_weight','lgb',downweight=True); add('hgb_level_regime','regime',kind='regime')
    for group in ['weather','demand']:
        add('lgb_lag_'+group,'lgb',group); add('hgb_lag_'+group,'hgb',group)
    add('lgb_lag_supply','lgb','supply'); add('lgb_lag_combined','lgb','combined')
    for q in [.1,.5,.9]: add('lgb_scaled_quantile_'+str(q),'lgb',kind='relative',quantile=q)
    add('gru_daily_shape_pilot','gru',kind='gru')
    return specs

def estimator(family,spec):
    if family=='lgb': return LGBMRegressor(n_estimators=180,num_leaves=20,min_child_samples=40,learning_rate=.04,n_jobs=4,verbosity=-1,random_state=42,**({'objective':'quantile','alpha':spec['quantile']} if 'quantile' in spec else {}))
    if family=='hgb': return HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=20,learning_rate=.05,early_stopping=False,random_state=42)
    if family=='et': return ExtraTreesRegressor(n_estimators=100,min_samples_leaf=15,max_features=.8,n_jobs=4,random_state=42)
    if family=='cat':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=180,depth=5,loss_function='MAE',thread_count=4,random_seed=42,verbose=False,allow_writing_files=False)
    if family=='xgb':
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=180,max_depth=4,learning_rate=.04,objective='reg:pseudohubererror',n_jobs=4,random_state=42,base_score=0)
    raise ValueError(family)

def fit(spec,train,cal,groups,anchor):
    f=train
    if spec.get('recent'): f=f[f.timestamp>=f.timestamp.max()-pd.Timedelta(days=spec['recent'])].copy()
    cols=groups[spec['group']]; bundle=dict(kind=spec['kind'],features=cols,operational_candidate=spec['group']=='price',trained_through=str(f.target_date.max()),spec=spec)
    y=f.actual_smp.to_numpy()-f.daily.to_numpy()
    if spec['kind']=='relative': y=(f.actual_smp.to_numpy()-f.profile_mean.to_numpy())/(f.profile_std.to_numpy()+5)
    if spec['family']=='gru': return fit_gru(bundle,f,cal)
    x=f[cols]; weights=np.ones(len(f))
    if spec.get('decay'): weights=np.exp(-(f.timestamp.max()-f.timestamp).dt.total_seconds().to_numpy()/86400/spec['decay'])
    if spec.get('downweight'): weights=np.where(f.actual_smp-f.previous_actual_smp<=-15,5.,1.)
    if spec['family']=='rank':
        model=LGBMRanker(n_estimators=140,num_leaves=15,learning_rate=.04,n_jobs=4,verbosity=-1,random_state=42)
        labels=f.groupby('target_date').actual_smp.rank(method='first').astype(int)-1
        model.fit(x,labels,group=f.groupby('target_date',sort=False).size().tolist()); bundle.update(model=model,anchor=anchor); return bundle
    if spec['family']=='regime':
        gate=float(f.mean168.median()); low=f.mean168.to_numpy()<gate
        a=estimator('hgb',{}); b=estimator('hgb',{}); a.fit(x[low],y[low]); b.fit(x[~low],y[~low]); bundle.update(low=a,high=b,threshold=gate); return bundle
    model=estimator(spec['family'],spec)
    if spec['family']=='et': bundle['imputer']=SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True); x=bundle['imputer'].fit_transform(x)
    model.fit(x,y,sample_weight=weights); bundle['model']=model
    return bundle

def sequences(f):
    # Seven already-known daily profile samples, not target observations.
    fields=['daily','weekly','origin_lag24','origin_lag168','mean24','mean168','origin_price']
    return f[fields].to_numpy(dtype='float32')[:,:,None]

def fit_gru(bundle,train,cal):
    import torch
    torch.set_num_threads(2); torch.manual_seed(42)
    from v14.sequence import ShapeGRU
    x=sequences(train); scale=float(np.std(x)); center=float(np.mean(x)); x=(x-center)/scale
    y=((train.actual_smp-train.daily).to_numpy()/scale).astype('float32')
    model=ShapeGRU(); optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    xx=torch.from_numpy(x); yy=torch.from_numpy(y); best=float('inf'); state=None
    cx=torch.from_numpy((sequences(cal)-center)/scale); cy=torch.from_numpy(((cal.actual_smp-cal.daily).to_numpy()/scale).astype('float32'))
    history=[]
    for epoch in range(6):
        model.train(); order=torch.randperm(len(xx))
        for batch in order.split(256):
            optimizer.zero_grad(); loss=torch.nn.functional.smooth_l1_loss(model(xx[batch]),yy[batch]); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): val=float(torch.mean(abs(model(cx)-cy)))
        history.append(val)
        if val<best: best=val; state={k:v.detach().clone() for k,v in model.state_dict().items()}
    bundle.update(state=state,center=center,scale=scale,calibration_loss=history,architecture='GRU over seven known summary samples, not a full168h sequence; small pilot')
    return bundle

def gru_predict(bundle,f):
    import torch
    from v14.sequence import ShapeGRU
    model=ShapeGRU(); model.load_state_dict(bundle['state']); model.eval(); torch.set_num_threads(2)
    with torch.no_grad(): p=model(torch.from_numpy((sequences(f)-bundle['center'])/bundle['scale'])).numpy()
    return f.daily.to_numpy()+p*bundle['scale']

def run():
    HOME.mkdir(parents=True,exist_ok=True); folder=HOME/'study'; folder.mkdir(exist_ok=True)
    db=connect(); prices=latest(db,'land'); db.close(); data,groups,provenance=augment(prices)
    design=dict(seed=42,origin='D-2 23:00 KST',train=['2023-01-01','2025-06-30'],calibration=['2025-07-01','2025-09-30'],selection=['2025-10-01','2025-12-31'],audit=['2026-01-01','2026-06-30'],not_unseen=True,known_october_not_used_for_selection=True,operating_model_promoted=False,specifications=specifications(),features=groups,external_provenance=provenance,selection_rule='Mean normalized actual G2V/V2G regret, MAE and peakMAE; price and EV improvements shown separately. Select once on2025Q4 before any2026 audit. No deployment promotion.',max_threads=4,source_sha256=hashlib.sha256(prices[['timestamp','smp']].to_csv(index=False).encode()).hexdigest())
    manifest=folder/'design_before_scores.json'
    if manifest.exists():
        old=json.loads(manifest.read_text()); assert old['source_sha256']==design['source_sha256'],'source changed; do not silently resume on new labels'
    else: write(manifest,design)
    def period(key):
        a,b=design[key]; return data[(data.target_date>=a)&(data.target_date<=b)].reset_index(drop=True)
    train,cal,val=period('train'),period('calibration'),period('selection'); q95=float(train.actual_smp.quantile(.95))
    rows=[]; experiments=[]; bundles={}; predictions={}; ORACLES.clear()
    anchor={'kind':'old','bundle':joblib.load(V12/'study/hgb_none.joblib'),'operational_candidate':True,'features':groups['price']}
    specs=[dict(name='V12_hgb_baseline',kind='baseline',group='price'),*specifications(),dict(name='shape_level_blend',kind='blend',group='price'),dict(name='rank_level_blend',kind='blend',group='price')]
    for spec in specs:
        name=spec['name']; started=time.monotonic(); checkpoint=folder/(name+'.joblib'); resultfile=folder/(name+'.result.json')
        try:
            if checkpoint.exists(): bundle=joblib.load(checkpoint)
            elif spec['kind']=='baseline': bundle=anchor
            elif spec['kind']=='blend':
                member=bundles['lgb_relative_shape' if name=='shape_level_blend' else 'lgb_hour_rank']
                bundle=dict(kind='blend',members=[anchor,member],weights=[.5,.5],operational_candidate=True)
            else: bundle=fit(spec,train,cal,groups,anchor)
            if not checkpoint.exists(): joblib.dump(bundle,checkpoint)
            p=predict(bundle,val); bundles[name]=bundle; predictions[name]=p
            if resultfile.exists(): record=json.loads(resultfile.read_text())
            else:
                ev,daily=ev_metrics(val,p); daily.to_csv(folder/(name+'.ev.csv'),index=False)
                record=dict(name=name,split='selection_2025Q4',operational_candidate=bundle['operational_candidate'],**scored(val,p,q95),**ev); write(resultfile,record)
            rows.append(record); experiments.append(dict(name=name,status='success',spec=spec,seconds=time.monotonic()-started,parameters=str(bundle.get('model',bundle.get('architecture',bundle['kind']))),features=bundle.get('features'),train=design['train'],calibration=design['calibration'],selection=design['selection'],seed=42))
            print(name,'MAE',round(record['MAE'],3),'G2V regret',round(record['G2V_regret'],2),'V2G regret',round(record['V2G_regret'],2),'reversals',record['reversal_count'],flush=True)
        except Exception as exc: experiments.append(dict(name=name,status='failed',reason=type(exc).__name__+': '+str(exc))); print(name,'FAILED',type(exc).__name__,str(exc),flush=True)
        write(folder/'experiments.json',experiments); pd.DataFrame(rows).to_csv(folder/'selection_metrics.csv',index=False)
    out=val[['timestamp','target_date','actual_smp','previous_actual_smp']].copy()
    for name,p in predictions.items(): out[name]=p
    out.to_csv(folder/'selection_predictions.csv',index=False)
    ranking=pd.DataFrame(rows); base=ranking[ranking.name=='V12_hgb_baseline'].iloc[0]
    ranking['decision_score']=sum(ranking[k]/max(float(base[k]),1e-6) for k in ['G2V_regret','V2G_regret','MAE','peak_MAE'])/4
    ranking=ranking.sort_values('decision_score'); top=ranking[(ranking.name!='V12_hgb_baseline')&ranking.operational_candidate].name.head(3).tolist()
    write(folder/'selection_before_audit.json',dict(top3=top,ranked=ranking.to_dict('records'),chosen_at=now(),not_promoted=True,warning='Historical periods previously inspected; none is a new independent test'))
    ranking.to_csv(folder/'selection_metrics.csv',index=False)
    # Budgeted dispatch-policy ablation on the frozen best price candidate, same validation labels.
    policies=[]
    for name in list(dict.fromkeys(['V12_hgb_baseline']+top[:1])):
        for policy in ['nominal','export_haircut','benefit_gate']:
            summary,daily=ev_metrics(val,predictions[name],policy); policies.append(dict(name=name,policy=policy,**summary)); daily.to_csv(folder/(name+'.policy_'+policy+'.csv'),index=False)
    pd.DataFrame(policies).to_csv(folder/'selection_policy_metrics.csv',index=False)
    audit=period('audit'); ORACLES.clear(); auditrows=[]; auditout=audit[['timestamp','target_date','actual_smp','previous_actual_smp']].copy()
    for name in ['V12_hgb_baseline']+top:
        p=predict(bundles[name],audit); auditout[name]=p; ev,daily=ev_metrics(audit,p); daily.to_csv(folder/(name+'.audit_ev.csv'),index=False)
        auditrows.append(dict(name=name,split='previously_seen_2026H1',**scored(audit,p,q95),**ev))
    auditout.to_csv(folder/'audit_predictions.csv',index=False); pd.DataFrame(auditrows).to_csv(folder/'audit_metrics.csv',index=False)
    print('Completed',len(rows)-1,'new candidate configurations; frozen top3',top,flush=True)

if __name__=='__main__': run()
