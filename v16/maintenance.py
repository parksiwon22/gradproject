"""Announced-plan research ablation; current attachment vintage is unverified."""
import json,hashlib,shutil
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from v11.features import FEATURES
from v11.store import now
from v15.research import FOLDS,HOME as V15
from v14.research import scored,ev_metrics
from v16.common import HOME,ROOT,write,csv

EXTRA=['maintenance_total_mw','maintenance_lng_mw','maintenance_unit_count','maintenance_age_days']

def register():
    path=HOME/'maintenance_study/design_before_scores.json'
    if path.exists():return
    write(path,dict(at=now(),amendment='New public weekly forecast documents discovered after initial information gate; permit retrospective-only ablation, not operational training',why='Known planned unavailable capacity may distinguish similar demand/price histories with different supply constraints',source='198 public2023-2026 weekly plans, current attachment original vintage unverified',features=FEATURES+EXTRA,folds=[FOLDS[1],FOLDS[2]],minimum_train_days=365,minimum_validation_coverage=.8,estimator='HGB250 leaves31 lr.05 seed42 early_stoppingFalse',common_train_rows_for_both_models=True,hypotheses=['same price21 HGB on common complete days','same HGB plus4 announced maintenance features'],qualification='Both historical folds:MAE<=baseline,peakMAE<=.98baseline,F1>=baseline,G2V/V2G regret<=baseline',no_2026_selection=True,no_operational_candidate_even_if_improved=True,warning='Issue time filter applied but current PDF hash not certified historical version; first receipt now, do not claim leakage-free operational vintage replay'))

def attach(frame,records,checks):
    rows=pd.DataFrame(records)
    for name in ['planned_start','planned_end']:rows[name]=pd.to_datetime(rows[name]).dt.tz_localize('Asia/Seoul')
    invalid=set(rows.loc[(rows.planned_end<=rows.planned_start)|~np.isfinite(rows.capacity_mw)|(rows.capacity_mw<=0),'document_id'].astype(str))
    good={str(c['document_id']) for c in checks if c['parsed_rows']>0 and c['ambiguous_rows']==0 and c.get('period_start') and c.get('period_end')}
    docs={}
    for ident,g in rows.groupby('document_id'):
        if str(ident) not in good or str(ident) in invalid or not g.posted_at.iloc[0]:continue
        issue=pd.Timestamp(g.posted_at.iloc[0]);start=pd.Timestamp(g.plan_period_start.iloc[0],tz='Asia/Seoul');end=pd.Timestamp(g.plan_period_end.iloc[0],tz='Asia/Seoul')+pd.Timedelta(days=1)
        if issue.tzinfo is None or not 0<(end-start).days<=14:continue
        docs[str(ident)]=(issue,start,end,g)
    f=frame.copy();output=[]
    for day,ids in f.groupby('target_date',sort=True).indices.items():
        g=f.iloc[ids];origin=pd.Timestamp(g.origin_timestamp.iloc[0]);target=pd.Timestamp(day,tz='Asia/Seoul')
        candidates=[(key,value) for key,value in docs.items() if value[0]<=origin and value[1]<=target and value[2]>=target+pd.Timedelta(days=1)]
        if not candidates:continue
        ident,(issue,start,end,plans)=max(candidates,key=lambda item:(item[1][0],item[0])); values=[]
        for t in g.timestamp:
            active=plans[(plans.planned_start<t+pd.Timedelta(hours=1))&(plans.planned_end>t)].copy()
            if active.unit.duplicated().any():values=[];break  # ambiguous overlapping unit; no double counting
            weights=((active.planned_end.clip(upper=t+pd.Timedelta(hours=1))-active.planned_start.clip(lower=t))/pd.Timedelta(hours=1)).to_numpy()
            mw=active.capacity_mw.to_numpy()*weights
            values.append([float(mw.sum()),float(mw[active.fuel.eq('LNG')].sum()),len(active),(origin-issue).total_seconds()/86400])
        if len(values)!=24:continue
        for col,value in zip(EXTRA,np.asarray(values).T):g=g.assign(**{col:value})
        g=g.assign(maintenance_document=ident,maintenance_posted_at=issue.isoformat(),source_first_seen=plans.first_seen.iloc[0],historical_vintage_certified=False);output.append(g)
    return pd.concat(output,ignore_index=True) if output else f.iloc[0:0]

def run():
    register();folder=HOME/'maintenance_study';records=json.loads((HOME/'outages/extracted_rows.json').read_text(encoding='utf-8'));checks=json.loads((HOME/'outages/extraction_checks.json').read_text())
    f=csv(V15/'model_study/frozen_matrix.csv');f.timestamp=pd.to_datetime(f.timestamp,utc=True).dt.tz_convert('Asia/Seoul');f.origin_timestamp=pd.to_datetime(f.origin_timestamp,utc=True).dt.tz_convert('Asia/Seoul')
    f=attach(f,records,checks);f.to_csv(folder/'retrospective_matrix.csv',index=False);write(folder/'coverage.json',dict(rows=len(f),years=f.groupby(f.timestamp.dt.year).size().to_dict(),source='Forecasted planned maintenance, not actual outage or full generation availability',operationally_certified=False))
    scores=[];qual=[]
    for fold in [FOLDS[1],FOLDS[2]]:
        train=f[f.target_date<=fold['train_end']];val=f[(f.target_date>=fold['val_start'])&(f.target_date<=fold['val_end'])].reset_index(drop=True)
        expected=(pd.Timestamp(fold['val_end'])-pd.Timestamp(fold['val_start'])).days+1
        if len(train)<365*24 or len(val)<.8*expected*24:
            qual.append(dict(fold=fold['name'],status='skipped_inadequate_complete_document_coverage',train_days=len(train)//24,validation_days=len(val)//24,expected_days=expected));continue
        out=val[['timestamp','target_date','actual_smp','previous_actual_smp','daily','maintenance_document','maintenance_posted_at','source_first_seen']].copy();foldscores={}
        fingerprint=hashlib.sha256(train[['target_date','actual_smp']+FEATURES+EXTRA].to_csv(index=False).encode()).hexdigest()
        for name,cols in [('price_only_common',FEATURES),('price_plus_maintenance',FEATURES+EXTRA)]:
            path=folder/(fold['name']+'_'+name+'.joblib')
            cached=joblib.load(path) if path.exists() else None
            if cached and cached.get('training_sha256')==fingerprint:model=cached['model']
            else:
                if path.exists():
                    backup=folder/'superseded_before_interval_QC';backup.mkdir(exist_ok=True);dest=backup/path.name
                    if not dest.exists():shutil.copy2(path,dest)
                model=HistGradientBoostingRegressor(max_iter=250,max_leaf_nodes=31,learning_rate=.05,early_stopping=False,random_state=42);model.fit(train[cols],train.actual_smp-train.daily)
                joblib.dump(dict(model=model,features=cols,trained_through=fold['train_end'],training_sha256=fingerprint,kind='RETROSPECTIVE_UNVERIFIED_ATTACHMENT_VINTAGE_NOT_OPERATIONAL'),path)
            p=val.daily.to_numpy()+model.predict(val[cols]);out[name]=p;ev,daily=ev_metrics(val,p);daily.to_csv(folder/(fold['name']+'_'+name+'_ev.csv'),index=False)
            result=dict(fold=fold['name'],model=name,train_days=len(train)//24,validation_days=len(val)//24,operational_candidate=False,**scored(val,p,float(train.actual_smp.quantile(.95))),**ev);scores.append(result);foldscores[name]=result
            pd.DataFrame(scores).to_csv(ROOT/'docs/V16_MAINTENANCE_ABLATION.csv',index=False);print(fold['name'],name,'MAE',result['MAE'],'peak',result['peak_MAE'],flush=True)
        base=foldscores['price_only_common'];new=foldscores['price_plus_maintenance'];ok=new['MAE']<=base['MAE'] and new['peak_MAE']<=.98*base['peak_MAE'] and new['peak_F1']>=base['peak_F1'] and new['G2V_regret']<=base['G2V_regret'] and new['V2G_regret']<=base['V2G_regret']
        qual.append(dict(fold=fold['name'],historical_metric_passed=bool(ok),operational_candidate=False));out.to_csv(folder/(fold['name']+'_predictions.csv'),index=False)
    write(folder/'qualification.json',dict(folds=qual,qualified_operating_models=[],reason='No original historical attachment receipt/vintage certification; no promotion'))
    write(ROOT/'docs/V16_MAINTENANCE_COVERAGE.json',json.loads((folder/'coverage.json').read_text()))
    print('Maintenance research complete; no operating model changed',flush=True)

if __name__=='__main__':run()
