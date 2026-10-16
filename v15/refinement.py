"""Two preregistered shrinkage followups; reuse fitted models, no2026 tuning."""
import json,joblib,numpy as np,pandas as pd
from v15.research import HOME,FOLDS,write
from v14.research import ev_metrics,scored
from v15.features import event_labels
from sklearn.metrics import f1_score
from v11.store import now

def run():
    src=HOME/'model_study'; folder=HOME/'shrinkage_followup'; folder.mkdir(exist_ok=True)
    design=dict(at=now(),reason='Relative two-stage improves mean peak and EV regret but violates3% MAE allowance; shrink towards frozen baseline',weights=[.25,.5],same_qualification_as_initial=True,uses_only_already_viewed_historical_CV=True,no2026_selection=True,no_additional_training=True)
    if not (folder/'design_before_scores.json').exists(): write(folder/'design_before_scores.json',design)
    rows=[]; matrix=pd.read_csv(src/'frozen_matrix.csv'); matrix.timestamp=pd.to_datetime(matrix.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
    for fold in FOLDS:
        old=pd.read_csv(src/(fold['name']+'_predictions.csv')); f=matrix[(matrix.target_date>=fold['val_start'])&(matrix.target_date<=fold['val_end'])].reset_index(drop=True)
        for weight in design['weights']:
            name='two_stage_shrink_'+str(weight); path=folder/(fold['name']+'_'+name+'.json')
            p=(1-weight)*old.baseline.to_numpy()+weight*old.relative_two_stage.to_numpy()
            if path.exists(): result=json.loads(path.read_text())
            else:
                ev,daily=ev_metrics(f,p); result=dict(name=name,fold=fold['name'],**scored(f,p,241.63),**ev,relative_event_F1=f1_score(event_labels(f),p-f.daily>=20,zero_division=0)); write(path,result)
            rows.append(result); pd.DataFrame(rows).to_csv(folder/'cross_validation.csv',index=False); print(fold['name'],name,'MAE',round(result['MAE'],3),flush=True)
    initial=pd.read_csv(src/'cross_validation.csv'); baseline=initial[initial.name=='baseline'].set_index('fold'); allrows=[]
    for weight in design['weights']:
        name='two_stage_shrink_'+str(weight); r=pd.DataFrame(rows).query('name==@name').set_index('fold'); result=dict(name=name)
        for k in ['MAE','peak_MAE','G2V_regret','V2G_regret']: result[k+'_ratio']=float((r[k]/baseline[k]).mean())
        ok=result['MAE_ratio']<=1.03 and result['peak_MAE_ratio']<=.98 and result['G2V_regret_ratio']<=1.02 and result['V2G_regret_ratio']<=1 and r.peak_F1.mean()>=baseline.peak_F1.mean() and (r.MAE/baseline.MAE).max()<=1.2
        result.update(qualified=bool(ok),not_promoted=True); allrows.append(result)
        if ok:
            bundle=dict(kind='blend',weight=weight,base=joblib.load(src/'winter2025_baseline.joblib'),event=joblib.load(src/'winter2025_relative_two_stage.joblib'),features=json.loads((src/'features.json').read_text()),trained_through='2025-06-30',selection_through='2025-12-31')
            dest=HOME/'candidates'; dest.mkdir(exist_ok=True); joblib.dump(bundle,dest/(name+'.joblib'))
    write(folder/'qualification.json',dict(results=allrows,operating_model_replaced=False)); print(allrows,flush=True)

if __name__=='__main__': run()
