"""Three bounded follow-ups motivated by training-level drift, validation only."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from v11.store import connect,latest,now
from v11.features import build,FEATURES
from v11.train import factories
from v12.research import HOME,predict,risk_predict,scored

def main():
    folder=HOME/'relative_followup'; folder.mkdir(parents=True,exist_ok=True)
    db=connect(); raw=latest(db,'land'); db.close(); data=build(raw,days_ahead=2)
    train=data[data.target_date<'2025-07-01']; cal=data[(data.target_date>='2025-07-01')&(data.target_date<'2025-10-01')]
    val=data[(data.target_date>='2025-10-01')&(data.target_date<'2026-01-01')].reset_index(drop=True)
    threshold=max(15,float((train.actual_smp-train.daily).quantile(.95))); labels=train.actual_smp-train.daily>=threshold
    design=dict(at=now(),motivation='Absolute training q95=241.63 is dominated by earlier price levels; predict positive profile deviation instead. Original classifier Brier did not beat constant prior.',
        label='actual SMP minus known two-day profile >= train-only q95 (floor15)',threshold=threshold,
        selection_period='2025Q4, same retrospective validation, not a new independent test',audit='not re-evaluated; only original frozen finalists retain their audits',budget='3 additional candidates; 30 total registered price candidates',features=FEATURES)
    (folder/'design_before_scores.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
    classifier=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,learning_rate=.05,random_state=42,early_stopping=False); classifier.fit(train[FEATURES],labels)
    cp=np.clip(classifier.predict_proba(cal[FEATURES])[:,1],1e-6,1-1e-6); ycal=(cal.actual_smp-cal.daily>=threshold).astype(int)
    calibrator=LogisticRegression().fit(np.log(cp/(1-cp)).reshape(-1,1),ycal) if ycal.nunique()==2 else None
    risk=dict(classifier=classifier,calibrator=calibrator,prior=float((ycal.sum()+1)/(len(ycal)+2)))
    gates=[.05,.1,.2,.35,.5]; prob=risk_predict(risk,cal); gate=max(gates,key=lambda t:f1_score(ycal,prob>=t,zero_division=0))
    expert=factories()['extra_trees'](); high=train[labels]; expert.fit(high[FEATURES],high.actual_smp-high.daily)
    anchor=joblib.load(HOME/'study/hgb_none.joblib'); rows=[]; experiments=[]
    for name,hard,strength in [('relative_soft',False,1),('relative_hard',True,1),('relative_soft_half',False,.5)]:
        bundle=dict(kind='relative_peak',risk=risk,expert=expert,anchor=anchor,threshold=gate,hard=hard,strength=strength)
        p=predict(bundle,val); rows.append(dict(name=name,split='selection',**scored(val,p,float(train.actual_smp.quantile(.95)))))
        experiments.append(dict(name=name,status='success',hypothesis=design['motivation'],classifier_parameters=classifier.get_params(),expert_parameters=expert.get_params(),strength=strength,gate_from_calibration=gate))
        joblib.dump(bundle,folder/(name+'.joblib'))
    pd.DataFrame(rows).to_csv(folder/'selection_metrics.csv',index=False)
    (folder/'experiments.json').write_text(json.dumps(experiments,indent=2,default=str),encoding='utf-8')
    print(pd.DataFrame(rows)[['name','MAE','peak_MAE','peak_F1','EV_regret']].to_string(index=False))

if __name__=='__main__': main()
