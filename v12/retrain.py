"""Explicit chronological candidate review on newly accumulated prices.

Creates a separate run directory. Never touches any operating/forward model.
"""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from v11.store import connect,latest,now
from v11.features import build,FEATURES
from v11.train import factories
from v12.research import HOME,predict,scored

def main():
    p=argparse.ArgumentParser(); p.add_argument('--as-of',required=True); args=p.parse_args()
    bound=pd.Timestamp(args.as_of)
    if bound.tzinfo is None: raise ValueError('--as-of must include timezone')
    if bound>pd.Timestamp.now(tz='UTC'): raise ValueError('future as-of not allowed')
    db=connect(); prices=latest(db,'land',as_of=bound.tz_convert('UTC').isoformat()); db.close(); data=build(prices,days_ahead=2)
    end=min(bound.tz_convert('Asia/Seoul').normalize(),data.timestamp.max().normalize()+pd.Timedelta(days=1)); test_start=end-pd.Timedelta(days=30); val_start=test_start-pd.Timedelta(days=90)
    train=data[data.timestamp<val_start]; val=data[(data.timestamp>=val_start)&(data.timestamp<test_start)].reset_index(drop=True); test=data[(data.timestamp>=test_start)&(data.timestamp<end)].reset_index(drop=True)
    if len(train)<365*24 or len(val)<60*24 or len(test)<25*24: raise ValueError('not enough complete chronological data')
    folder=HOME/'candidate_reviews'/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%f'); folder.mkdir(parents=True,exist_ok=False)
    q95=float(train.actual_smp.quantile(.95)); config=dict(at=now(),as_of=bound.isoformat(),periods=dict(train_end=str(val_start),validation_end=str(test_start),audit_end=str(end)),features=FEATURES,automatic_promotion=False,warning='Retrospective candidate review, not automatically an unseen test',q95=q95)
    (folder/'design_before_scores.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    rows=[]; bundles={}
    for family,create in list(factories().items())[:5]:
        for weighted in (False,True):
            estimator=create(); estimator.fit(train[FEATURES],train.actual_smp-train.daily,sample_weight=np.where(train.actual_smp>=150,6 if weighted else 1,1))
            name=family+('_peak' if weighted else '_base'); bundle=dict(kind='residual',model=estimator); bundles[name]=bundle; joblib.dump(bundle,folder/(name+'.joblib'))
            rows.append(dict(name=name,split='validation',**scored(val,predict(bundle,val),q95)))
    selected=pd.DataFrame(rows).sort_values(['MAE','peak_MAE']).name.head(3).tolist()
    (folder/'selection_before_audit.json').write_text(json.dumps(dict(top3=selected,automatic_promotion=False),indent=2),encoding='utf-8')
    for name in selected: rows.append(dict(name=name,split='audit_already_may_be_seen',**scored(test,predict(bundles[name],test),q95)))
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
    print('Candidate-only review saved:',folder)

if __name__=='__main__': main()
