"""Immutable D-2 23h-input forecasts and later independent reconciliations."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from v11.store import connect,latest,now
from v11.features import FEATURES,build
from v11.forecast import write_once
from v11.evaluate import metrics
from v12.research import HOME,predict,risk_predict

def issue(clock=None):
    clock=pd.Timestamp(clock) if clock is not None else pd.Timestamp.now(tz='Asia/Seoul')
    origin=clock.normalize()+pd.Timedelta(hours=23)
    if clock<origin: origin-=pd.Timedelta(days=1)
    target=origin.normalize()+pd.Timedelta(days=2); times=pd.date_range(target,periods=24,freq='h')
    folder=HOME/'forecasts'; folder.mkdir(parents=True,exist_ok=True)
    db=connect(); raw=latest(db,'land',as_of=origin.tz_convert('UTC').isoformat()); current=latest(db,'land',as_of=clock.tz_convert('UTC').isoformat()); db.close()
    path=HOME/'forward_model.joblib'
    if not path.exists(): return []
    production=joblib.load(path)
    selection=json.loads((HOME/'study/selection_before_audit.json').read_text(encoding='utf-8'))
    candidates=[('hgb_none',production['bundle'])]
    for name in selection['top3']:
        if name!='hgb_none': candidates.append((name,joblib.load(HOME/'study'/(name+'.joblib'))))
    output=[]
    for name,bundle in candidates:
        tag=hashlib.sha256((name+origin.isoformat()+hashlib.sha256(path.read_bytes()).hexdigest()).encode()).hexdigest()[:24]
        destination=folder/(target.strftime('%Y-%m-%d')+'_'+name+'_'+tag+'.json')
        if destination.exists(): output.append(destination); continue
        f=build(raw,targets=times,days_ahead=2,include_labels=False)
        known=current[current.timestamp.isin(times)]
        record=dict(schema=12,issued_at=clock.isoformat(),reference_origin=origin.isoformat(),target_date=target.strftime('%Y-%m-%d'),market='land',horizon='D-2 23h',
            actual_generation_delay_minutes=(clock-origin).total_seconds()/60,model=name,model_version=hashlib.sha256((HOME/'study'/(name+'.joblib')).read_bytes()).hexdigest(),
            trained_through=production['trained_through'],weather_forecast_used=False,demand_forecast_used=False,weather_forecast_issue=None,demand_forecast_issue=None,
            publication_claim='Only checked unauthenticated public sources; universal first-publication time unknown',official_smp_known_at_issue=len(known)==24,features=FEATURES)
        if len(f)!=24:
            record.update(status='skipped',reason='Incomplete eligible first-seen<=origin price history; no future-value filling',rows=[])
        else:
            p=predict(bundle,f); probability=risk_predict(production['risk'],f)
            # All source rows used by the 168h rolling/origin and lagged profile.
            needed=raw[(raw.timestamp>=origin-pd.Timedelta(days=8))&(raw.timestamp<=origin)]
            record.update(status='created',mode='official_known_diagnostic' if len(known)==24 else 'not_observed_on_checked_sources',
                inputs_first_seen_max=needed.first_seen.max(),source_prices=needed[['timestamp','smp','first_seen','snapshot']].to_dict('records'),
                rows=[dict(timestamp=t.isoformat(),predicted_smp=float(price),spike_probability=float(prob),inputs={c:float(row[c]) for c in FEATURES}) for t,price,prob,(_,row) in zip(times,p,probability,f.iterrows())])
        write_once(destination,record); output.append(destination)
    print('V12 immutable records:',[p.name for p in output],flush=True); return output

def reconcile():
    folder=HOME/'evaluations'; folder.mkdir(parents=True,exist_ok=True)
    db=connect(); prices=latest(db,'land'); db.close(); series=prices.set_index('timestamp').smp
    summaries=[]
    for file in (HOME/'forecasts').glob('*.json'):
        r=json.loads(file.read_text(encoding='utf-8'))
        if r['status']!='created': continue
        f=pd.DataFrame(r['rows']); f.timestamp=pd.to_datetime(f.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
        f['actual_smp']=series.reindex(pd.DatetimeIndex(f.timestamp)).to_numpy(); f['previous_actual_smp']=series.reindex(pd.DatetimeIndex(f.timestamp-pd.Timedelta(hours=1))).to_numpy(); f['target_date']=r['target_date']
        complete=not f[['actual_smp','previous_actual_smp']].isna().any().any()
        result=dict(forecast=file.name,at=now(),target_date=r['target_date'],model=r['model'],available_hours=int(f.actual_smp.notna().sum()),status='evaluated' if complete else 'awaiting_actual',interpretation='known-price diagnostic' if r['official_smp_known_at_issue'] else 'prospective receipt record; universal prepublication not certified')
        if complete:
            result.update(metrics(f,f.predicted_smp)); digest=hashlib.sha256(f.actual_smp.to_csv(index=False).encode()).hexdigest()[:16]
            path=folder/(file.stem+'_'+digest+'.json')
            if not path.exists(): write_once(path,result)
            pairs=folder/(file.stem+'_'+digest+'.csv')
            if not pairs.exists(): f.to_csv(pairs,index=False)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            image=folder/(file.stem+'_'+digest+'.png')
            if not image.exists():
                fig,ax=plt.subplots(figsize=(10,3)); ax.plot(f.timestamp,f.actual_smp,label='Actual'); ax.plot(f.timestamp,f.predicted_smp,label='Saved forecast'); ax.legend(); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(image,dpi=140); plt.close(fig)
        summaries.append(result)
    (HOME/'evaluation_index.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    return summaries

if __name__=='__main__': issue(); reconcile()
