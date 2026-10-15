"""Immutable issued forecasts, official-known mode, and later reconciliation."""
import argparse
import hashlib
import json
from datetime import datetime,timedelta
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from v11.store import HOME,KST,connect,latest,now
from v11.features import build,predict,FEATURES
from v11.evaluate import metrics

def write_once(path,record):
    with Path(path).open('x',encoding='utf-8') as handle: json.dump(record,handle,ensure_ascii=False,indent=2,default=str)

def issue(preview=False,home=HOME):
    home=Path(home); issued=datetime.now(KST); db=connect(home)
    raw=latest(db,'land',as_of=issued.astimezone(__import__('datetime').timezone.utc).isoformat()); db.close()
    if raw.empty: raise ValueError('no official history')
    folder=home/'forecasts'; folder.mkdir(exist_ok=True)
    records=[]
    public=json.loads((home/'public_availability.json').read_text(encoding='utf-8')) if (home/'public_availability.json').exists() else []
    for ahead in (1,2):
        target=(issued.date()+timedelta(days=ahead)).isoformat()
        timestamps=pd.date_range(target,periods=24,freq='h',tz='Asia/Seoul')
        known=raw.set_index('timestamp').reindex(timestamps).smp
        probe=next((p for p in public if p['day']==target),None)
        available=known.notna().all()
        origin=pd.Timestamp(target,tz='Asia/Seoul')-pd.Timedelta(days=ahead)+pd.Timedelta(hours=23)
        if not available and issued.hour<23 and not preview: continue
        modelpath=home/('production.joblib' if ahead==1 else 'production_2day.joblib')
        if not modelpath.exists(): raise ValueError('trained model missing')
        bundle=joblib.load(modelpath)
        digest=hashlib.sha256(modelpath.read_bytes()).hexdigest()
        datahash=hashlib.sha256(raw[['timestamp','smp','snapshot']].to_csv(index=False).encode()).hexdigest()
        mode='official_known' if available else ('pre23_preview_availability_unverified' if issued.hour<23 else 'forecast_availability_unverified')
        # Both general-public and EPSIS absence still cannot prove universal nonpublication.
        # At/after23 use only vintages first received by the23h origin. Preview
        # explicitly uses current receipt time and is never called a23h run.
        feature_asof=min(pd.Timestamp(issued),origin).tz_convert('UTC').isoformat()
        db=connect(home); feature_raw=latest(db,'land',as_of=feature_asof); db.close()
        frame=build(feature_raw,targets=timestamps,days_ahead=ahead,include_labels=False)
        if len(frame)!=24: raise ValueError('not enough complete input history')
        if raw.first_seen.max()>issued.astimezone(__import__('datetime').timezone.utc).isoformat(): raise ValueError('future first_seen leaked')
        p=predict(bundle,frame)
        band=bundle['interval_radius90']
        tag=f'{target}_{ahead}day_{mode}_{digest[:12]}'
        existing=list(folder.glob(tag+'_*.json'))
        if existing:
            records.append(str(existing[0])); continue
        rows=[]
        for idx,t in enumerate(timestamps):
            price=float(known.iloc[idx]) if available else float(p[idx])
            rows.append(dict(timestamp=t.isoformat(),predicted_smp=float(p[idx]),official_smp_at_issue=float(known.iloc[idx]) if available else None,decision_price=price,lower90=float(p[idx]-band),upper90=float(p[idx]+band),high_price_risk='high' if price>=150 else ('possible' if p[idx]+band>=150 else 'low')))
        record=dict(schema=1,issued_at=issued.isoformat(),target_date=target,market='land',mode=mode,days_ahead=ahead,reference_23h_origin=origin.isoformat(),exact_23h_policy_run=bool(issued.hour>=23),origin_note='Before23 preview uses official prices already received; reference event cutoff may be later than issue time; not a claimed23h run' if issued.hour<23 else '23h feature cutoff, actual issue timestamp recorded separately',availability_probe=probe,availability_claim='official price already acquired' if available else 'not acquired on checked sources; no claim of completely unpublished price',model_family=bundle['family'],model_hash=digest,model_trained_through=bundle['trained_through'],training_data_hash=bundle['data_hash'],input_data_hash=datahash,input_latest_official_interval=raw.timestamp.max().isoformat(),input_first_seen_max=raw.first_seen.max(),interval_note='symmetric 90% validation residual band; no conditional coverage guarantee, not calibrated spike probability',rows=rows)
        path=folder/(tag+'_'+issued.strftime('%Y%m%dT%H%M%S%f')+'.json')
        write_once(path,record); records.append(str(path))
    (home/'latest_forecast_index.json').write_text(json.dumps(dict(updated_at=now(),files=records),indent=2),encoding='utf-8')
    return records

def reconcile(home=HOME):
    home=Path(home); db=connect(home); raw=latest(db,'land'); db.close()
    folder=home/'evaluations'; folder.mkdir(exist_ok=True)
    results=[]
    for file in sorted((home/'forecasts').glob('*.json')):
        record=json.loads(file.read_text(encoding='utf-8'))
        f=pd.DataFrame(record['rows']); f['timestamp']=pd.to_datetime(f.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
        merged=f.merge(raw[['timestamp','smp','first_seen','snapshot']],on='timestamp',how='left')
        merged=merged.rename(columns={'smp':'actual_smp'})
        result=dict(forecast_file=file.name,evaluated_at=now(),issued_at=record['issued_at'],mode=record['mode'],available_hours=int(merged.actual_smp.notna().sum()),pairs=merged.replace({np.nan:None}).to_dict('records'))
        if merged.actual_smp.notna().all():
            result['MAE']=float(np.mean(abs(merged.predicted_smp-merged.actual_smp)))
            result['interpretation']='known-price diagnostic, not prospective forecast score' if record['mode']=='official_known' else 'record issued before target, publication eligibility remains unverified'
        vintage=hashlib.sha256(merged.to_csv(index=False).encode()).hexdigest()[:12]
        output=folder/(file.stem+'_'+vintage+'.json')
        if not output.exists(): write_once(output,result)
        results.append(dict(file=output.name,target_date=record['target_date'],mode=record['mode'],hours=result['available_hours'],MAE=result.get('MAE')))
    (home/'recent_accuracy.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    return results

def plots(home=HOME):
    home=Path(home)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    db=connect(home); actual=latest(db,'land'); db.close()
    for path in (home/'forecasts').glob('*.json'):
        r=json.loads(path.read_text(encoding='utf-8')); f=pd.DataFrame(r['rows']); f['timestamp']=pd.to_datetime(f.timestamp)
        f=f.merge(actual[['timestamp','smp']],on='timestamp',how='left')
        fig,ax=plt.subplots(figsize=(12,4)); ax.plot(f.timestamp,f.predicted_smp,label='Model estimate'); ax.fill_between(f.timestamp,f.lower90,f.upper90,alpha=.2,label='Validation residual band'); ax.plot(f.timestamp,f.smp,label='Official when available'); ax.set_title(r['target_date']+' | '+r['mode']); ax.set_ylabel('KRW/kWh'); ax.legend(); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(path.with_suffix('.png')); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4)); tail=actual.tail(7*24); ax.plot(tail.timestamp,tail.smp); ax.set_title('Latest acquired official mainland SMP'); ax.set_ylabel('KRW/kWh'); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(home/'latest_actual.png'); plt.close(fig)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--preview',action='store_true'); args=p.parse_args()
    print(issue(preview=args.preview)); print(reconcile()); plots()

if __name__=='__main__': main()
