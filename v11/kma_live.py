"""Forward validation of delayed-ASOS research; never the default EV model."""
import hashlib
import json
import joblib
import pandas as pd
from v11.store import connect,latest,now
from v11.features import build
from v11.forecast import write_once
from v11.evaluate import metrics
from v11.kma_apihub import FOLDER
from v11.kma_hourly import parse
from v11.kma_hourly_model import attach

def observations_asof(times,bound):
    frames=[]
    for marker in (FOLDER/'period_raw').glob('*.manifest.json'):
        m=json.loads(marker.read_text(encoding='utf-8'))
        if pd.Timestamp(m['first_seen'])>bound: continue
        start=pd.to_datetime(m['params']['tm1'],format='%Y%m%d%H%M').tz_localize('Asia/Seoul')
        end=pd.to_datetime(m['params']['tm2'],format='%Y%m%d%H%M').tz_localize('Asia/Seoul')
        if end<times.min() or start>times.max(): continue
        payload=(marker.parent/m['raw']).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=m['sha256']: raise ValueError('as-of raw hash mismatch')
        f=parse(payload); f=f[f.timestamp.isin(times)].copy()
        f['first_seen']=m['first_seen']; f['source_sha256']=m['sha256']; frames.append(f)
    if not frames: return pd.DataFrame()
    return pd.concat(frames).sort_values('first_seen').drop_duplicates(['timestamp','station_id'],keep='last')

def issue():
    model_path=FOLDER/'hourly_study/OBS_SMP_delayed_ASOS_NOT_FORECAST.joblib'
    if not model_path.exists(): return
    issued=pd.Timestamp.now(tz='Asia/Seoul'); origin=issued.normalize()+pd.Timedelta(hours=23)
    bound=min(issued,origin); target=origin.normalize()+pd.Timedelta(days=1)
    times=pd.date_range(target,periods=24,freq='h')
    record=dict(issued_at=issued.isoformat(),reference_origin=origin.isoformat(),input_asof=bound.isoformat(),target_date=target.strftime('%Y-%m-%d'),
                weather_forecast_used=False,weather_forecast_issued_at=None,demand_forecast_used=False,demand_forecast_issued_at=None,
                model_version=hashlib.sha256(model_path.read_bytes()).hexdigest(),kind='DELAYED_ASOS_FORWARD_RESEARCH_NOT_DEFAULT',
                not_a_weather_forecast_model=True,pre23_preview=issued<origin,late_issue=issued>origin)
    obs=observations_asof(times-pd.Timedelta(hours=48),bound)
    db=connect(); prices=latest(db,'land',as_of=bound.isoformat()); db.close()
    record['official_smp_known_at_input_cutoff']=bool(len(prices[prices.timestamp.isin(times)])==24)
    if obs.empty:
        record.update(status='skipped',reason='No ASOS source receipts existed at the D-1 23h cutoff; do not backdate current collection')
    else:
        frame=build(prices,targets=times,include_labels=False); f,columns=attach(frame,obs)
        required=[c for c in columns if any(s in c for s in ('temperature_c','humidity_percent','wind_ms'))]
        if len(f)!=24 or f[required].isna().any().any():
            record.update(status='skipped',reason='Incomplete as-of inputs; no interpolation')
        else:
            bundle=joblib.load(model_path); prediction=f.daily+bundle['model'].predict(f[bundle['features']])
            record.update(status='created',mode='official_known_diagnostic' if record['official_smp_known_at_input_cutoff'] else 'price_not_observed_in_collected_sources',
                features=bundle['features'],trained_through=bundle['trained_through'],
                source_observations=json.loads(obs.to_json(orient='records',date_format='iso')),
                rows=[dict(timestamp=t.isoformat(),predicted_smp=float(v),inputs={c:None if pd.isna(r[c]) else float(r[c]) for c in bundle['features']}) for t,v,(_,r) in zip(times,prediction,f.iterrows())])
    folder=FOLDER/'forward_records'; folder.mkdir(exist_ok=True)
    write_once(folder/(issued.strftime('%Y%m%dT%H%M%S%f')+'.json'),record)
    print('Delayed-ASOS forward record:',record['status'],flush=True)

def reconcile():
    folder=FOLDER/'forward_evaluations'; folder.mkdir(exist_ok=True)
    db=connect(); prices=latest(db,'land'); db.close(); s=prices.set_index('timestamp').smp
    for file in (FOLDER/'forward_records').glob('*.json'):
        if (folder/file.name).exists(): continue
        record=json.loads(file.read_text(encoding='utf-8'))
        if record['status']!='created': continue
        f=pd.DataFrame(record['rows']); f.timestamp=pd.to_datetime(f.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
        f['actual_smp']=s.reindex(pd.DatetimeIndex(f.timestamp)).to_numpy()
        f['previous_actual_smp']=s.reindex(pd.DatetimeIndex(f.timestamp-pd.Timedelta(hours=1))).to_numpy()
        f['target_date']=record['target_date']
        if f[['actual_smp','previous_actual_smp']].isna().any().any(): continue
        write_once(folder/file.name,dict(at=now(),original_record=file.name,mode=record['mode'],true_on_time_forecast=not record['official_smp_known_at_input_cutoff'] and not record['pre23_preview'] and not record['late_issue'],**metrics(f,f.predicted_smp)))

if __name__=='__main__': issue(); reconcile()
