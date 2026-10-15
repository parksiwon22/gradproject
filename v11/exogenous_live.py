"""Actual received exogenous inputs and separate experimental issued forecasts."""
import argparse
import hashlib
import html
import json
import re
import sqlite3
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from v11.store import HOME,KST,connect,latest,now
from v11.exogenous_collect import EXO,ELEMENTS,fetch,save
from v11.exogenous_model import read_date,attach,predict
from v11.features import build
from v11.forecast import write_once

FORECAST_URL='https://www.weather.go.kr/w/wnuri-eng/main/digital-forecast.do?code=1159068000&unit=m%2Fs&hr1=Y&lat=37.49&lon=126.92'

def parse_forecast(text,retrieved):
    match=re.search(r'Hourly Forecast\s*:\s*<span>([^<]+)',text)
    if not match: raise ValueError('official forecast issuance time absent')
    issued=pd.Timestamp(datetime.strptime(match.group(1).strip(),'%H:%M, %a, %d %b %Y'),tz='Asia/Seoul')
    short=re.search(r'Very Short-Range Forecast:</dt><dd>([^<]+)',text)
    shortissued=pd.Timestamp(datetime.strptime(short.group(1).strip(),'%H:%M, %a, %d %b %Y'),tz='Asia/Seoul') if short else issued
    rows=[]
    for classes,day,hour,body in re.findall(r'<ul class="(item[^\"]*)"[^>]*data-date="([^\"]+)"[^>]*data-time="(\d{2}:\d{2})">(.*?)</ul>',text,re.S):
        temp=re.search(r'Temperature\(Feel\).*?<span class="[^"]*feel">(-?[\d.]+)',body,re.S)
        rh=re.search(r'Humidity:.*?<span>([\d.]+)%',body,re.S)
        wind=re.search(r'<span class="wspd">([\d.]+)<',body,re.S)
        calm='title="No wind"' in body
        if not all([temp,rh]) or (wind is None and not calm): continue
        hh,mm=map(int,hour.split(':')); target=pd.Timestamp(day,tz='Asia/Seoul')+pd.Timedelta(hours=hh,minutes=mm)
        row=dict(timestamp=target,temperature_c=float(temp.group(1)),humidity_percent=float(rh.group(1)),wind_ms=float(wind.group(1)) if wind else 0.,issued_at=(shortissued if 'vs-item' in classes else issued).isoformat(),first_seen=retrieved,location='Seoul1159068000 public-grid; not identical to ASOS108')
        if pd.Timestamp(row['issued_at'])>pd.Timestamp(retrieved): raise ValueError('future issue timestamp')
        rows.append(row)
    if not rows: raise ValueError('no usable hourly KMA forecasts')
    return pd.DataFrame(rows).drop_duplicates('timestamp',keep='last').sort_values('timestamp')

def collect_forecast():
    payload=fetch(FORECAST_URL); meta=save('latest_weather_forecast.html',payload,FORECAST_URL,dict(kind='actual_forecast',station='Seoul1159068000'))
    result=parse_forecast(payload.decode('utf-8'),meta['retrieved_at'])
    folder=EXO/'weather_forecast_vintages'; folder.mkdir(exist_ok=True)
    path=folder/(meta['sha256']+'.csv')
    if not path.exists(): result.to_csv(path,index=False)
    result.to_csv(EXO/'latest_weather_forecast.csv',index=False)
    return result

def forecast_asof(bound):
    frames=[]
    for file in (EXO/'weather_forecast_vintages').glob('*.csv'):
        f=read_date(file,'timestamp')
        f=f[(pd.to_datetime(f.first_seen,utc=True)<=pd.Timestamp(bound))&(pd.to_datetime(f.issued_at,utc=True)<=pd.Timestamp(bound))]
        frames.append(f)
    if not frames: return pd.DataFrame()
    return pd.concat(frames).sort_values(['issued_at','first_seen']).drop_duplicates('timestamp',keep='last')

def ledger():
    db=sqlite3.connect(EXO/'observations.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS observations (kind TEXT,date TEXT,field TEXT,value REAL,first_seen TEXT,last_seen TEXT,revision INTEGER,PRIMARY KEY(kind,date,field,revision))')
    return db

def record_observations(stamp=None):
    db=ledger(); stamp=stamp or now()
    for kind,name in [('weather','weather_daily_all.csv'),('power','power_daily_all.csv')]:
        data=read_date(EXO/name,'date')
        with db:
            for row in data.itertuples(index=False,name=None):
                date=row[0].isoformat()
                for field,value in zip(data.columns[1:],row[1:]):
                    if not pd.notna(value): continue
                    old=db.execute('SELECT value,revision FROM observations WHERE kind=? AND date=? AND field=? ORDER BY revision DESC LIMIT 1',(kind,date,field)).fetchone()
                    if old and old[0]==float(value): db.execute('UPDATE observations SET last_seen=? WHERE kind=? AND date=? AND field=? AND revision=?',(stamp,kind,date,field,old[1]))
                    else: db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?)',(kind,date,field,float(value),stamp,stamp,old[1]+1 if old else 1))
    db.close()

def asof_observations(bound):
    db=ledger()
    raw=pd.read_sql_query('SELECT * FROM (SELECT *,ROW_NUMBER() OVER(PARTITION BY kind,date,field ORDER BY revision DESC) AS n FROM observations WHERE first_seen<=?) WHERE n=1',db,params=[bound]); db.close()
    outputs=[]
    for kind in ('weather','power'):
        f=raw[raw.kind==kind].pivot(index='date',columns='field',values='value').reset_index()
        f['date']=pd.to_datetime(f.date,utc=True).dt.tz_convert('Asia/Seoul'); outputs.append(f)
    return outputs

def issue():
    current=pd.Timestamp.now(tz='Asia/Seoul'); folder=EXO/'forecasts'; folder.mkdir(exist_ok=True)
    db=connect(); actual=latest(db,'land'); db.close()
    output=[]
    for ahead in (1,2):
        target=(current.normalize()+pd.Timedelta(days=ahead)); origin=current.normalize()+pd.Timedelta(hours=23)
        bound=min(current,origin).tz_convert('UTC').isoformat()
        db=connect(); prices=latest(db,'land',as_of=bound); db.close()
        weather,power=asof_observations(bound)
        timestamps=pd.date_range(target,periods=24,freq='h')
        f=build(prices,targets=timestamps,days_ahead=ahead,include_labels=False)
        f,w,p,_=attach(f,weather,power,days_ahead=ahead)
        path=EXO/('production_exogenous.joblib' if ahead==1 else 'production_exogenous_2day.joblib')
        bundle=joblib.load(path)
        if len(f)!=24 or f[bundle['features']].isna().any().any(): raise ValueError('complete as-of demand/weather inputs unavailable; no silent SMP fallback')
        prediction=predict(bundle,f)
        known=actual.set_index('timestamp').smp.reindex(timestamps)
        mode='official_known' if known.notna().all() else ('pre23_exogenous_preview_availability_unverified' if current.hour<23 else 'exogenous_forecast_availability_unverified')
        # Store the exact external input values, actual receipt cutoff, and observed dates.
        details=dict(issued_at=current.isoformat(),target_date=target.strftime('%Y-%m-%d'),days_ahead=ahead,reference_23h_origin=origin.isoformat(),input_available_before=bound,mode=mode,model_family=bundle['family'],model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),features=bundle['features'],exogenous_resolution='daily observed Seoul weather and national demand/supply reserve, lagged origin-day1/7; no target actual observations',historical_training_availability='unverified retrospective release timings',first_exogenous_day=(origin.normalize()-pd.Timedelta(days=1)).isoformat(),input_rows=f[['timestamp']+w+p].to_dict('records'),rows=[dict(timestamp=t.isoformat(),predicted_smp=float(v),official_smp_at_issue=float(k) if pd.notna(k) else None) for t,v,k in zip(timestamps,prediction,known)],weather_forecast_used=False,demand_forecast_used=False)
        filename=folder/(target.strftime('%Y-%m-%d')+'_'+str(ahead)+'day_'+current.strftime('%Y%m%dT%H%M%S%f')+'.json')
        write_once(filename,details); output.append(str(filename))
    (EXO/'latest_exogenous_forecast_index.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    return output

def issue_weather_proxy():
    current=pd.Timestamp.now(tz='Asia/Seoul'); origin=current.normalize()+pd.Timedelta(hours=23); bound=min(current,origin).tz_convert('UTC').isoformat()
    forecast=forecast_asof(bound)
    if forecast.empty: return []
    db=connect(); raw=latest(db,'land',as_of=bound); official=latest(db,'land'); db.close()
    folder=EXO/'weather_proxy_forecasts'; folder.mkdir(exist_ok=True)
    outputs=[]
    for ahead in (1,2):
        target=current.normalize()+pd.Timedelta(days=ahead); timestamps=pd.date_range(target,periods=24,freq='h'); weather=forecast[forecast.timestamp.isin(timestamps)]
        modelpath=EXO/f'weather_forecast_proxy_{ahead}day.joblib'
        if len(weather)!=24 or not modelpath.exists(): continue
        b=joblib.load(modelpath); f=build(raw,targets=timestamps,days_ahead=ahead,include_labels=False)
        values=dict(temperature_mean_c=float(weather.temperature_c.mean()),temperature_max_c=float(weather.temperature_c.max()),temperature_min_c=float(weather.temperature_c.min()),wind_mean_ms=float(weather.wind_ms.mean()),humidity_mean_percent=float(weather.humidity_percent.mean()))
        for name,value in values.items(): f['PROXY_'+name]=value
        pred=predict(b,f); known=official.set_index('timestamp').smp.reindex(timestamps)
        record=dict(issued_at=current.isoformat(),target_date=target.strftime('%Y-%m-%d'),days_ahead=ahead,kind='EXPERIMENTAL_FORECAST_INPUT_OBSERVED_WEATHER_TRAINING_PROXY',warning='Historical training used target-day observed weather; live uses actual published weather forecasts. No historical forecast-input accuracy validation. Not an approved replacement model.',weather_forecast_used=True,weather_input_values=values,weather_issued_at=sorted(weather.issued_at.unique()),weather_first_seen_max=weather.first_seen.max(),input_asof=bound,location_warning='ASOS108 history versus Seoul public forecast grid; spatial mismatch',target_smp_already_known=bool(known.notna().all()),features=b['features'],model_hash=hashlib.sha256(modelpath.read_bytes()).hexdigest(),rows=[dict(timestamp=t.isoformat(),predicted_smp=float(v),official_smp_at_issue=float(k) if pd.notna(k) else None) for t,v,k in zip(timestamps,pred,known)])
        path=folder/(target.strftime('%Y-%m-%d')+'_'+str(ahead)+'day_'+current.strftime('%Y%m%dT%H%M%S%f')+'.json'); write_once(path,record); outputs.append(str(path))
    return outputs

def reconcile():
    db=connect(); actual=latest(db,'land'); db.close()
    results=[]; folder=EXO/'evaluations'; folder.mkdir(exist_ok=True)
    for source in ['forecasts','weather_proxy_forecasts']:
        for file in (EXO/source).glob('*.json'):
            record=json.loads(file.read_text(encoding='utf-8')); f=pd.DataFrame(record['rows']); f.timestamp=pd.to_datetime(f.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
            f=f.merge(actual[['timestamp','smp','snapshot','first_seen']],on='timestamp',how='left')
            known_before=record.get('mode')=='official_known' or record.get('target_smp_already_known',False)
            result=dict(forecast_file=file.name,kind=record.get('kind',record.get('mode')),issued_at=record['issued_at'],target_date=record['target_date'],actual_hours=int(f.smp.notna().sum()),model_MAE=float(np.mean(abs(f.predicted_smp-f.smp))) if f.smp.notna().all() else None,known_before_issue=known_before,interpretation='Official-known diagnostic' if known_before else 'Prospective record; publication eligibility and proxy training limits remain',rows=f.replace({np.nan:None}).to_dict('records'))
            digest=hashlib.sha256(f.to_csv(index=False).encode()).hexdigest()[:12]
            path=folder/(file.stem+'_'+digest+'.json')
            if not path.exists(): write_once(path,result)
            results.append({k:v for k,v in result.items() if k!='rows'})
    (EXO/'recent_accuracy.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    return results

def main():
    p=argparse.ArgumentParser(); p.add_argument('--refresh-forecast',action='store_true'); args=p.parse_args()
    if args.refresh_forecast: collect_forecast()
    record_observations(); print(issue()); print('weather forecast proxy:',issue_weather_proxy()); reconcile()

if __name__=='__main__': main()
