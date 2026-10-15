"""Resumable hourly ASOS retrieval using the individually authorized API.

No interpolation or replacement of stations. Missing flags stay missing.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import time
import uuid
import numpy as np
import pandas as pd
from v11.kma_apihub import FOLDER, request
from v11.authenticated_data import key
from v11.store import now

STATIONS={108:'Seoul',133:'Daejeon',143:'Daegu',156:'Gwangju',159:'Busan',105:'Gangneung',184:'Jeju'}
FIELDS=['temperature_c','humidity_percent','wind_ms','solar_radiation_mj_m2','sunshine_hours','rainfall_mm']
PATH='typ01/url/kma_sfctm2.php'

def atomic_text(path,text):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(text,encoding='utf-8'); temporary.replace(path)

def atomic_csv(frame,path):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    frame.to_csv(temporary,index=False); temporary.replace(path)

def parse(payload):
    rows=[]
    for line in payload.decode('cp949',errors='replace').splitlines():
        if not re.match(r'^\d{12}\s',line): continue
        p=line.split()
        if len(p)<40: raise ValueError('truncated ASOS row')
        stn=int(p[1])
        if stn not in STATIONS: continue
        # The 22-character WW field can be blank or contain spaces. Numeric
        # fields after it must be indexed from the fixed numeric tail.
        values=[float(p[11]),float(p[13]),float(p[3]),float(p[-12]),float(p[-13]),float(p[15])]
        values=[np.nan if v in ((-99,-999) if i==0 else (-9,-99,-999)) else v for i,v in enumerate(values)]
        if any(v<0 for v in values[1:] if np.isfinite(v)): raise ValueError('invalid negative non-temperature ASOS value')
        rows.append(dict(timestamp=p[0],station_id=stn,**dict(zip(FIELDS,values))))
    frame=pd.DataFrame(rows,columns=['timestamp','station_id']+FIELDS)
    frame['timestamp']=pd.to_datetime(frame.timestamp,format='%Y%m%d%H%M').dt.tz_localize('Asia/Seoul')
    return frame

def collect_one(timestamp, refresh=False):
    credential=key('KMA_APIHUB_AUTH_KEY')
    cache=FOLDER/'hourly_raw'; cache.mkdir(parents=True,exist_ok=True)
    stamp=timestamp.strftime('%Y%m%d%H%M')
    marker=cache/(stamp+'.json')
    if marker.exists() and not refresh: return stamp,'cached'
    params=dict(tm=stamp,stn=':'.join(map(str,STATIONS)),help=0)
    result={}
    for attempt in range(3):
        payload,result=request(PATH,params,credential)
        if payload is not None:
            frame=parse(payload)
            if frame.empty: result={'reason':'no_valid_observation_rows'}; break
            if not (frame.timestamp==timestamp).all(): raise ValueError('ASOS returned wrong timestamp')
            digest=hashlib.sha256(payload).hexdigest()
            raw=cache/(stamp+'_'+digest+'.raw'); raw.write_bytes(payload)
            old=json.loads(marker.read_text(encoding='utf-8')) if marker.exists() else {}
            if old:
                old_revision=cache/(stamp+'_'+old['sha256']+'.manifest.json')
                if not old_revision.exists(): atomic_text(old_revision,json.dumps(old))
            first_seen=old['first_seen'] if old.get('sha256')==digest else now()
            manifest=dict(first_seen=first_seen,last_seen=now(),sha256=digest,raw=raw.name,params=params,station_rows=len(frame),source=PATH,kind='observation_not_forecast')
            revision=cache/(stamp+'_'+digest+'.manifest.json')
            if not revision.exists(): atomic_text(revision,json.dumps(manifest))
            atomic_text(marker,json.dumps(manifest))
            return stamp,'success'
        if result.get('http_status') in (401,403): break
        time.sleep(1+attempt)
    return stamp,result

def export():
    frames=[]
    for marker in sorted((FOLDER/'hourly_raw').glob('????????????.json')):
        m=json.loads(marker.read_text(encoding='utf-8')); raw=marker.parent/m['raw']
        if hashlib.sha256(raw.read_bytes()).hexdigest()!=m['sha256']: raise ValueError('raw hash mismatch')
        f=parse(raw.read_bytes()); f['first_seen']=m['first_seen']; f['source_sha256']=m['sha256']; frames.append(f)
    if not frames: return
    f=pd.concat(frames,ignore_index=True).sort_values(['timestamp','station_id'])
    if f.duplicated(['timestamp','station_id']).any(): raise ValueError('duplicate station hour')
    atomic_csv(f,FOLDER/'observations_hourly.csv')
    rows=[]
    for (year,stn),g in f.groupby([f.timestamp.dt.year,'station_id']):
        atomic_csv(g,FOLDER/f'observations_{year}_{stn}.csv')
        start=pd.Timestamp(f'{year}-01-01',tz='Asia/Seoul'); end=min(pd.Timestamp(f'{year+1}-01-01',tz='Asia/Seoul'),pd.Timestamp.now(tz='Asia/Seoul').floor('h'))
        expected=int((end-start)/pd.Timedelta(hours=1))
        rows.append(dict(year=int(year),station_id=int(stn),station=STATIONS[stn],rows=len(g),expected_elapsed_hours=expected,uncollected_or_absent_hours=expected-len(g),operation_period_certified=False,first=str(g.timestamp.min()),last=str(g.timestamp.max()),missing_fraction={c:float(g[c].isna().mean()) for c in FIELDS}))
    atomic_text(FOLDER/'inventory.json',json.dumps(dict(at=now(),coverage='PARTIAL until full requested range collected; absent hours are not imputed',time_definition='TM is observation instant KST; RN is source precipitation field, not RN_DAY cumulative; no undocumented interval conversion',quality='source missing sentinels masked; no independent QC flags supplied by this endpoint',stations=rows),indent=2))
    print('Exported observation rows',len(f),flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--start',default='2023-01-01'); p.add_argument('--end'); p.add_argument('--limit',type=int,default=48); p.add_argument('--workers',type=int,default=2); p.add_argument('--export-only',action='store_true'); p.add_argument('--refresh',action='store_true'); a=p.parse_args()
    if a.export_only: export(); return
    if not key('KMA_APIHUB_AUTH_KEY'): raise SystemExit('KMA_APIHUB_AUTH_KEY unavailable')
    end=pd.Timestamp(a.end,tz='Asia/Seoul') if a.end else pd.Timestamp.now(tz='Asia/Seoul').floor('h')-pd.Timedelta(hours=2)
    targets=pd.date_range(pd.Timestamp(a.start,tz='Asia/Seoul'),end,freq='h')
    pending=[t for t in targets if a.refresh or not (FOLDER/'hourly_raw'/(t.strftime('%Y%m%d%H%M')+'.json')).exists()][:a.limit]
    results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for stamp,result in pool.map(lambda t:collect_one(t,a.refresh),pending):
            results.append(dict(timestamp=stamp,result=result))
            if len(results)%24==0: print('Checkpoint',len(results),'/',len(pending),flush=True)
    FOLDER.mkdir(parents=True,exist_ok=True)
    (FOLDER/('attempts_'+pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S')+'.json')).write_text(json.dumps(results,indent=2),encoding='utf-8')
    export()

if __name__=='__main__': main()
