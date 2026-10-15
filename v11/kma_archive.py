"""Month-sized authorized ASOS backfill, checkpointed private source versions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import time
import pandas as pd
from v11.kma_apihub import FOLDER,request
from v11.kma_hourly import STATIONS,FIELDS,parse,atomic_text,atomic_csv
from v11.authenticated_data import key
from v11.store import now

PATH='typ01/url/kma_sfctm3.php'

def month(start,end,refresh=False):
    folder=FOLDER/'period_raw'; folder.mkdir(parents=True,exist_ok=True)
    stamp=start.strftime('%Y%m%d%H%M')+'_'+end.strftime('%Y%m%d%H%M')
    marker=folder/(stamp+'.json')
    if marker.exists() and not refresh: return stamp,'cached'
    params=dict(tm1=start.strftime('%Y%m%d%H%M'),tm2=end.strftime('%Y%m%d%H%M'),stn=':'.join(map(str,STATIONS)),help=0)
    result={}
    for attempt in range(3):
        payload,result=request(PATH,params,key('KMA_APIHUB_AUTH_KEY'))
        if payload is not None:
            f=parse(payload)
            if f.empty: result={'reason':'no_valid_rows'}; break
            if not f.timestamp.between(start,end).all() or f.duplicated(['timestamp','station_id']).any(): raise ValueError('invalid period response')
            digest=hashlib.sha256(payload).hexdigest(); raw=folder/(stamp+'_'+digest+'.raw'); raw.write_bytes(payload)
            old=json.loads(marker.read_text(encoding='utf-8')) if marker.exists() else {}
            m=dict(source=PATH,params=params,sha256=digest,raw=raw.name,first_seen=old['first_seen'] if old.get('sha256')==digest else now(),last_seen=now(),rows=len(f),kind='observation_not_forecast')
            revision=folder/(stamp+'_'+digest+'.manifest.json')
            if not revision.exists(): atomic_text(revision,json.dumps(m))
            atomic_text(marker,json.dumps(m)); return stamp,'success'
        if result.get('http_status') in (401,403): break
        time.sleep(1+attempt)
    return stamp,result

def export():
    frames=[]
    for marker in sorted((FOLDER/'period_raw').glob('????????????_????????????.json')):
        m=json.loads(marker.read_text(encoding='utf-8')); payload=(marker.parent/m['raw']).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=m['sha256']: raise ValueError('ASOS period hash mismatch')
        f=parse(payload); f['first_seen']=m['first_seen']; f['source_sha256']=m['sha256']; frames.append(f)
    if not frames: return
    f=pd.concat(frames,ignore_index=True).sort_values(['first_seen','timestamp','station_id']).drop_duplicates(['timestamp','station_id'],keep='last').sort_values(['timestamp','station_id'])
    atomic_csv(f,FOLDER/'observations_period_hourly.csv')
    # Preserve instant-query exports separately; downstream research prefers the
    # period dataset once present, so simultaneous instant jobs cannot truncate it.
    inventory=[]
    for (year,stn),g in f.groupby([f.timestamp.dt.year,'station_id']):
        atomic_csv(g,FOLDER/f'period_observations_{year}_{stn}.csv')
        start=pd.Timestamp(f'{year}-01-01',tz='Asia/Seoul'); end=min(pd.Timestamp(f'{year+1}-01-01',tz='Asia/Seoul'),pd.Timestamp.now(tz='Asia/Seoul').floor('h')-pd.Timedelta(hours=1))
        expected=max(0,int((end-start)/pd.Timedelta(hours=1)))
        inventory.append(dict(year=int(year),station_id=int(stn),station=STATIONS[stn],rows=len(g),expected_requested_hours=expected,absent_hours=max(0,expected-len(g)),first=str(g.timestamp.min()),last=str(g.timestamp.max()),missing_fraction={c:float(g[c].isna().mean()) for c in FIELDS},station_operation_period='not certified beyond returned timestamps'))
    atomic_text(FOLDER/'period_inventory.json',json.dumps(dict(at=now(),stations=inventory,time_definition='TM observation time in KST; SI MJ/m2, SS hours, RN source precipitation mm; no cumulative or interval redefinition',quality='source missing sentinels retained as NaN; temperature -9 C retained; no filling, station substitution, spatial weighting or independent QC supplied',history='observation revisions, not historical forecast vintages'),indent=2))
    print('Period export:',len(f),'station-hours',flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--export-only',action='store_true'); p.add_argument('--refresh-latest',action='store_true'); a=p.parse_args()
    if a.export_only: export(); return
    end=pd.Timestamp.now(tz='Asia/Seoul').floor('h')-pd.Timedelta(hours=2)
    starts=pd.date_range(pd.Timestamp('2023-01-01',tz='Asia/Seoul'),end,freq='MS')
    tasks=[(s,min(s+pd.offsets.MonthBegin(1)-pd.Timedelta(hours=1),end)) for s in starts]
    if a.refresh_latest: tasks=tasks[-1:]
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for stamp,result in pool.map(lambda pair:month(*pair,refresh=a.refresh_latest),tasks):
            results.append(dict(period=stamp,result=result)); print('ASOS month',stamp,result,flush=True)
    atomic_text(FOLDER/('period_attempts_'+pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S')+'.json'),json.dumps(results,indent=2))
    export()
    if any(isinstance(x['result'],dict) for x in results): raise SystemExit(1)

if __name__=='__main__': main()
