"""Bounded resumable ASOS addition to the existing hourly scheduler."""
from concurrent.futures import ThreadPoolExecutor
import json
import pandas as pd
from v11.authenticated_data import key
from v11.kma_apihub import FOLDER
from v11.kma_hourly import collect_one,export
from v11.store import now
from v11.kma_archive import month,export as export_period

def main():
    if not key('KMA_APIHUB_AUTH_KEY'): return
    end=pd.Timestamp.now(tz='Asia/Seoul').floor('h')-pd.Timedelta(hours=2)
    if (FOLDER/'observations_period_hourly.csv').exists():
        # Recheck a 72-hour correction window instead of repeatedly fetching
        # an entire month on a personal laptop.
        start=end.normalize()-pd.Timedelta(days=2)
        stamp,result=month(start,end,refresh=True)
        if isinstance(result,dict):
            (FOLDER/'refresh_status.json').write_text(json.dumps(dict(at=now(),failed=1,period=stamp,result=result)),encoding='utf-8')
            raise SystemExit(1)
        export_period()
        (FOLDER/'refresh_status.json').write_text(json.dumps(dict(at=now(),failed=0,period=stamp,forecast_api='authorization still denied; no automatic promotion')),encoding='utf-8')
        return
    recent=list(pd.date_range(end-pd.Timedelta(hours=3),end,freq='h'))
    results=[]
    # Recent observations are refreshed so corrections append immutable raw
    # versions. Historical backfill never overwrites an acquired source body.
    for t in recent: results.append(collect_one(t,refresh=True))
    pending=[]
    for t in pd.date_range(pd.Timestamp('2023-01-01',tz='Asia/Seoul'),end,freq='h'):
        if not (FOLDER/'hourly_raw'/(t.strftime('%Y%m%d%H%M')+'.json')).exists(): pending.append(t)
        if len(pending)==48: break
    with ThreadPoolExecutor(max_workers=2) as pool: results.extend(pool.map(collect_one,pending))
    export()
    status=dict(at=now(),attempts=len(results),failed=sum(isinstance(r,dict) for _,r in results),forecast_api='not polled automatically while service authorization is denied',historical_backfill_complete=not pending)
    FOLDER.mkdir(parents=True,exist_ok=True)
    (FOLDER/'refresh_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    if status['failed']: raise SystemExit(1)

if __name__=='__main__': main()
