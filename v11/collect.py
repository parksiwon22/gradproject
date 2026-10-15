"""Idempotent EPSIS collector: python -m v11.collect --backfill."""
import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from v11.store import HOME,KST,connect,parse,ingest,latest,export,now

URL='https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpShd.ajax'

def request(market,begin,end):
    params=dict(beginDate=begin.replace('-',''),endDate=end.replace('-',''),selYear='',selMonth='',selKind=market,locale='')
    req=urllib.request.Request(URL,urllib.parse.urlencode(params).encode(),headers={'User-Agent':'gradproject-research/11 (local educational data collection)'})
    return urllib.request.urlopen(req,timeout=50).read()

def collect_range(db,market,begin,end,home=HOME,fetch=request,sleep=time.sleep):
    error=None
    for attempt in range(3):
        try:
            payload=fetch(market,begin,end)
            data=parse(payload.decode('utf-8'))
            dates=data.timestamp.dt.strftime('%Y-%m-%d')
            if (dates<begin).any() or (dates>end).any(): raise ValueError('server returned dates outside request')
            result=ingest(db,market,data,payload,URL,home)
            with db:
                db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?)',(now(),market,begin,end,'success',json.dumps(result)))
                db.execute('INSERT OR REPLACE INTO state VALUES (?,?)',('last_success_'+market,now()))
            print(market,begin,end,result,flush=True)
            return result
        except Exception as exc:
            # No URLs containing credentials or raw payloads in logs.
            error=f'{type(exc).__name__}: official request or integrity check failed'
            with db: db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?)',(now(),market,begin,end,'failure',error))
            if attempt<2: sleep(2**attempt)
    raise RuntimeError(error)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--backfill',action='store_true'); parser.add_argument('--market',choices=['land','jeju','both'],default='both'); args=parser.parse_args()
    db=connect(); today=datetime.now(KST).date(); end=today+timedelta(days=1)
    failures=[]
    for market in (['land','jeju'] if args.market=='both' else [args.market]):
        data=latest(db,market)
        ranges=[]
        if args.backfill or data.empty:
            for year in range(2023,today.year+1): ranges.append((f'{year}-01-01',min(end,datetime(year,12,31).date()).isoformat()))
        else:
            # Recheck trailing seven days for corrections; --backfill audits older vintages.
            ranges=[((data.timestamp.max().date()-timedelta(days=7)).isoformat(),end.isoformat())]
        for begin,finish in ranges:
            try: collect_range(db,market,begin,finish)
            except RuntimeError as exc: failures.append(dict(market=market,begin=begin,end=finish,error=str(exc)))
    summary=export(db)
    status=dict(at=now(),failures=failures,coverage=summary)
    (HOME/'collection_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8'); db.close()
    print(json.dumps(status,indent=2),flush=True)
    if failures: raise SystemExit(1)

if __name__=='__main__': main()
