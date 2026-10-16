"""Promote validated cached official API price fields with original receipt time.

No request, forecast rewrite, inferred release time, or model fitting occurs here.
API publication is not certification of final settlement or delivery-day outcome.
"""
import json
from pathlib import Path
import numpy as np,pandas as pd
from v11.authenticated_data import KPX
from v11.store import connect,ingest,validate,now
from v17.common import HOME,ROOT,write,sha

def parse_api(data,target):
    response=data.get('response',{})
    if str(response.get('header',{}).get('resultCode')) not in ['00','0']:raise ValueError('Unsuccessful official API response')
    items=response.get('body',{}).get('items',{});items=items.get('item',[]) if isinstance(items,dict) else items
    if not isinstance(items,list) or len(items)!=24:raise ValueError('Expected exactly 24 official price hours')
    if any(str(r.get('date'))!=target or r.get('areaName')!='육지' for r in items):raise ValueError('Wrong official target date or market')
    hours=[int(r['hour']) for r in items]
    if sorted(hours)!=list(range(1,25)):raise ValueError('Duplicate or missing official hours')
    prices=pd.to_numeric(pd.Series([r.get('smp') for r in items]),errors='coerce').to_numpy()
    if not np.isfinite(prices).all() or np.all(prices==0):raise ValueError('Missing or ambiguous all-zero price fields')
    frame=pd.DataFrame(dict(timestamp=[pd.Timestamp(target,tz='Asia/Seoul')+pd.Timedelta(hours=h-1) for h in hours],smp=prices)).sort_values('timestamp').reset_index(drop=True);validate(frame);return frame

def run(publish=False):
    entries=[];db=connect()
    try:
        for file in sorted((HOME/'sources').glob('*_demand_check.json')):
            manifest=json.loads(file.read_text());digest=manifest.get('sha256')
            if not digest or manifest.get('resultCode') not in ['00','0']:continue
            raw=HOME/'sources'/(digest+'.raw')
            if sha(raw)!=digest:raise ValueError('Cached API source hash mismatch')
            payload=raw.read_bytes();frame=parse_api(json.loads(payload),manifest['target_date']);receipt=pd.Timestamp(manifest['first_seen'])
            if receipt>pd.Timestamp.now(tz='UTC'):raise ValueError('Future receipt timestamp rejected')
            existing=db.execute('SELECT 1 FROM snapshots WHERE hash=?',(digest,)).fetchone()
            current=pd.read_sql_query('SELECT first_seen FROM prices WHERE market=? AND interval_start>=? AND interval_start<=?',db,params=['land',frame.timestamp.iloc[0].isoformat(),frame.timestamp.iloc[-1].isoformat()])
            newer=not current.empty and (pd.to_datetime(current.first_seen,utc=True,format='mixed')>receipt).any()
            store_home=Path(db.execute('PRAGMA database_list').fetchone()[2]).parent
            info=dict(skipped='Already ingested snapshot; never replay stale correction') if existing else dict(skipped='Newer official price vintage exists; cached older response not applied') if newer else ingest(db,'land',frame,payload,KPX,home=store_home,retrieved=receipt.isoformat())
            entries.append(dict(target_date=manifest['target_date'],source=KPX,first_received_at=receipt.isoformat(),validated_hours=len(frame),sha256=digest,ingest=info,interpretation='Official API published price at receipt; original first public release and final settlement not certified'))
    finally:db.close()
    write(HOME/'official_api_price_ingest.json',dict(at=now(),entries=entries))
    if publish:write(ROOT/'docs/V17_API_PRICE_RECEIPTS.json',entries)
    print('Validated official API price receipt days:',len(entries))

if __name__=='__main__':run(publish=True)
