"""Reuse fresh probes; inspect only missing D3 source states, once per source."""
import hashlib,json,re,urllib.request
import pandas as pd
from v11.store import connect,ingest,now
from v11.collect import request as epsis_request
from v11.public_prices import parse_public
from v12.publication import parse_eps_response,classify
from v12.records import write_once
from v15.forward import publication_eligible
from v17.common import HOME,ROOT

def observations():
    rows=[]
    for folder in [ROOT/'artifacts/v12/publication',HOME/'publication']:
        for file in folder.glob('*.json'):
            if file.name=='summary.json':continue
            value=json.loads(file.read_text())
            if isinstance(value,list):rows.extend(r for r in value if all(k in r for k in ['checked_at','target_date','source','market','status']))
    return rows

def ensure(days):
    rows=observations();new=[]
    for day in days:
        clock=pd.Timestamp.now(tz='Asia/Seoul');ok,reason=publication_eligible(rows,day,clock)
        if ok:continue
        for source in ['general_public','epsis']:
            prior=[r for r in rows if r['target_date']==day and r['market']=='land' and r['source']==source]
            if prior:
                latest=max(prior,key=lambda r:r['checked_at'])
                # A previous positive observation must never be turned into absence.
                if any(r['status']=='official_acquired' for r in prior):continue
                if (clock-pd.Timestamp(latest['checked_at'])).total_seconds()<1200:continue
            entry=dict(target_date=day,source=source,market='land')
            try:
                if source=='general_public':
                    url='https://www.kpx.or.kr/smpInland.es?mid=a10404080100&issue_date='+day
                    payload=urllib.request.urlopen(url,timeout=25).read();text=payload.decode('utf-8')
                    if not set(map(str,range(1,25))).issubset(re.findall(r'>\s*(\d{1,2})h\s*<',text)):raise ValueError('Unrecognized page is not absence')
                    frame=parse_public(text,day)
                else:url='https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpShd.ajax';payload=epsis_request('land',day,day);frame=parse_eps_response(payload)
                state=classify(frame,day);digest=hashlib.sha256(payload).hexdigest();raw=HOME/'publication'/(digest+'.raw');raw.parent.mkdir(exist_ok=True,parents=True)
                if not raw.exists():raw.write_bytes(payload)
                entry.update(status=state,sha256=digest,source_url=url)
                if state=='official_acquired':
                    db=connect();ingest(db,'land',frame,payload,url);db.close()
            except Exception as exc:entry.update(status='failed',reason=type(exc).__name__)
            entry['checked_at']=now();rows.append(entry);new.append(entry)
    if new:write_once(HOME/'publication'/(pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%f')+'.json'),new)
    return rows
