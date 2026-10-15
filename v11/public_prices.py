"""Probe the general-public KPX mainland page, separately from EPSIS history."""
import hashlib
import json
import re
import urllib.request
from datetime import datetime,timedelta
import pandas as pd
from v11.store import HOME,KST,now,connect,ingest,export

def parse_public(text,day):
    # The seven displayed date columns end at issue_date; use only the last column.
    if f'value="{day}"' not in text: raise ValueError('public page date mismatch')
    rows=[]
    for block in re.findall(r'<tr[^>]*>(.*?)</tr>',text,re.S):
        cells=re.findall(r'<td[^>]*>(.*?)</td>',block,re.S)
        if cells and re.fullmatch(r'\s*(\d{1,2})h\s*',cells[0]):
            hour=int(re.search(r'(\d+)h',cells[0]).group(1))
            value=re.sub(r'<[^>]*>','',cells[-1]).strip().replace(',','')
            if not re.fullmatch(r'-?\d+(?:\.\d+)?',value): return None
            rows.append((pd.Timestamp(day,tz='Asia/Seoul')+pd.Timedelta(hours=hour-1),float(value)))
    if len(rows)!=24: return None
    frame=pd.DataFrame(rows,columns=['timestamp','smp']).sort_values('timestamp')
    # Suppress placeholder all-zero unpublished pages, rather than certify availability.
    if frame.smp.eq(0).all(): return None
    return frame

def main():
    db=connect(); current=datetime.now(KST); probes=[]
    for offset in (0,1,2):
        day=(current.date()+timedelta(days=offset)).isoformat()
        url=f'https://www.kpx.or.kr/smpInland.es?mid=a10404080100&issue_date={day}'
        try:
            payload=urllib.request.urlopen(url,timeout=40).read(); text=payload.decode('utf-8')
            frame=parse_public(text,day)
            digest=hashlib.sha256(payload).hexdigest()
            folder=HOME/'public_probes'; folder.mkdir(exist_ok=True); (folder/(digest+'.html')).write_bytes(payload)
            record=dict(day=day,checked_at=now(),url=url,payload_hash=digest,status='official_available' if frame is not None else 'not_observed_on_this_page')
            if frame is not None: record['ingestion']=ingest(db,'land',frame,payload,url)
        except Exception as exc: record=dict(day=day,checked_at=now(),url=url,status='probe_failed',error=type(exc).__name__)
        probes.append(record)
    stamp=current.strftime('%Y%m%dT%H%M%S%f')
    (HOME/'public_probes'/(stamp+'.json')).write_text(json.dumps(probes,indent=2),encoding='utf-8')
    (HOME/'public_availability.json').write_text(json.dumps(probes,indent=2),encoding='utf-8')
    export(db); db.close(); print(json.dumps(probes,indent=2))

if __name__=='__main__': main()
