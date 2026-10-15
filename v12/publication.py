"""Public-source receipt windows, not inferred universal publication times."""
import hashlib
import json
import re
import urllib.request
import pandas as pd
from v11.store import connect,ingest,latest,now
from v11.public_prices import parse_public
from v11.collect import request as epsis_request
from v11.store import parse as parse_epsis
from v12.records import write_once
from v12.research import HOME

def parse_eps_response(payload):
    text=payload.decode('utf-8')
    if not re.search(r'gridData\s*=\s*\[\s*\]',text) or 'var c1' not in text:
        raise ValueError('unrecognized EPSIS body is a failed probe, not absence')
    if not re.search(r'gridData\.push\(',text): return None
    return parse_epsis(text)

def classify(frame,day):
    if frame is None: return 'not_observed'
    matched=frame[frame.timestamp.dt.strftime('%Y-%m-%d')==day]
    if len(matched)==24 and matched.smp.eq(0).all(): return 'unverified_all_zero'
    if matched.timestamp.duplicated().any(): return 'invalid_duplicate_hours'
    return 'official_acquired' if len(matched)==24 else 'not_observed'

def monitor():
    folder=HOME/'publication'; folder.mkdir(parents=True,exist_ok=True)
    clock=pd.Timestamp.now(tz='Asia/Seoul'); rows=[]; db=connect()
    for offset in (1,2):
        day=(clock.normalize()+pd.Timedelta(days=offset)).strftime('%Y-%m-%d')
        for source,market in [('general_public','land'),('epsis','land'),('epsis','jeju')]:
            entry=dict(target_date=day,source=source,market=market,checked_at=now(),access='unauthenticated general-public endpoint',publication_time_claim='unknown; absence on a sampled source is not universal nonpublication')
            try:
                if source=='general_public':
                    url='https://www.kpx.or.kr/smpInland.es?mid=a10404080100&issue_date='+day
                    payload=urllib.request.urlopen(url,timeout=25).read(); text=payload.decode('utf-8')
                    labels=re.findall(r'>\s*(\d{1,2})h\s*<',text)
                    if not set(map(str,range(1,25))).issubset(labels): raise ValueError('unrecognized public page is not absence')
                    frame=parse_public(text,day)
                else:
                    url='https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpShd.ajax'
                    payload=epsis_request(market,day,day)
                    frame=parse_eps_response(payload)
                status=classify(frame,day); digest=hashlib.sha256(payload).hexdigest()
                path=folder/(digest+'.raw')
                if not path.exists(): path.write_bytes(payload)
                entry.update(status=status,source_url=url,sha256=digest)
                if status=='official_acquired': ingest(db,market,frame,payload,url)
            except Exception as exc: entry.update(status='failed',reason=type(exc).__name__)
            rows.append(entry)
    db.close(); write_once(folder/(clock.strftime('%Y%m%dT%H%M%S%f')+'.json'),rows)
    summarize(); print('Public-source probes:',[(x['target_date'],x['source'],x['market'],x['status']) for x in rows],flush=True)

def summarize():
    rows=[]
    for file in (HOME/'publication').glob('*.json'):
        if file.name=='summary.json': continue
        data=json.loads(file.read_text(encoding='utf-8'))
        if isinstance(data,list): rows.extend(data)
    # Reuse original V11 public logs without rewriting them.
    from v11.store import HOME as OLD
    for file in (OLD/'public_probes').glob('*.json'):
        for row in json.loads(file.read_text(encoding='utf-8')):
            rows.append(dict(target_date=row['day'],source='general_public',market='land',checked_at=row['checked_at'],status='official_acquired' if row['status']=='official_available' else 'failed' if row['status']=='probe_failed' else 'not_observed'))
    result=[]
    if rows:
        f=pd.DataFrame(rows)
        for (day,source,market),g in f.groupby(['target_date','source','market']):
            g=g.sort_values('checked_at'); yes=g[g.status=='official_acquired']
            first=yes.checked_at.min() if len(yes) else None
            before=g[(g.status=='not_observed')&(g.checked_at<first)] if first else g[g.status=='not_observed']
            success=g[g.status.isin(['official_acquired','not_observed'])]
            gaps=pd.to_datetime(success.checked_at,utc=True).sort_values().diff().dt.total_seconds()/60
            result.append(dict(target_date=day,source=source,market=market,first_observed_available=first,last_successful_absence_before_first=before.checked_at.max() if len(before) else None,successful_observation_count=len(success),longest_successful_observation_gap_minutes=float(gaps.max()) if gaps.notna().any() else None,exact_first_publication_unknown=True,method='sampled local receipt; interval bound applies only to queried source and assumes no cache/reversion; collection can have gaps'))
    (HOME/'publication/summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__': monitor()
