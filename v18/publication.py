"""Separate receipt evidence per public source; no inferred universal release clock."""
import argparse,hashlib,json,re,urllib.request,time
from pathlib import Path
import numpy as np,pandas as pd
from v11.authenticated_data import key,call,KPX
from v11.collect import request as epsis_request
from v11.public_prices import parse_public
from v12.publication import parse_eps_response
from v11.store import validate
from v18.common import HOME,ROOT,write,read,write_once,sha,now
POSITIVE='official_acquired'

def capture(payload,entry):
    digest=hashlib.sha256(payload).hexdigest();raw=HOME/'publication/raw'/(digest+'.raw');raw.parent.mkdir(parents=True,exist_ok=True)
    if not raw.exists():raw.write_bytes(payload)
    entry['sha256']=digest

def normalize_day(day):return pd.Timestamp(day).strftime('%Y-%m-%d')

def inspect_api(data,day,market='land'):
    response=data.get('response',{});code=str(response.get('header',{}).get('resultCode'))
    if code not in ['0','00']:return 'api_error',None
    items=response.get('body',{}).get('items',{});items=items.get('item',[]) if isinstance(items,dict) else items
    if not items:
        total=response.get('body',{}).get('totalCount')
        return ('normal_no_data' if str(total)=='0' else 'empty_response'),None
    if not isinstance(items,list):raise ValueError('Invalid API item container')
    area={'land':'육지','jeju':'제주'}[market];selected=[r for r in items if r.get('areaName')==area]
    if not selected:return 'normal_no_data',None
    if any(str(r.get('date'))!=pd.Timestamp(day).strftime('%Y%m%d') for r in selected):raise ValueError('API target date mismatch')
    if len(selected)!=24:return 'partial_response',None
    hours=[int(r['hour']) for r in selected]
    if sorted(hours)!=list(range(1,25)):raise ValueError('Duplicate or missing API hours')
    values=pd.to_numeric(pd.Series([r.get('smp') for r in selected]),errors='coerce')
    if values.isna().any():return 'partial_response',None
    if values.eq(0).all():return 'unverified_all_zero',None
    f=pd.DataFrame(dict(timestamp=[pd.Timestamp(day,tz='Asia/Seoul')+pd.Timedelta(hours=h-1) for h in hours],smp=values.to_numpy())).sort_values('timestamp').reset_index(drop=True);validate(f);return POSITIVE,f

def revalidate_cached_api():
    """Append a parser audit; preserve failed log and actual original receipt."""
    annotations=[]
    for file in (HOME/'publication/events').glob('*.json'):
        for r in read(file,[]):
            if r['source']!='kpx_api' or r['status']!='failed' or not r.get('sha256'):continue
            dest=HOME/'publication/events'/('revalidated_'+r['sha256'][:16]+'.json')
            if dest.exists():continue
            raw=HOME/'publication/raw'/(r['sha256']+'.raw')
            if not raw.exists():continue
            state,frame=inspect_api(read(raw),r['target_date'])
            if state==POSITIVE:annotations.append(dict(r,status=state,prices=[dict(timestamp=t.isoformat(),smp=float(p)) for t,p in zip(frame.timestamp,frame.smp)],parser_revalidated_at=now(),original_event=str(file.relative_to(ROOT)),revalidation_note='Original 48-row response split by market; checked_at is preserved actual receipt, not a new request or release time'));write_once(dest,[annotations[-1]])
    return annotations

def inspect_public(payload,day):
    if not payload:return 'empty_response',None
    text=payload.decode('utf-8')
    if f'value="{day}"' not in text:raise ValueError('Target date mismatch')
    cells=[]
    for block in re.findall(r'<tr[^>]*>(.*?)</tr>',text,re.S):
        row=re.findall(r'<td[^>]*>(.*?)</td>',block,re.S)
        if row and re.fullmatch(r'\s*\d{1,2}h\s*',row[0]):cells.append(re.sub(r'<[^>]*>','',row[-1]).strip().replace(',',''))
    if len(cells)!=24:raise ValueError('Unrecognized public hourly grid')
    if all(c in ['','-','&nbsp;'] for c in cells):return 'not_observed',None
    if any(not re.fullmatch(r'-?\d+(?:\.\d+)?',c) for c in cells):return 'partial_response',None
    if all(float(c)==0 for c in cells):return 'unverified_all_zero',None
    f=parse_public(text,day)
    if f is None:raise ValueError('Unrecognized numeric public grid')
    validate(f)
    return POSITIVE,f

def inherited():
    rows=[]
    for version in ['v12','v17']:
        for p in (ROOT/'artifacts'/version/'publication').glob('*.json'):
            if p.name=='summary.json':continue
            value=read(p)
            if isinstance(value,list):
                for r in value:
                    if all(k in r for k in ['target_date','source','checked_at','status','market']):rows.append(dict(r,evidence_file=str(p.relative_to(ROOT)),legacy_parser=r['source']=='general_public'))
    for p in (ROOT/'artifacts/v11/public_probes').glob('*.json'):
        value=read(p)
        if isinstance(value,list):
            for r in value:rows.append(dict(target_date=r['day'],source='general_public',market='land',checked_at=r['checked_at'],status=POSITIVE if r['status']=='official_available' else 'not_observed' if r['status']=='not_observed_on_this_page' else 'failed',legacy_parser=True,evidence_file=str(p.relative_to(ROOT))))
    for p in (ROOT/'artifacts/v17/sources').glob('*_demand_check.json'):
        r=read(p);raw=p.parent/(r.get('sha256','missing')+'.raw')
        if raw.exists():
            try:state,f=inspect_api(read(raw),normalize_day(r['target_date']))
            except (ValueError,KeyError,TypeError):state='invalid_response'
            rows.append(dict(target_date=normalize_day(r['target_date']),source='kpx_api',market='land',checked_at=r.get('first_seen',r['checked_at']),status=state,sha256=r.get('sha256'),evidence_file=str(p.relative_to(ROOT)),legacy_parser=False))
    return rows

def all_rows():
    rows=inherited()
    for p in (HOME/'publication/events').glob('*.json'):rows.extend(read(p,[]))
    return rows

def share_new_official_days():
    """Tell legacy collectors a new day is already known, without replaying corrections."""
    from v11.store import connect,latest,ingest
    candidates={}
    for row in all_rows():
        if row['status']==POSITIVE and row.get('prices'):candidates.setdefault((row['target_date'],row['market']),[]).append(row)
    db=connect();added=[]
    try:
        for (day,market),rows in sorted(candidates.items()):
            known=latest(db,market)
            # Existing vintages remain the existing collector's responsibility.
            if not known.empty and known.timestamp.dt.strftime('%Y-%m-%d').eq(day).any():continue
            rank={'general_public':3,'epsis':2,'kpx_api':1};r=max(rows,key=lambda x:(rank.get(x['source'],0),pd.Timestamp(x['checked_at'])))
            f=pd.DataFrame(r['prices']);f.timestamp=pd.to_datetime(f.timestamp,utc=True,format='mixed').dt.tz_convert('Asia/Seoul');payload=(HOME/'publication/raw'/(r['sha256']+'.raw')).read_bytes()
            if hashlib.sha256(payload).hexdigest()!=r['sha256']:raise ValueError('Official source hash mismatch')
            store_home=Path(db.execute('PRAGMA database_list').fetchone()[2]).parent
            ingest(db,market,f,payload,r.get('source_url',KPX),home=store_home,retrieved=r['checked_at']);added.append(dict(day=day,market=market,first_seen=r['checked_at']))
    finally:db.close()
    return added

def probe(days=None,force=False,budget_seconds=240):
    clock=pd.Timestamp.now(tz='Asia/Seoul');days=days or [(clock.normalize()+pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in [0,1,2,3]];old=all_rows();events=[];deadline=time.monotonic()+budget_seconds
    for day in days:
        for source,market in [('kpx_api','land'),('general_public','land'),('epsis','land'),('epsis','jeju')]:
            prior=[r for r in old if r['target_date']==day and r['source']==source and r['market']==market]
            recent=max(prior,key=lambda r:pd.Timestamp(r['checked_at'])) if prior else None
            if not force and recent and (pd.Timestamp.now(tz='UTC')-pd.Timestamp(recent['checked_at'])).total_seconds()<900:continue
            entry=dict(target_date=day,source=source,market=market,started_at=now(),access='public API key' if source=='kpx_api' else 'unauthenticated public',release_time_certified=False)
            if time.monotonic()>=deadline:
                entry.update(status='deferred_budget',checked_at=now(),reason='No request attempted; total probe budget exhausted; retry next tick');events.append(entry);continue
            try:
                if source=='kpx_api':
                    credential=key('KPX_SERVICE_KEY')
                    if not credential:raise PermissionError('Dedicated key unavailable')
                    data=call(KPX,dict(pageNo=1,numOfRows=100,dataType='json',date=day.replace('-','')),credential);payload=json.dumps(data).encode();capture(payload,entry);state,frame=inspect_api(data,day);url=KPX
                elif source=='general_public':
                    url='https://www.kpx.or.kr/smpInland.es?mid=a10404080100&issue_date='+day
                    payload=urllib.request.urlopen(url,timeout=20).read();capture(payload,entry);state,frame=inspect_public(payload,day)
                else:
                    url='https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpShd.ajax';payload=epsis_request(market,day,day);capture(payload,entry);frame=parse_eps_response(payload)
                    state='not_observed' if frame is None else 'unverified_all_zero' if frame.smp.eq(0).all() else POSITIVE
                    if frame is not None and set(frame.timestamp.dt.strftime('%Y-%m-%d'))!={day}:raise ValueError('EPSIS target date mismatch')
                digest=hashlib.sha256(payload).hexdigest();raw=HOME/'publication/raw'/(digest+'.raw');raw.parent.mkdir(parents=True,exist_ok=True)
                if not raw.exists():raw.write_bytes(payload)
                summary={'payload_bytes':len(payload),'parsed_status':state}
                if source=='kpx_api':
                    response=data.get('response',{});body=response.get('body',{});items=body.get('items',{})
                    items=items.get('item',[]) if isinstance(items,dict) else items
                    summary.update(api_result_code=str(response.get('header',{}).get('resultCode','missing')),
                                   api_total_count=body.get('totalCount'),api_item_count=len(items) if isinstance(items,list) else None)
                entry.update(status=state,sha256=digest,source_url=url,response_summary=summary)
                if state==POSITIVE:
                    entry['prices']=[dict(timestamp=t.isoformat(),smp=float(p)) for t,p in zip(frame.timestamp,frame.smp)]
            except Exception as exc:
                safe_detail=str(exc) if source=='kpx_api' and isinstance(exc,RuntimeError) else ''
                if not re.fullmatch(r'API HTTP \d{3}|API transport failure: [A-Za-z0-9_]+|API returned non-JSON response; content suppressed|Dedicated key unavailable',safe_detail):safe_detail=''
                entry.update(status='failed',reason=type(exc).__name__,response_error=safe_detail or 'Details suppressed')
            entry['checked_at']=now();events.append(entry)
    if events:write_once(HOME/'publication/events'/(pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%f')+'.json'),events)
    summarize();print('V18 source probes:',[(r['target_date'],r['source'],r['market'],r['status']) for r in events]);return events

def summarize(rows=None):
    rows=all_rows() if rows is None else rows;result=[]
    groups={}
    for r in rows:groups.setdefault((r['target_date'],r['source'],r['market']),[]).append(r)
    for (day,source,market),g in sorted(groups.items()):
        g=sorted(g,key=lambda r:pd.Timestamp(r['checked_at']));yes=[r for r in g if r['status']==POSITIVE];first=yes[0]['checked_at'] if yes else None
        negative=[r for r in g if r['status'] in ['not_observed','normal_no_data'] and (first is None or pd.Timestamp(r['checked_at'])<pd.Timestamp(first))]
        # Legacy blank/all-zero parser could label ambiguous responses as absent.
        strict=[r for r in negative if not r.get('legacy_parser',False)]
        result.append(dict(target_date=day,source=source,market=market,first_normal_receipt=first,last_legacy_normal_nonobservation=negative[-1]['checked_at'] if negative else None,last_strict_normal_nonobservation=strict[-1]['checked_at'] if strict else None,first_before_D2_23=pd.Timestamp(first)<=pd.Timestamp(day,tz='Asia/Seoul')-pd.Timedelta(days=2)+pd.Timedelta(hours=23) if first else None,first_before_D1_23=pd.Timestamp(first)<=pd.Timestamp(day,tz='Asia/Seoul')-pd.Timedelta(days=1)+pd.Timedelta(hours=23) if first else None,counts={s:sum(r['status']==s for r in g) for s in sorted({r['status'] for r in g})},exact_first_publication='unverified',final_corrected_statistics='unverified',membership_notice_time='not inferred from public receipts'))
    write(HOME/'publication/summary.json',result);return result

def gate(rows,day,clock):
    selected=[r for r in rows if r['target_date']==day and r['market']=='land' and pd.Timestamp(r['checked_at'])<=clock]
    if any(r['status']==POSITIVE for r in selected):return False,'official_price_observed_on_public_source'
    for source in ['kpx_api','general_public','epsis']:
        series=[r for r in selected if r['source']==source]
        if not series:return False,'missing_source_check_'+source
        r=max(series,key=lambda x:pd.Timestamp(x['checked_at']))
        if r['status'] not in ['not_observed','normal_no_data'] or r.get('legacy_parser',False):return False,'unverified_absence_'+source
        if (clock-pd.Timestamp(r['checked_at'])).total_seconds()>1200:return False,'stale_source_check_'+source
    return True,'fresh_normal_nonobservations_not_universal_release_proof'

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--force',action='store_true');a=parser.parse_args();probe(force=a.force)
