"""Read-only eligibility decisions and evidence derived from existing collectors."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from v19.common import ROOT,read,write,write_once,sha,now
from v17.charging import require_prices
HOME=ROOT/'artifacts/v20'


def preserve(start=False):
    manifest=HOME/'preservation_before.json'
    if start:
        if manifest.exists():return
        files=set(read(ROOT/'artifacts/v19/preservation_before.json')['files'])
        for p in (ROOT/'artifacts/v19').rglob('*'):
            if p.is_file() and (p.suffix=='.joblib' or p.name=='model_registry.json' or
                any(x in ['forecasts','decisions','interactive_decisions','commits','prepared'] for x in p.parts)):
                files.add(str(p))
        write_once(manifest,dict(at=now(),files={s:sha(s) for s in sorted(files)}));return
    entries=read(manifest)['files'];bad=[p for p,h in entries.items() if not Path(p).exists() or sha(p)!=h]
    result=dict(at=now(),checked=len(entries),changed=len(bad));write(HOME/'preservation_after.json',result)
    if bad:raise ValueError('Protected original changed')
    return result


def eligible(record,meta,clock):
    """Automatic recommendation requires verified on-time prepublication evidence."""
    if record.get('mode')=='SIMULATION_ONLY':return False
    required=['strict_on_time_eligible','model_locked_before_origin','input_receipts_eligible']
    if not all(meta.get(k) is True for k in required):return False
    if meta.get('official_positive_observed_by_issue') is not False:return False
    try:
        issued=pd.Timestamp(meta['issued_at']);origin=pd.Timestamp(meta['reference_origin'])
        return bool(origin<=issued<=clock and 0<=(issued-origin).total_seconds()<=60)
    except (ValueError,KeyError,TypeError):return False


def select_auto(official,items,start,end,clock=None):
    clock=pd.Timestamp.now(tz='Asia/Seoul') if clock is None else pd.Timestamp(clock)
    try:
        prices=require_prices(official[['timestamp','price']],start,end)
        if not np.isfinite(prices.price).all():raise ValueError('nonfinite')
        return dict(mode='official',label='공식 SMP',prices=prices,reason='연결기간 전체 공식가격 확보',records=[])
    except (ValueError,KeyError):pass
    groups={}
    for path,r,f,m in items:
        if not eligible(r,m,clock):continue
        model_hash=r.get('model_sha256',r.get('model_version'))
        if not model_hash:continue
        key=(m['reference_origin'],m['model'],model_hash)
        groups.setdefault(key,[]).append((path,r,f,m))
    candidates=[]
    for key,g in groups.items():
        f=pd.concat([x[2][['timestamp','predicted_smp']] for x in g]).rename(columns={'predicted_smp':'price'})
        if f.timestamp.duplicated().any():
            conflicts=f.groupby('timestamp').price.nunique()
            if (conflicts>1).any():continue
            f=f.drop_duplicates('timestamp')
        try:prices=require_prices(f,start,end)
        except ValueError:continue
        if not np.isfinite(prices.price).all():continue
        # Latest eligible origin, then latest issue; no selection using future scores.
        candidates.append((pd.Timestamp(key[0]),max(pd.Timestamp(x[3]['issued_at']) for x in g),key,g,prices))
    if candidates:
        _,_,key,g,prices=max(candidates,key=lambda x:(x[0],x[1],x[2][1]))
        return dict(mode='saved_ai',label='저장된 적격 AI · '+key[1],prices=prices,
                    reason='동일 기준시각·동일 모델 해시의 정시 적격 저장본. 현재 새 발급 아님',
                    records=[dict(file=x[0].name,reference_origin=x[3]['reference_origin'],issued_at=x[3]['issued_at'],sha256=sha(x[0])) for x in g])
    return dict(mode='waiting',label='수신 대기',prices=None,records=[],
                reason='연결기간 전체 공식가격 또는 정시·모델사전고정·입력시각·공개상태가 검증된 저장 AI가 없습니다. 날짜 복제·준비본 사용·공식/AI 임의 혼합을 하지 않습니다.')


def publication_audit(rows):
    output=[]
    for (day,source,market),g in __import__('itertools').groupby(sorted(rows,key=lambda r:(r['target_date'],r['source'],r['market'],pd.Timestamp(r['checked_at']))),key=lambda r:(r['target_date'],r['source'],r['market'])):
        g=list(g);positive=[r for r in g if r['status']=='official_acquired']
        first=positive[0]['checked_at'] if positive else None
        negative=[r for r in g if r['status'] in ['not_observed','normal_no_data'] and not r.get('legacy_parser',False) and (first is None or pd.Timestamp(r['checked_at'])<pd.Timestamp(first))]
        zeros=[r for r in g if r['status']=='unverified_all_zero']
        others=[r for r in rows if r['target_date']==day and r['market']==market and r['source']!=source and r['status']=='official_acquired']
        # Cross-source evidence explains discrepancy; never relax the publication gate.
        conflict=any(any(pd.Timestamp(o['checked_at'])<=pd.Timestamp(z['checked_at']) for o in others) for z in zeros)
        output.append(dict(target_date=day,source=source,market=market,latest_state=g[-1]['status'],latest_check=g[-1]['checked_at'],
            first_normal_receipt=first,last_strict_normal_nonobservation=negative[-1]['checked_at'] if negative else None,
            zero_response_count=len(zeros),zero_last_received=zeros[-1]['checked_at'] if zeros else None,
            zero_conflicts_with_already_acquired_other_source=conflict,
            zero_followed_by_numeric_receipt=bool(zeros and any(pd.Timestamp(p['checked_at'])>pd.Timestamp(z['checked_at']) for z in zeros for p in positive)),
            classification='unverified_all_zero_NOT_proof_of_unpublished' if g[-1]['status']=='unverified_all_zero' else g[-1]['status']))
    return pd.DataFrame(output)


def grid_evidence(payload,day):
    """Independent column check; no interpretation of a numeric zero as absence."""
    text=payload.decode('utf-8',errors='replace');grids=[]
    for table in re.findall(r'<table\b[^>]*>.*?</table>',text,re.S|re.I):
        rows=re.findall(r'<tr\b[^>]*>(.*?)</tr>',table,re.S|re.I)
        hourly=[]
        for row in rows:
            cells=re.findall(r'<td\b[^>]*>(.*?)</td>',row,re.S|re.I)
            if cells and re.fullmatch(r'\s*\d{1,2}h\s*',cells[0]):hourly.append(cells)
        if len(hourly)!=24:continue
        headers=re.findall(r'<th\b[^>]*>(.*?)</th>',table,re.S|re.I)
        target=pd.Timestamp(day).strftime('%m.%d');last=re.sub('<[^>]*>','',headers[-1]) if headers else ''
        cells=[re.sub('<[^>]*>','',r[-1]).strip() for r in hourly]
        numeric=all(re.fullmatch(r'-?\d+(?:\.\d+)?',c) for c in cells)
        grids.append(dict(hourly_rows=24,last_column_matches_target=target in last,
                          numeric_cells=numeric,all_zero=numeric and all(float(c)==0 for c in cells)))
    return grids


def slots(clock=None):
    """Past scheduled times are not successes without actual worker evidence."""
    clock=pd.Timestamp.now(tz='Asia/Seoul') if clock is None else pd.Timestamp(clock)
    origin=clock.normalize()+pd.Timedelta(hours=23);records=[]
    runs=[read(p) for p in (ROOT/'artifacts/v19/observer_runs').glob('*.json')]
    attempts=[read(p) for p in (ROOT/'artifacts/v19/worker_attempts').glob('*_started.json')]
    for label,offset,evidence in [('22:50 준비',-10,runs),('22:58 준비',-2,runs),('23:00 발급',0,attempts)]:
        due=origin+pd.Timedelta(minutes=offset)
        matches=[r for r in evidence if due-pd.Timedelta(seconds=45)<=pd.Timestamp(r['started_at'])<=due+pd.Timedelta(minutes=3)]
        state='not_due' if clock<due else 'worker_observed_NOT_issuance_success' if matches else 'awaiting_grace' if clock<=due+pd.Timedelta(minutes=3) else 'missing_unverified_sleep_or_failure'
        records.append(dict(slot=label,due=due.isoformat(),status=state,worker_observations=len(matches)))
    return records


def audit():
    from v18.publication import all_rows
    from v19.evaluation import evaluate
    from v19.publisher import check_ledger
    from v19.common import next_origin
    base=evaluate();clock=pd.Timestamp.now(tz='Asia/Seoul');origin=next_origin(clock)
    p=publication_audit(all_rows());p.to_csv(HOME/'publication.csv',index=False)
    grid=[]
    for r in all_rows():
        if r['source']=='general_public' and r.get('sha256') and not r.get('legacy_parser',False):
            path=ROOT/'artifacts/v18/publication/raw'/(r['sha256']+'.raw')
            if path.exists():grid.append(dict(target_date=r['target_date'],checked_at=r['checked_at'],response_sha256=r['sha256'],parser_status=r['status'],grids=grid_evidence(path.read_bytes(),r['target_date'])))
    write(HOME/'grid_audit.json',dict(at=now(),rows=grid,interpretation='All-zero numeric column independently confirmed; publication meaning remains unverified'))
    ledger=read(ROOT/'artifacts/v19/publication_ledger.json',{});ready=[]
    for h in [2,3]:
        day=(origin+pd.Timedelta(days=h)).strftime('%Y-%m-%d');ok,reason=check_ledger(ledger,day,clock)
        ready.append(dict(target_date=day,horizon=h,eligible_now=ok,reason=reason,notes='Current check is not a future23h issuance result'))
    last=read(ROOT/'artifacts/v19/observer_status.json',{})
    result=dict(at=now(),status='OBSERVATION_PENDING_REAL23H' if clock<clock.normalize()+pd.Timedelta(hours=23) else 'CHECK_REAL_WORKER_RECORDS',
        **{k:v for k,v in base.items() if k!='at'},readiness=ready,scheduled_slots=slots(clock),observer=last,
        publisher=read(ROOT/'artifacts/v19/publish_status.json',{}),new_V20_models=0,
        architecture='V19 unchanged issuer and collector reused; V20 read-only audit and EV routing',
        default_recommendation='G2V')
    write(HOME/'status.json',result);write_once(HOME/'audits'/(clock.strftime('%Y%m%dT%H%M%S%f')+'.json'),result)
    return result


if __name__=='__main__':
    HOME.mkdir(parents=True,exist_ok=True);print(audit())
