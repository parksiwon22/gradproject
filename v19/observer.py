"""Slow path only: network collection, compact ledger, precomputation, evaluation."""
import os,time
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import pandas as pd
from v18.publication import probe,all_rows,share_new_official_days
from v19.common import HOME,next_origin,read,write,write_once,now,lock

def ledger():
    rows=all_rows();days={}
    for r in rows:
        if r['market']!='land':continue
        day=days.setdefault(r['target_date'],dict(ever_positive=False,latest={}))
        day['ever_positive']|=r['status']=='official_acquired';old=day['latest'].get(r['source'])
        if old is None or pd.Timestamp(r['checked_at'])>=pd.Timestamp(old['checked_at']):day['latest'][r['source']]={k:r.get(k) for k in ['source','status','checked_at','legacy_parser','sha256']}
    value=dict(at=now(),days=days);write(HOME/'publication_ledger.json',value);return value

def run():
    started=pd.Timestamp.now(tz='Asia/Seoul');stages=[]
    with lock('observer'):
        bound=next_origin(started);days=[(bound.normalize()+pd.Timedelta(days=d)).strftime('%Y-%m-%d') for d in [3,2]];preflight=started.hour==22 and started.minute>=45
        def collect():
            result=probe(days=days,force=preflight,budget_seconds=45);share_new_official_days();ledger();return dict(probes=len(result),statuses={s:sum(r['status']==s for r in result) for s in {r['status'] for r in result}})
        from v19.preparation import prepare
        from v19.evaluation import evaluate
        for name,fn in [('collection_and_publication',collect),('inference_and_EV_preparation',prepare),('post_publication_evaluation',evaluate)]:
            tick=time.perf_counter()
            try:result=fn();stages.append(dict(stage=name,status='success',seconds=time.perf_counter()-tick,result=result if name=='collection_and_publication' else None))
            except Exception as exc:stages.append(dict(stage=name,status='failed',seconds=time.perf_counter()-tick,reason=type(exc).__name__))
        previous=sorted((HOME/'observer_runs').glob('*.json'));gap=None
        if previous:
            minutes=(started-pd.Timestamp(read(previous[-1])['started_at'])).total_seconds()/60
            if minutes>30:gap=dict(minutes=minutes,status='missing_observation',cause='Unverified sleep/power/failure; no fabricated tick')
        completed=pd.Timestamp.now(tz='Asia/Seoul');r=dict(started_at=started.isoformat(),completed_at=completed.isoformat(),stages=stages,gap=gap,issues_forecasts=False);write_once(HOME/'observer_runs'/(started.strftime('%Y%m%dT%H%M%S%f')+'.json'),r);write(HOME/'observer_status.json',r);print(r)
        if any(s['status']=='failed' for s in stages):raise SystemExit(1)

if __name__=='__main__':run()
