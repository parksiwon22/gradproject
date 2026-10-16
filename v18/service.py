"""Single writer for V18 issuance; locked, bounded probes and immutable run logs."""
from contextlib import contextmanager
import os,json
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import pandas as pd
from v18.common import HOME,read,write,write_once,now

@contextmanager
def lock():
    import msvcrt
    HOME.mkdir(exist_ok=True,parents=True);f=(HOME/'service.lock').open('a+b')
    try:
        if os.fstat(f.fileno()).st_size==0:f.write(b'0');f.flush()
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:f.close();raise RuntimeError('Concurrent V18 run rejected') from None
    try:yield
    finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1);f.close()

def run():
    started=pd.Timestamp.now(tz='UTC');stages=[]
    with lock():
        from v18.publication import probe
        from v18.forward import issue
        from v18.evaluation import evaluate
        from v18.legacy import run as legacy
        for name,fn in [('publication',probe),('forward',issue),('legacy_V12_evaluation',lambda:legacy('v12')),('integrated_evaluation',evaluate)]:
            try:
                value=fn()
                if name=='publication':
                    counts={s:sum(r['status']==s for r in value) for s in {r['status'] for r in value}};bad=sum(counts.get(s,0) for s in ['failed','api_error','partial_response']);stages.append(dict(stage=name,status='partial_failure' if bad else 'success',result=counts,retry='Next scheduled tick for stale failed sources; failed or empty never certifies absence'))
                else:stages.append(dict(stage=name,status='success',result=value if name=='forward' else None))
            except Exception as exc:stages.append(dict(stage=name,status='failed',reason=type(exc).__name__,retry='Next scheduled tick; preserve prior data, no synthetic success'))
        previous=sorted((HOME/'service_runs').glob('*.json'));gap=None
        if previous:
            last=read(previous[-1]);minutes=(started-pd.Timestamp(last['started_at'])).total_seconds()/60
            if minutes>30:gap=dict(minutes=minutes,status='missing_observation',cause='Unverified power/sleep/failure; no fabricated ticks')
        r=dict(started_at=started.isoformat(),completed_at=now(),stages=stages,gap=gap,automatic_training=False,automatic_model_replacement=False)
        write_once(HOME/'service_runs'/(started.strftime('%Y%m%dT%H%M%S%f')+'.json'),r);write(HOME/'service_status.json',r);print(json.dumps(r))
        if any(s['status'] in ['failed','partial_failure'] for s in stages):raise SystemExit(1)

if __name__=='__main__':run()
