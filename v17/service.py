"""V15-compatible scheduler migration plus independent V17 shadow issuance."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from contextlib import contextmanager
import json
import pandas as pd
from v11.store import now
from v12.records import write_once
from v17.common import HOME,write

@contextmanager
def lock():
    HOME.mkdir(parents=True,exist_ok=True);handle=(HOME/'service.lock').open('a+b');handle.seek(0)
    if handle.read(1)==b'':handle.write(b'0');handle.flush()
    handle.seek(0)
    import msvcrt
    try:msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:handle.close();raise RuntimeError('V17 overlapping run rejected') from None
    try:yield
    finally:handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1);handle.close()

def run():
    started=now();stages=[];source_safe=False
    with lock():
        from v17.legacy import monitor
        try:
            from v17.sources import run as source_capture
            source_capture(publish=False)
            from v17.source_prices import run as cached_prices
            cached_prices();source_safe=True;stages.append(dict(stage='cached_official_API_price_receipts',status='success'))
        except Exception as exc:stages.append(dict(stage='cached_official_API_price_receipts',status='failed',reason=type(exc).__name__))
        try:monitor(allow_issue=source_safe);stages.append(dict(stage='legacy_V15_issue_reconcile_status',status='success'))
        except Exception as exc:stages.append(dict(stage='legacy_V15_issue_reconcile_status',status='failed',reason=type(exc).__name__+': '+str(exc)))
        if (HOME/'study/selection_before_audit.json').exists() and (HOME.parent.parent/'v17/forward.py').exists():
            try:
                from v17.forward import issue,reconcile,status
                if source_safe:issue()
                else:stages.append(dict(stage='V17_issue',status='skipped',reason='Official API receipt validation failed; evaluate existing records only'))
                reconcile();status();stages.append(dict(stage='V17_shadow',status='success'))
            except Exception as exc:stages.append(dict(stage='V17_shadow',status='failed',reason=type(exc).__name__+': '+str(exc)))
        else:stages.append(dict(stage='V17_shadow',status='skipped',reason='Frozen V17 study not ready; no fabricated forecast'))
        previous=sorted((HOME/'service_runs').glob('*.json'));gap=None
        if previous:
            last=json.loads(previous[-1].read_text(encoding='utf-8'));minutes=(pd.Timestamp(started)-pd.Timestamp(last['started_at'])).total_seconds()/60
            if minutes>30:gap=dict(minutes=minutes,status='missing_observation',cause='Unknown; sleep/power/failure not inferred')
        record=dict(started_at=started,completed_at=now(),stages=stages,gap=gap,no_duplicate_bulk_collection=True,publication_checks='Reuse fresh V12 observations; bounded missing/stale D3 checks only',operating_model_changed=False)
        write_once(HOME/'service_runs'/(pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%f')+'.json'),record);write(HOME/'service_status.json',record)
        print(json.dumps(record))
        if any(s['status']=='failed' for s in stages):raise SystemExit(1)

if __name__=='__main__':run()
