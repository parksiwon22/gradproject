"""Refresh an internal live clock after I/O; keep legacy source/model bytes intact."""
import importlib.util,json
import pandas as pd
from v11.store import ROOT,connect,latest

def origin(clock):
    value=clock.normalize()+pd.Timedelta(hours=23)
    return value if clock>=value else value-pd.Timedelta(days=1)

def fresh_continuation(initial,current):
    if initial.tzinfo is None or current.tzinfo is None or current<initial:raise ValueError('Invalid live clock progression')
    if origin(initial)!=origin(current):raise ValueError('Reference origin crossed during I/O; retry with actual clock')
    return current

def run():
    # Isolated import: no monkey patch to v15.forward, no source modification.
    spec=importlib.util.spec_from_file_location('_v17_v15_live_adapter',ROOT/'v15/forward.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);original=module._issue
    def continuation(initial):
        current=fresh_continuation(initial,pd.Timestamp.now(tz='Asia/Seoul'));target=(origin(current).normalize()+pd.Timedelta(days=2)).strftime('%Y-%m-%d')
        db=connect();prices=latest(db,'land');db.close()
        if not prices.timestamp.dt.strftime('%Y-%m-%d').eq(target).any():
            rows=[]
            for p in (ROOT/'artifacts/v12/publication').glob('*.json'):
                if p.name=='summary.json':continue
                value=json.loads(p.read_text())
                if isinstance(value,list):rows.extend(r for r in value if all(k in r for k in ['checked_at','source','status','target_date','market']))
            ok,reason=module.publication_eligible(rows,target,current)
            if not ok:raise ValueError('Publication state no longer eligible: '+reason)
        # Fresh timestamp is actual invocation time, never substituted historical23h.
        return original(pd.Timestamp.now(tz='Asia/Seoul'))
    module._issue=continuation
    result=module.issue();module.reconcile();return result

if __name__=='__main__':run()
