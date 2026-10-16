"""Actual run gaps are missing observations, never synthesized successful ticks."""
import hashlib,json
import pandas as pd
from v11.store import now
from v12.records import write_once
from v15.research import HOME,write
from v15.forward import issue,reconcile,qualified_names

def run():
    HOME.mkdir(exist_ok=True); folder=HOME/'monitor_runs'; folder.mkdir(exist_ok=True)
    previous=sorted(folder.glob('*.json')); clock=pd.Timestamp.now(tz='UTC'); gaps=[]
    if previous:
        last=json.loads(previous[-1].read_text()); minutes=(clock-pd.Timestamp(last['at'])).total_seconds()/60
        if minutes>30:
            gap=dict(start=last['at'],end=clock.isoformat(),minutes=minutes,status='missing_observation',reason='No run record; sleep, power-off and failure cannot be distinguished without system logs')
            write_once(HOME/'gaps'/(clock.strftime('%Y%m%dT%H%M%S%f')+'.json'),gap); gaps.append(gap)
    names=qualified_names(); write(HOME/'selection_for_forward.json',dict(top3=names,qualification_required=True,not_promoted=True))
    stages=[]
    for name,fn in [('candidate_issue',issue),('candidate_evaluate',reconcile)]:
        try: fn(); stages.append(dict(stage=name,status='success'))
        except Exception as exc: stages.append(dict(stage=name,status='failed',reason=type(exc).__name__+': '+str(exc)))
    rows=[]; targets=set(); waiting=set(); completed=set(); spikes=set()
    for version in ['v12','v14','v15']:
        base=HOME.parent/version
        records={p.name:json.loads(p.read_text()) for p in (base/'forecasts').glob('*.json')}
        targets.update(r['target_date'] for r in records.values())
        path=base/'evaluation_index.json'; values=json.loads(path.read_text()) if path.exists() else []
        for r in values:
            rows.append(dict(version=version,**r))
            if r['status']=='evaluated':
                completed.add(r['target_date'])
                if r.get('peak_hours',0): spikes.add(r['target_date'])
        waiting.update(r['target_date'] for r in records.values() if r['target_date'] not in completed)
    sourcegaps=HOME.parent/'v13/status.json'; inherited=json.loads(sourcegaps.read_text()).get('observed_source_gaps',[]) if sourcegaps.exists() else []
    summary=dict(at=now(),independent_target_days=len(targets),evaluated_days=len(completed),awaiting_days=len(targets-completed),evaluated_spike_days=len(spikes),evaluated_dates=sorted(completed),awaiting_dates=sorted(targets-completed),V15_candidate_records=len(list((HOME/'forecasts').glob('*.json'))),qualified_models=names,new_monitor_gaps=gaps,inherited_collection_gaps=inherited,operating_model_replaced=False)
    write(HOME/'prospective_status.json',summary); write(HOME/'combined_evaluations.json',rows)
    write_once(folder/(clock.strftime('%Y%m%dT%H%M%S%f')+'.json'),dict(at=clock.isoformat(),stages=stages)); print(json.dumps(summary))
    if any(r['status']=='failed' for r in stages): raise SystemExit(1)

if __name__=='__main__': run()
