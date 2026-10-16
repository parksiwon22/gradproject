"""Read prior immutable forecasts; write V16-only receipt/evaluation summaries."""
import json
import pandas as pd
from v11.store import now
from v12.records import write_once
from v16.common import HOME,write

def run():
    folder=HOME/'monitor_runs';folder.mkdir(parents=True,exist_ok=True);clock=pd.Timestamp.now(tz='UTC');previous=sorted(folder.glob('*.json'));gap=None
    if previous:
        last=json.loads(previous[-1].read_text(encoding='utf-8'));minutes=(clock-pd.Timestamp(last['at'])).total_seconds()/60
        if minutes>30:gap=dict(start=last['at'],end=clock.isoformat(),minutes=minutes,status='missing_observation',cause='Unknown; no synthetic run or claim of sleep/power state')
    inherited=HOME.parent/'v15/prospective_status.json';prior=json.loads(inherited.read_text()) if inherited.exists() else {}
    records=[];evaluations=[]
    for version in ['v12','v14','v15']:
        base=HOME.parent/version
        for p in (base/'forecasts').glob('*.json'):
            r=json.loads(p.read_text());records.append(dict(version=version,target_date=r['target_date'],official_known_at_issue=r.get('official_smp_known_at_issue'),issued_at=r.get('issued_at'),model=r.get('model')))
        path=base/'evaluation_index.json'
        if path.exists():evaluations.extend(dict(version=version,**r) for r in json.loads(path.read_text()))
    targets={r['target_date'] for r in records};done={r['target_date'] for r in evaluations if r['status']=='evaluated'};spike={r['target_date'] for r in evaluations if r['status']=='evaluated' and r.get('peak_hours',0)>0}
    summary=dict(at=now(),independent_target_days=len(targets),official_price_comparison_complete_days=len(done),spike_days=len(spike),waiting_days=len(targets-done),target_dates=sorted(targets),evaluated_dates=sorted(done),elapsed_evaluated_days=sum(pd.Timestamp(d,tz='Asia/Seoul')+pd.Timedelta(days=1)<=clock for d in done),V16_new_predictions=0,V16_reason='No certified historical exogenous operating model; inherited V15 task handles qualified shadow issuance',default_recommendation='G2V',operating_model_replaced=False,V15_status_at=prior.get('at'),inherited_gaps=prior.get('inherited_collection_gaps',[]),new_gap=gap)
    write(HOME/'prospective_status.json',summary);write(HOME/'inherited_evaluations.json',evaluations)
    write_once(folder/(clock.strftime('%Y%m%dT%H%M%S%f')+'.json'),dict(at=clock.isoformat(),status='observed',gap=gap));print(json.dumps({k:v for k,v in summary.items() if k!='inherited_gaps'}))

if __name__=='__main__':run()
