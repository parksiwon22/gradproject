"""Freeze generic research dispatches, evaluate only after all actuals arrive."""
import hashlib
import json
import numpy as np
import pandas as pd
from v11.store import connect,latest,now
from v12.records import write_once
from v12.research import HOME
from v12.ev import EV,optimize,baseline,regret

def save_interactive(plans,context):
    folder=HOME/'interactive_decisions'; folder.mkdir(parents=True,exist_ok=True)
    stamp=pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%f')
    record=dict(at=now(),scenario='user-configured hypothetical SMP settlement',context=context,plans={})
    for name,plan in plans.items():
        record['plans'][name]={k:v for k,v in plan.items() if k!='schedule'}
        record['plans'][name]['schedule']=plan['schedule'].to_dict('records')
    path=folder/(stamp+'.json'); write_once(path,record); return path

def freeze():
    folder=HOME/'decisions'; folder.mkdir(parents=True,exist_ok=True)
    for file in (HOME/'forecasts').glob('*.json'):
        record=json.loads(file.read_text(encoding='utf-8'))
        if record['status']!='created' or (folder/file.name).exists(): continue
        f=pd.DataFrame(record['rows']); times=pd.to_datetime(f.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
        ev=EV(degradation=5)
        result=dict(at=now(),source_forecast=file.name,source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),scenario='generic research EV connected for entire24h, not a real personal vehicle',official_known_at_forecast_issue=record['official_smp_known_at_issue'],plans={})
        for name,v2g in [('G2V',False),('V2G',True)]:
            plan=optimize(f.predicted_smp,times,ev,v2g=v2g)
            result['plans'][name]={k:v for k,v in plan.items() if k!='schedule'}
            result['plans'][name]['schedule']=plan['schedule'].to_dict('records')
        result['immediate']=baseline(f.predicted_smp,times,ev)
        result['uniform']=baseline(f.predicted_smp,times,ev,uniform=True)
        write_once(folder/file.name,result)

def evaluate():
    folder=HOME/'decision_evaluations'; folder.mkdir(parents=True,exist_ok=True)
    db=connect(); prices=latest(db,'land').set_index('timestamp').smp; db.close(); summaries=[]
    for file in list((HOME/'decisions').glob('*.json'))+list((HOME/'interactive_decisions').glob('*.json')):
        record=json.loads(file.read_text(encoding='utf-8'))
        for name,plan in record['plans'].items():
            f=pd.DataFrame(plan['schedule']); times=pd.DatetimeIndex(pd.to_datetime(f.timestamp,utc=True).dt.tz_convert('Asia/Seoul'))
            actual=prices.reindex(times)
            result=dict(decision=file.name,cohort=file.parent.name,plan=name,status='awaiting_actual',available_hours=int(actual.notna().sum()))
            if actual.notna().all():
                ev=EV(**plan['assumptions']); loaded={**plan,'schedule':f}
                result.update(status='evaluated',**regret(loaded,actual,times,ev,v2g=name=='V2G'))
                digest=hashlib.sha256(actual.to_csv().encode()).hexdigest()[:16]; path=folder/(file.stem+'_'+name+'_'+digest+'.json')
                if not path.exists(): write_once(path,result)
            summaries.append(result)
    (HOME/'decision_evaluation_index.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')

if __name__=='__main__': freeze(); evaluate()
