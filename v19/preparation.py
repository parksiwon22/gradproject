"""All network/history/model inference/EV solving happens BEFORE publication."""
import time,joblib,numpy as np,pandas as pd
from v11.store import connect,latest
from v12.ev import EV
from v15.policies import dispatch
from v17.charging import priority_plan
from v17.forward import pack
from v19.models import predict_model
from v19.common import HOME,ROOT,next_origin,read,write,write_once,sha,now

def prepare():
    started=pd.Timestamp.now(tz='Asia/Seoul');bound=next_origin(started);registry=read(HOME/'model_registry.json');run=dict(at=now(),reference_origin=bound.isoformat(),mode='UNISSUED_PREPARATION_NOT_LIVE_FORECAST',records=[],new_live_forecasts=0)
    if not registry:run['reason']='Registry not frozen';return run
    if pd.Timestamp(registry['locked_at'])>started:raise ValueError('Registry timestamp after preparation')
    db=connect();past=latest(db,'land',as_of=started.tz_convert('UTC').isoformat());current=latest(db,'land');db.close();past=past[past.timestamp<=bound].copy();ledger=read(HOME/'publication_ledger.json',{});ev=EV(degradation=5);packets=[]
    for ahead in [2,3]:
        day=(bound.normalize()+pd.Timedelta(days=ahead)).strftime('%Y-%m-%d')
        if current.timestamp.dt.strftime('%Y-%m-%d').eq(day).any() or ledger.get('days',{}).get(day,{}).get('ever_positive'):
            run['records'].append(dict(target_date=day,horizon=ahead,status='official_already_known'));continue
        for name,m in registry['entries'].items():
            if ahead not in m['horizons']:continue
            try:
                if sha(ROOT/m['path'])!=m['sha256']:raise ValueError('Model hash changed')
                tick=time.perf_counter();f,p,priority,planning=predict_model(name,past,day,ahead);inference=time.perf_counter()-tick;tick=time.perf_counter();g=priority_plan(planning,f.timestamp,priority,ev);v=dispatch('general',planning,f.timestamp,None,ev);ev_seconds=time.perf_counter()-tick
                packets.append(dict(model=name,model_sha256=m['sha256'],comparison_group=m['group'],target_date=day,horizon_days=ahead,trained_through=m['trained_through'],price_rows=[dict(timestamp=t.isoformat(),predicted_smp=float(x)) for t,x in zip(f.timestamp,p)],features=f.select_dtypes(include='number').to_dict('records'),inference_seconds=inference,EV_seconds=ev_seconds,settlement_assumptions=ev.__dict__,decisions=dict(G2V=pack(g),V2G=pack(v)),priority_scores=priority.tolist() if priority is not None else None,decision_prices_are_proxy=priority is not None))
                run['records'].append(dict(model=name,horizon=ahead,target_date=day,status='prepared_not_issued',inference_seconds=inference,EV_seconds=ev_seconds))
            except (ValueError,KeyError,TypeError) as exc:run['records'].append(dict(model=name,horizon=ahead,target_date=day,status='failed',reason=type(exc).__name__+': '+str(exc)))
    completed=pd.Timestamp.now(tz='Asia/Seoul');source=past[past.timestamp>=bound-pd.Timedelta(days=10)]
    if completed>=bound:run['reason']='Preparation missed origin; never eligible for on-time issue'
    elif packets:
        stage=dict(schema=19,mode='UNISSUED_PREPARATION_NOT_LIVE_FORECAST',prepared_at=completed.isoformat(),input_snapshot_at=started.isoformat(),reference_origin=bound.isoformat(),registry_sha256=sha(HOME/'model_registry.json'),registry_locked_at=registry['locked_at'],source_prices=source[['timestamp','smp','first_seen','snapshot']].to_dict('records'),input_first_seen_max=source.first_seen.max(),packets=packets,operating_model_replaced=False)
        file=HOME/'prepared'/(bound.strftime('%Y%m%dT%H%M')+'_'+completed.strftime('%H%M%S%f')+'.json');write_once(file,stage);write(HOME/'ready'/ (bound.strftime('%Y%m%dT%H%M')+'.json'),dict(path=str(file.relative_to(HOME)),sha256=sha(file),prepared_at=completed.isoformat()));run.update(stage=file.name,stage_sha256=sha(file))
    run['completed_at']=now();write_once(HOME/'preparation_runs'/(completed.strftime('%Y%m%dT%H%M%S%f')+'.json'),run);write(HOME/'preparation_status.json',run);return run

if __name__=='__main__':print(prepare())
