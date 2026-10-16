"""Use the existing fixed D2 baseline; never train or backfill a live forecast."""
import json
import joblib,pandas as pd,numpy as np
from v11.store import connect,latest
from v11.features import build,FEATURES
from v12.research import predict
from v12.ev import EV
from v15.policies import dispatch
from v17.charging import priority_plan
from v17.legacy import origin
from v17.forward import pack
from v18.common import HOME,ROOT,read,write,write_once,sha,now
from v18.publication import all_rows,gate

MODEL=ROOT/'artifacts/v12/forward_model.joblib'

def register():
    path=HOME/'model_registry.json'
    if not path.exists():
        bundle=joblib.load(MODEL)
        write_once(path,dict(registered_at=now(),model='V12_hgb_none_preserved',model_relative_path=str(MODEL.relative_to(ROOT)),model_sha256=sha(MODEL),trained_through=bundle['trained_through'],horizon_days=2,new_training=False,operating_model_replaced=False,late_threshold_seconds=60))
    return read(path)

def eligible(meta,bound,receipts):
    if pd.Timestamp(meta['registered_at'])>bound:return False,'model_registry_created_after_reference_origin'
    if pd.Timestamp(meta['trained_through']).date()>=bound.date():return False,'training_labels_not_strictly_before_origin'
    if len(receipts) and (pd.to_datetime(receipts,utc=True,format='mixed')>bound).any():return False,'input_first_receipt_after_reference_origin'
    return True,'model_and_receipts_before_origin'

def issue(clock=None):
    started=pd.Timestamp.now(tz='Asia/Seoul')
    if clock is not None and abs((pd.Timestamp(clock)-started).total_seconds())>5:raise ValueError('Cannot backdate live issue')
    bound=origin(started);day=(bound.normalize()+pd.Timedelta(days=2)).strftime('%Y-%m-%d');run=dict(started_at=started.isoformat(),reference_origin=bound.isoformat(),target_date=day,status='skipped',mode='awaiting_eligibility')
    meta=read(HOME/'model_registry.json');events=all_rows();db=connect();current=latest(db,'land');past=latest(db,'land',as_of=bound.tz_convert('UTC').isoformat());db.close()
    observed=current.timestamp.dt.strftime('%Y-%m-%d').eq(day).any() or any(r['target_date']==day and r['market']=='land' and r['status']=='official_acquired' for r in events)
    if observed:run.update(mode='official_price_optimization',reason='Official price already observed; no new AI prior forecast')
    elif not meta:run['reason']='No frozen model registry'
    elif sha(MODEL)!=meta['model_sha256']:run['reason']='Registered model hash changed; refuse implicit replacement'
    else:
        past=past[past.timestamp<=bound].copy();ok,reason=eligible(meta,bound,past.first_seen);public_ok,public_reason=gate(events,day,started);dest=HOME/'forecasts'/(day+'_'+meta['model_sha256'][:12]+'.json')
        if dest.exists():run.update(status='existing_immutable_record',file=dest.name)
        elif not ok:run['reason']=reason
        elif not public_ok:run['reason']=public_reason
        else:
            times=pd.date_range(pd.Timestamp(day,tz='Asia/Seoul'),periods=24,freq='h');f=build(past,targets=times,days_ahead=2,include_labels=False)
            if len(f)!=24:run['reason']='Incomplete eligible price features; no filling with future data'
            else:
                model=joblib.load(MODEL);p=predict(model['bundle'],f);ev=EV(degradation=5);g=priority_plan(p,times,ev=ev);v=dispatch('general',p,times,None,ev);completed=pd.Timestamp.now(tz='Asia/Seoul')
                db=connect();check=latest(db,'land');db.close();public_ok,public_reason=gate(all_rows(),day,completed)
                if origin(completed)!=bound:run['reason']='Origin changed during generation'
                elif not public_ok or check.timestamp.dt.strftime('%Y-%m-%d').eq(day).any():run['reason']='Publication state changed during generation; no AI issuance'
                else:
                    needed=past[past.timestamp>=bound-pd.Timedelta(days=8)];delay=(completed-bound).total_seconds()/60
                    r=dict(schema=18,issued_at=completed.isoformat(),generation_started_at=started.isoformat(),reference_origin=bound.isoformat(),target_date=day,model=meta['model'],model_sha256=meta['model_sha256'],model_registered_at=meta['registered_at'],trained_through=meta['trained_through'],actual_delay_minutes=delay,timing='on_time_within_60s' if delay<=1 else 'late',official_known_at_issue=False,input_first_seen_max=needed.first_seen.max(),source_prices=needed[['timestamp','smp','first_seen','snapshot']].to_dict('records'),features=FEATURES,rows=[dict(timestamp=t.isoformat(),predicted_smp=float(value)) for t,value in zip(times,p)],settlement_assumptions=ev.__dict__,decisions=dict(G2V=pack(g),V2G=pack(v)),publication_evidence=[r for r in all_rows() if r['target_date']==day],uncertainty='Preserved price baseline; no new calibrated interval. Universal prepublication is not certified from sampled public checks.',weather_forecast_used=False,demand_forecast_used=False,operating_model_replaced=False)
                    write_once(dest,r);run.update(status='created',mode='AI_prediction_before_observed_publication',file=dest.name,issued_at=completed.isoformat(),delay_minutes=delay)
    run['completed_at']=now();write_once(HOME/'forward_runs'/(pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%f')+'.json'),run);write(HOME/'forward_status.json',run);return run

if __name__=='__main__':register();print(issue())
