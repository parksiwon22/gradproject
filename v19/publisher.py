"""Hot path: no API, model inference, EV solve or historical full-table scan."""
import argparse,sqlite3,time,hashlib
import pandas as pd
from v17.legacy import origin
from v19.common import HOME,ROOT,next_origin,read,write,write_once,sha,now,lock

def local_known(day):
    file=ROOT/'artifacts/v11/prices.sqlite'
    if not file.exists():raise ValueError('Official price store unavailable')
    db=sqlite3.connect('file:'+file.as_posix()+'?mode=ro',uri=True,timeout=2)
    try:
        begin=pd.Timestamp(day,tz='Asia/Seoul');end=begin+pd.Timedelta(days=1)
        return db.execute('SELECT 1 FROM prices WHERE market=? AND interval_start>=? AND interval_start<? LIMIT 1',('land',begin.isoformat(),end.isoformat())).fetchone() is not None
    finally:db.close()

def check_ledger(ledger,day,clock):
    entry=ledger.get('days',{}).get(day)
    if not entry:return False,'missing_publication_check'
    if entry.get('ever_positive'):return False,'official_price_already_observed'
    for source in ['kpx_api','general_public','epsis']:
        r=entry.get('latest',{}).get(source)
        if not r:return False,'missing_source_'+source
        if r['status'] not in ['not_observed','normal_no_data'] or r.get('legacy_parser',False):return False,'unverified_publication_'+source+'_'+r['status']
        age=(clock-pd.Timestamp(r['checked_at'])).total_seconds()
        if age<0 or age>180:return False,'publication_check_not_fresh_'+source
    return True,'fresh_sampled_negative_checks_not_universal_release_proof'

def check_stage(stage,registry,bound):
    if pd.Timestamp(stage['reference_origin'])!=bound:return False,'reference_origin_mismatch'
    if pd.Timestamp(stage['prepared_at'])>=bound:return False,'preparation_after_origin'
    if pd.Timestamp(registry['locked_at'])>=bound:return False,'model_locked_after_origin'
    cutoff=pd.Timestamp(stage['input_snapshot_at'])
    if cutoff>=bound:return False,'input_snapshot_not_before_origin'
    if pd.Timestamp(stage['input_first_seen_max'])>cutoff:return False,'input_receipt_after_snapshot'
    for r in stage['source_prices']:
        if pd.Timestamp(r['first_seen'])>cutoff or pd.Timestamp(r['timestamp'])>bound:return False,'future_source_in_packet'
    return True,'prepared_before_origin'

def publish_once():
    started=pd.Timestamp.now(tz='Asia/Seoul');bound=origin(started);run=dict(started_at=started.isoformat(),reference_origin=bound.isoformat(),mode='LIVE_LOCAL_PUBLISH_CHECK',records=[],network_calls=0,model_inference_in_hot_path=False,EV_optimization_in_hot_path=False);tick=time.perf_counter()
    registry=read(HOME/'model_registry.json');pointer=read(HOME/'ready'/(bound.strftime('%Y%m%dT%H%M')+'.json'))
    if not registry or not pointer:run['reason']='No packet prepared before this actual origin; no backfill'
    else:
        file=HOME/pointer['path']
        if not file.resolve().is_relative_to((HOME/'prepared').resolve()) or sha(file)!=pointer['sha256']:raise ValueError('Prepared packet integrity mismatch')
        stage=read(file);ok,reason=check_stage(stage,registry,bound)
        if stage['registry_sha256']!=sha(HOME/'model_registry.json'):ok=False;reason='Registry changed since preparation'
        if not ok:run['reason']=reason
        else:
            for p in stage['packets']:
                day=p['target_date'];name=p['model'];entry=dict(model=name,target_date=day,horizon_days=p['horizon_days'],status='skipped');dest=HOME/'forecasts'/(day+'_D'+str(p['horizon_days'])+'_'+name+'_'+bound.strftime('%Y%m%d')+'.json')
                if dest.exists():entry.update(status='existing_immutable_record',file=dest.name)
                else:
                    clock=pd.Timestamp.now(tz='Asia/Seoul');eligible,why=check_ledger(read(HOME/'publication_ledger.json',{}),day,clock)
                    if local_known(day):eligible=False;why='official_price_already_in_local_store'
                    if origin(clock)!=bound:eligible=False;why='origin_changed'
                    meta=registry['entries'][name]
                    if pd.Timestamp(meta['trained_through']).date()>=bound.date():eligible=False;why='training_labels_not_before_origin'
                    if p['horizon_days'] not in meta['horizons'] or day!=(bound.normalize()+pd.Timedelta(days=p['horizon_days'])).strftime('%Y-%m-%d'):eligible=False;why='target_or_horizon_mismatch'
                    if sha(ROOT/meta['path'])!=p['model_sha256'] or meta['sha256']!=p['model_sha256']:eligible=False;why='model_hash_changed'
                    if not eligible:entry['reason']=why
                    else:
                        issued=pd.Timestamp.now(tz='Asia/Seoul');delay=(issued-bound).total_seconds()/60
                        r=dict(schema=19,issued_at=issued.isoformat(),reference_origin=bound.isoformat(),target_date=day,model=name,horizon_days=p['horizon_days'],comparison_group=p['comparison_group'],model_sha256=p['model_sha256'],model_registered_at=registry['locked_at'],model_registry_sha256=sha(HOME/'model_registry.json'),trained_through=p['trained_through'],actual_delay_minutes=delay,timing='on_time_within_60s' if delay<=1 else 'late',computed_at=stage['prepared_at'],input_snapshot_at=stage['input_snapshot_at'],input_first_seen_max=stage['input_first_seen_max'],source_prices=stage['source_prices'],features=p['features'],rows=p['price_rows'],decisions=p['decisions'],settlement_assumptions=p['settlement_assumptions'],decision_prices_are_proxy=p['decision_prices_are_proxy'],priority_scores=p['priority_scores'],official_known_at_issue=False,publication_evidence=read(HOME/'publication_ledger.json',{}).get('days',{}).get(day),publication_caveat='Sampled sources only; D3 not guaranteed earlier than every official route',prepared_packet_sha256=pointer['sha256'],inference_seconds=p['inference_seconds'],EV_prepare_seconds=p['EV_seconds'],operating_model_replaced=False,policy_recommendation='G2V; V2G hypothetical settlement only',uncertainty='Price and ranking shadows; no guaranteed conditional interval or savings')
                        r['mode']='SIMULATION_ONLY' if stage.get('simulation_fixture') else 'ACTUAL_PUBLISH_FROM_PREPARED_INPUTS'
                        write_once(dest,r);committed=pd.Timestamp.now(tz='Asia/Seoul');commit=dict(file=dest.name,sha256=sha(dest),committed_at=committed.isoformat(),commit_delay_seconds=(committed-bound).total_seconds(),timing='on_time_within_60s' if (committed-bound).total_seconds()<=60 else 'late');write_once(HOME/'commits'/(dest.stem+'.json'),commit);entry.update(status='created',**commit)
                run['records'].append(entry)
    completed=pd.Timestamp.now(tz='Asia/Seoul');run.update(completed_at=completed.isoformat(),local_publish_seconds=time.perf_counter()-tick);write_once(HOME/'publish_runs'/(completed.strftime('%Y%m%dT%H%M%S%f')+'.json'),run);write(HOME/'publish_status.json',run);return run

def run(warm=False):
    started=pd.Timestamp.now(tz='Asia/Seoul');attempt=started.strftime('%Y%m%dT%H%M%S%f');write_once(HOME/'worker_attempts'/(attempt+'_started.json'),dict(started_at=started.isoformat(),warm=warm,mode='REAL_WORKER_ATTEMPT_NOT_FORECAST'))
    try:
        with lock('publisher'):
            if warm:
                clock=pd.Timestamp.now(tz='Asia/Seoul');previous=origin(clock)
                if (clock-previous).total_seconds()<=300:return publish_once()
                bound=next_origin(clock);wait=(bound-clock).total_seconds()
                if wait>60:
                    r=dict(at=now(),status='skipped',reason='Outside final60s prewarm or first5min after origin; no hours-long wait or backfill');write_once(HOME/'worker_attempts'/(attempt+'_skipped.json'),r);write(HOME/'publish_status.json',r);return r
                read(HOME/'ready'/(bound.strftime('%Y%m%dT%H%M')+'.json'));read(HOME/'model_registry.json')
                while pd.Timestamp.now(tz='Asia/Seoul')<bound:time.sleep(min(.1,max(.001,(bound-pd.Timestamp.now(tz='Asia/Seoul')).total_seconds())))
            return publish_once()
    except Exception as exc:
        r=dict(at=now(),status='failed',reason=type(exc).__name__,attempt_started_at=started.isoformat(),retry='Scheduled fallback or next actual origin; preserve any committed record');write_once(HOME/'worker_attempts'/(attempt+'_failed.json'),r);write(HOME/'publish_status.json',r);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--warm',action='store_true');a=p.parse_args();print(run(a.warm))
