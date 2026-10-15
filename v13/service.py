"""Additive V13 audit/intervals and corrected dispatch evaluation, no model promotion."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from v11.store import ROOT,connect,latest,now
from v11.features import build,FEATURES
from v12.research import HOME as V12,predict
from v12.ev import EV,regret
from v12.records import write_once
from v13.package import HOME,forecast_hashes

def read(path,default=None):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else default

def audit_record(record,prices=None):
    origin=pd.Timestamp(record['reference_origin']); issued=pd.Timestamp(record['issued_at'])
    if origin.tzinfo is None or issued.tzinfo is None: raise ValueError('timezone missing')
    if issued<origin: raise ValueError('issue precedes claimed reference cutoff')
    source=pd.DataFrame(record['source_prices']); source.timestamp=pd.to_datetime(source.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
    receipts=pd.to_datetime(source.first_seen,utc=True,format='ISO8601')
    if (receipts>origin).any() or (source.timestamp>origin).any(): raise ValueError('input received or valid after cutoff')
    rows=pd.DataFrame(record['rows']); times=pd.to_datetime(rows.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
    expected=pd.date_range(record['target_date'],periods=24,freq='h',tz='Asia/Seoul')
    if len(rows)!=24 or not np.array_equal(times.to_numpy(),expected.to_numpy()): raise ValueError('invalid target24h')
    if pd.Timestamp(record['trained_through']).date()>=expected[0].date(): raise ValueError('training crosses target')
    replay=build(source,targets=expected,days_ahead=2,include_labels=False)
    saved=pd.DataFrame(rows.inputs.tolist())[FEATURES]
    if len(replay)!=24 or not np.allclose(saved,replay[FEATURES],atol=1e-7,rtol=1e-9): raise ValueError('stored input features do not match receipt snapshot replay')
    return dict(valid=True,input_first_seen_max=receipts.max().isoformat(),reference_origin=origin.isoformat(),actual_issue_delay_minutes=(issued-origin).total_seconds()/60,feature_replay_matches=True,original_mode=record['mode'])

def audit_originals():
    preserved=read(HOME/'packages/package_results.json',{})
    before_files=sorted((HOME/'preservation').glob('*_before.json'))
    before=read(before_files[-1],{}) if before_files else {}
    after=forecast_hashes(); changed=[p for p,h in before.items() if after.get(p)!=h]
    rows=[]
    for path in (V12/'forecasts').glob('*.json'):
        record=read(path)
        try: result=audit_record(record)
        except (ValueError,KeyError,TypeError) as exc: result=dict(valid=False,reason=str(exc))
        rows.append(dict(file=path.name,**result))
    result=dict(at=now(),originals_checked=len(before),original_hashes_unchanged=not changed,changed_paths=changed,records=rows,package_oct17_count=preserved.get('original_oct17_forecasts'))
    (HOME/'original_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if changed or any(not r['valid'] for r in rows): raise ValueError('original record audit failed')
    return result

def interval_record(source_file,clock=None):
    clock=pd.Timestamp(clock) if clock is not None else pd.Timestamp.now(tz='Asia/Seoul')
    original=read(source_file); digest=hashlib.sha256(Path(source_file).read_bytes()).hexdigest()
    if clock<pd.Timestamp(original['issued_at']): raise ValueError('interval annotation cannot precede original issue')
    folder=HOME/'interval_records'; folder.mkdir(parents=True,exist_ok=True); destination=folder/(Path(source_file).stem+'_'+digest[:12]+'.json')
    if destination.exists(): return destination
    audit_record(original); origin=pd.Timestamp(original['reference_origin'])
    model_path=V12/'study'/(original['model']+'.joblib')
    if hashlib.sha256(model_path.read_bytes()).hexdigest()!=original['model_version']: raise ValueError('original model bytes unavailable; cannot calibrate a substitute')
    db=connect(); past=latest(db,'land',as_of=origin.tz_convert('UTC').isoformat()); current=latest(db,'land',as_of=clock.tz_convert('UTC').isoformat()); db.close()
    historical=build(past,days_ahead=2)
    calibration=historical[(historical.target_date>='2025-07-01')&(historical.target_date<='2025-09-30')].reset_index(drop=True)
    if len(calibration)<24*30: raise ValueError('eligible calibration data insufficient')
    model=joblib.load(model_path); errors=abs(calibration.actual_smp.to_numpy()-predict(model,calibration))
    radius=float(np.quantile(errors,.9)); rows=[]
    for row in original['rows']:
        hour=pd.Timestamp(row['timestamp']).hour; subset=errors[calibration.timestamp.dt.hour.to_numpy()==hour]
        halfwidth=float(np.quantile(subset,.9)) if len(subset)>=40 else radius
        rows.append({**row,'lower90_residual':float(row['predicted_smp']-halfwidth),'upper90_residual':float(row['predicted_smp']+halfwidth)})
    known=len(current[current.timestamp.dt.strftime('%Y-%m-%d')==original['target_date']])==24
    record=dict(schema=13,generated_at=clock.isoformat(),source_forecast=Path(source_file).name,source_sha256=digest,source_issued_at=original['issued_at'],reference_origin=original['reference_origin'],target_date=original['target_date'],model=original['model'],model_sha256=original['model_version'],independent_new_point_forecast=False,
        official_known_at_original_issue=original['official_smp_known_at_issue'],official_known_at_interval_generation=known,weather_forecast_used=False,demand_forecast_used=False,
        interval_note='Added after original point forecast; never backdated. Hourwise90% absolute residual envelope from2025Q3. Not guaranteed conditional coverage or calibrated spike probability. Historical release vintages unknown.',
        calibration=dict(start=calibration.target_date.min(),end=calibration.target_date.max(),rows=len(calibration),label_latest_timestamp=calibration.timestamp.max().isoformat(),input_receipts_cutoff=origin.isoformat(),global_radius90=radius,source_sha256=hashlib.sha256(past[['timestamp','smp','first_seen','snapshot']].to_csv(index=False).encode()).hexdigest()),rows=rows)
    write_once(destination,record); return destination

def decision_result(record,prices):
    results=[]; context=record.get('context',{})
    for name,plan in record['plans'].items():
        f=pd.DataFrame(plan['schedule']); times=pd.DatetimeIndex(pd.to_datetime(f.timestamp,utc=True).dt.tz_convert('Asia/Seoul')); actual=prices.reindex(times)
        result=dict(plan=name,available_hours=int(actual.notna().sum()),status='awaiting_actual')
        if actual.notna().all():
            result['actual_prices_sha256']=hashlib.sha256(actual.to_csv().encode()).hexdigest()
            start=context.get('start'); end=context.get('end')
            if not start and (f.connected_hours!=1).any():
                result.update(status='ineligible',reason='Partial connection without saved start/end; cannot invent oracle window')
            else:
                ev=EV(**plan['assumptions']); result.update(status='evaluated',**regret({**plan,'schedule':f},actual,times,ev,start,end,name=='V2G'),connection_start=start,connection_end=end)
        results.append(result)
    return results

def evaluate_decisions():
    folder=HOME/'decision_evaluations'; folder.mkdir(parents=True,exist_ok=True)
    db=connect(); prices=latest(db,'land').set_index('timestamp').smp; db.close(); results=[]
    for directory,base in [('decisions',V12),('interactive_decisions',V12),('v13_interactive_decisions',HOME)]:
        for path in (base/directory).glob('*.json'):
            record=read(path); digest=hashlib.sha256(path.read_bytes()).hexdigest()
            for item in decision_result(record,prices):
                result=dict(decision_file=path.name,decision_sha256=digest,cohort=directory,evaluation_version='V13_saved_connection_window',**item)
                if item['status']=='evaluated':
                    tag=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()[:16]; destination=folder/(path.stem+'_'+item['plan']+'_'+tag+'.json')
                    if not destination.exists(): write_once(destination,result)
                results.append(result)
    (HOME/'decision_index.json').write_text(json.dumps(results,indent=2),encoding='utf-8'); return results

def save_interactive(plans,context):
    folder=HOME/'v13_interactive_decisions'; folder.mkdir(parents=True,exist_ok=True)
    record=dict(at=now(),schema=13,context=context,plans={})
    for name,plan in plans.items():
        record['plans'][name]={k:v for k,v in plan.items() if k!='schedule'}
        record['plans'][name]['schedule']=plan['schedule'].to_dict('records')
    path=folder/(pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%f')+'.json')
    write_once(path,record); return path

def monitoring_summary():
    source=read(V12/'publication/summary.json',[])
    gaps=[r for r in source if (r.get('longest_successful_observation_gap_minutes') or 0)>20]
    evaluations=read(V12/'evaluation_index.json',[])
    completed={r['target_date'] for r in evaluations if r['status']=='evaluated' and r['interpretation'].startswith('prospective')}
    targets={read(p)['target_date'] for p in (V12/'forecasts').glob('*.json')}
    old=read(ROOT/'artifacts/v11/pipeline_status.json',{})
    summary=dict(at=now(),independent_target_days=len(targets),evaluated_forward_days=len(completed),point_forecast_records=len(list((V12/'forecasts').glob('*.json'))),interval_annotations=len(list((HOME/'interval_records').glob('*.json'))),annotations_are_not_additional_samples=True,observed_source_gaps=gaps,missing_run_reason='No observation record during gap; cannot distinguish sleep/power-off/task failure from these data alone',v11_failed_stages=[r for r in old.get('stages',[]) if r.get('exit_code')],v11_exogenous_note='Missing complete as-of demand/weather input: fail closed, no invented future weather. See V11 pipeline log; task nonzero is not successful full execution.',model_promotion=False,more_prospective_data_required=len(completed)<30)
    (HOME/'status.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); return summary

@contextmanager
def single_run():
    HOME.mkdir(parents=True,exist_ok=True); handle=(HOME/'.pipeline.lock').open('a+b')
    handle.seek(0,2)
    if handle.tell()==0: handle.write(b'0'); handle.flush()
    handle.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        handle.close(); raise RuntimeError('overlapping V13 run rejected') from None
    try: yield
    finally:
        if os.name=='nt':
            handle.seek(0); msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
        else: fcntl.flock(handle,fcntl.LOCK_UN)
        handle.close()

def run():
    from v12.forward import reconcile
    started=now(); folder=HOME/'runs'; folder.mkdir(parents=True,exist_ok=True)
    with single_run():
        stages=[]
        actions=[('original_audit',audit_originals),('interval_annotations',lambda:[interval_record(p) for p in (V12/'forecasts').glob('*.json')]),('point_reconcile',reconcile),('corrected_decision_evaluation',evaluate_decisions),('prospective_status',monitoring_summary)]
        for name,action in actions:
            try: action(); stages.append(dict(stage=name,status='success'))
            except Exception as exc: stages.append(dict(stage=name,status='failed',reason=type(exc).__name__+': '+str(exc)))
        record=dict(started_at=started,completed_at=now(),stages=stages,collection='V11/V12 existing tasks; V13 does not duplicate network collection',automatic_training=False,automatic_promotion=False)
        write_once(folder/(pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%f')+'.json'),record)
        (HOME/'pipeline_status.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        if any(r['status']=='failed' for r in stages): raise SystemExit(1)
    print('V13 audit/interval/evaluation pipeline completed')

if __name__=='__main__': run()
