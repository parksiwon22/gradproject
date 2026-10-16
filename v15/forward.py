"""Append-only candidate forecasts; published target prices cause an explicit skip."""
import hashlib,json
import joblib,numpy as np,pandas as pd
from v11.store import connect,latest,now
from v12.records import write_once
from v15.features import build as feature_build
from v15.research import HOME,predict
from v14.research import scored
from v12.ev import EV,optimize,regret
from v15.policies import POLICIES,dispatch,residual_templates

def qualified_names():
    names=[]
    for path in [HOME/'model_study/qualification.json',HOME/'shrinkage_followup/qualification.json']:
        if path.exists(): names.extend(r['name'] for r in json.loads(path.read_text())['results'] if r['qualified'])
    return names

def publication_eligible(rows,target,clock):
    selected=[r for r in rows if r.get('target_date')==target and r.get('market')=='land' and pd.Timestamp(r['checked_at'])<=clock]
    if any(r['status']=='official_acquired' for r in selected): return False,'official_price_observed_on_public_source'
    for source in ['general_public','epsis']:
        series=sorted([r for r in selected if r['source']==source],key=lambda x:x['checked_at'])
        if not series or series[-1]['status']!='not_observed': return False,'public_source_state_missing_or_failed'
        if (clock-pd.Timestamp(series[-1]['checked_at'])).total_seconds()>1200: return False,'public_source_check_stale'
    return True,'fresh_negative_checks_not_universal_proof'

def _issue(clock=None):
    clock=pd.Timestamp(clock) if clock is not None else pd.Timestamp.now(tz='Asia/Seoul')
    if abs((clock-pd.Timestamp.now(tz='Asia/Seoul')).total_seconds())>5: raise ValueError('Cannot fabricate a live forecast with a historical or future issue clock')
    origin=clock.normalize()+pd.Timedelta(hours=23)
    if clock<origin: origin-=pd.Timedelta(days=1)
    target=origin.normalize()+pd.Timedelta(days=2); times=pd.date_range(target,periods=24,freq='h')
    run=dict(at=now(),reference_origin=origin.isoformat(),target_date=target.strftime('%Y-%m-%d'),status='skipped',reason=None)
    run['actual_delay_minutes']=(clock-origin).total_seconds()/60
    selection=HOME/'selection_for_forward.json'
    if not selection.exists(): run['reason']='candidate_selection_not_complete'
    else:
        db=connect(); current=latest(db,'land',as_of=clock.tz_convert('UTC').isoformat()); past=latest(db,'land',as_of=origin.tz_convert('UTC').isoformat()); db.close()
        known=current[current.timestamp.isin(times)]
        if len(known)>0: run.update(reason='official_target_prices_already_acquired; use_official_mode',official_known=True,official_hours=len(known))
        else:
            past=past[past.timestamp<=origin].copy(); frame,fields_all=feature_build(past,targets=times,include_labels=False); feature_groups={'price':fields_all}
            if len(frame)!=24: run['reason']='insufficient_eligible_asof_price_history'
            else:
                names=json.loads(selection.read_text())['top3']; paths=[]
                history,_=feature_build(past)
                calibration=history[(history.target_date>='2025-07-01')&(history.target_date<='2025-09-30')].reset_index(drop=True)
                if len(calibration)<24*30: raise ValueError('As-of calibration history insufficient')
                for name in names:
                    modelpath=HOME/'candidates'/(name+'.joblib'); digest=hashlib.file_digest(modelpath.open('rb'),'sha256').hexdigest(); path=HOME/'forecasts'/(target.strftime('%Y-%m-%d')+'_'+name+'_'+digest[:12]+'.json')
                    if path.exists(): paths.append(path.name); continue
                    bundle=joblib.load(modelpath)
                    if name not in qualified_names(): raise ValueError('Model did not pass frozen V15 chronological validation')
                    p=predict(bundle,frame)
                    radius=float(np.quantile(abs(calibration.actual_smp.to_numpy()-predict(bundle,calibration)),.9))
                    fields=bundle.get('features',feature_groups['price'])
                    rows=[dict(timestamp=t.isoformat(),predicted_smp=float(value),lower90_residual=float(value-radius),upper90_residual=float(value+radius),inputs={k:float(row[k]) for k in fields}) for t,value,(_,row) in zip(times,p,frame.iterrows())]
                    snapshot=past[past.timestamp>=origin-pd.Timedelta(days=8)]
                    record=dict(schema=15,issued_at=clock.isoformat(),reference_origin=origin.isoformat(),target_date=target.strftime('%Y-%m-%d'),model=name,model_sha256=digest,trained_through='2025-06-30',official_smp_known_at_issue=False,publication_claim='not acquired on checked local source; universal prepublication not certified',mode='V15_CV_qualified_candidate_not_operating_model',weather_forecast_used=False,demand_forecast_used=False,input_first_seen_max=snapshot.first_seen.max(),source_prices=snapshot[['timestamp','smp','first_seen','snapshot']].to_dict('records'),uncertainty='Point candidate only; historical validation residuals are not calibrated future coverage. Spike warning unvalidated.',rows=rows)
                    record.update(uncertainty='Symmetric2025Q3 absolute residual90% range; no guaranteed future conditional coverage or calibrated spike probability',calibration=dict(start='2025-07-01',end='2025-09-30',rows=len(calibration),radius90=radius))
                    record.update(actual_delay_minutes=(clock-origin).total_seconds()/60,model_selection_through='2025-12-31',exact_23h_execution=abs((clock-origin).total_seconds())<60)
                    record['decisions']={}; templates=residual_templates((calibration.actual_smp.to_numpy()-predict(bundle,calibration)).reshape(-1,24))
                    record['settlement_assumptions']=EV(degradation=5).__dict__; record['policy_recommendation']='G2V; no active V2G policy passed prior validation; other plans are shadow research comparisons'
                    record['scenario_templates']=templates.tolist()
                    for label in POLICIES:
                        try: plan=dispatch(label,p,times,templates,EV(degradation=5))
                        except ValueError: plan=dispatch('G2V',p,times,templates,EV(degradation=5)); plan['fallback_reason']='Scenario optimization failed; no invented price'
                        record['decisions'][label]={**{k:v for k,v in plan.items() if k!='schedule'},'schedule':plan['schedule'].to_dict('records')}
                    write_once(path,record); paths.append(path.name)
                run.update(status='created_or_existing',reason=None,records=paths,independent_target_days=1)
    folder=HOME/'forward_runs'; folder.mkdir(parents=True,exist_ok=True)
    write_once(folder/(pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%f')+'.json'),run)
    print(json.dumps(run)); return run

def issue(clock=None):
    clock=pd.Timestamp(clock) if clock is not None else pd.Timestamp.now(tz='Asia/Seoul')
    if abs((clock-pd.Timestamp.now(tz='Asia/Seoul')).total_seconds())>5: raise ValueError('Cannot fabricate historical or future issue clock')
    origin=clock.normalize()+pd.Timedelta(hours=23)
    if clock<origin: origin-=pd.Timedelta(days=1)
    target=(origin.normalize()+pd.Timedelta(days=2)).strftime('%Y-%m-%d')
    db=connect(); current=latest(db,'land'); db.close()
    if (current.timestamp.dt.strftime('%Y-%m-%d')==target).any(): return _issue(clock)
    rows=[]
    for file in (HOME.parent/'v12/publication').glob('*.json'):
        if file.name=='summary.json': continue
        content=json.loads(file.read_text())
        if isinstance(content,list): rows.extend(content)
    eligible,reason=publication_eligible(rows,target,clock)
    if not qualified_names(): eligible=False; reason='no_model_passed_frozen_validation'
    if eligible: return _issue(clock)
    run=dict(at=now(),reference_origin=origin.isoformat(),target_date=target,status='skipped',reason=reason)
    write_once(HOME/'forward_runs'/(pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%f')+'.json'),run); print(json.dumps(run)); return run

def reconcile():
    db=connect(); prices=latest(db,'land'); db.close(); summaries=[]
    for path in (HOME/'forecasts').glob('*.json'):
        record=json.loads(path.read_text()); f=pd.DataFrame(record['rows']); f.timestamp=pd.to_datetime(f.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
        f=f.merge(prices[['timestamp','smp','first_seen']],on='timestamp',how='left'); f['target_date']=record['target_date']; f['actual_smp']=f.smp
        row=dict(model=record['model'],target_date=record['target_date'],status='awaiting_actual',available_hours=int(f.smp.notna().sum()))
        if f.smp.notna().all():
            previous=prices.set_index('timestamp').smp.reindex(f.timestamp-pd.Timedelta(hours=1)).to_numpy()
            if not np.isfinite(previous).all(): row['reason']='missing_evaluation_boundary'; summaries.append(row); continue
            f['previous_actual_smp']=previous; ev={}
            for label,plan in record.get('decisions',{}).items():
                saved={**plan,'schedule':pd.DataFrame(plan['schedule'])}; ev[label]=regret(saved,f.smp,pd.DatetimeIndex(f.timestamp),EV(**record['settlement_assumptions']),v2g=label!='G2V')
            row.update(status='evaluated',**scored(f,f.predicted_smp,150),frozen_decision_evaluation=ev,forecast_sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),actual_sha256=hashlib.sha256(f[['timestamp','smp','first_seen']].to_csv(index=False).encode()).hexdigest())
            tag=row['actual_sha256'][:12]; dest=HOME/'evaluations'/(path.stem+'_'+tag+'.json')
            if not dest.exists(): write_once(dest,row)
        summaries.append(row)
    (HOME/'evaluation_index.json').write_text(json.dumps(summaries,indent=2,default=str)); return summaries

if __name__=='__main__': issue(); reconcile()
