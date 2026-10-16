"""Shared-origin48h shadow predictions and frozen connection-aware EV plans."""
import json,hashlib
from pathlib import Path
import joblib,numpy as np,pandas as pd
from v11.store import connect,latest,now
from v12.records import write_once
from v12.ev import EV,score,baseline
from v15.policies import dispatch
from v15.forward import publication_eligible
from v17.common import HOME,ROOT,write,sha
from v17.legacy import origin
from v17.publication import ensure,observations
from v17.study import matrix,query,infer,MODELS,PATTERNS,INPUTS
from v17.charging import priority_plan

def pack(plan):return {**{k:v for k,v in plan.items() if k!='schedule'},'schedule':plan['schedule'].to_dict('records')}

def issue(clock=None):
    started=pd.Timestamp.now(tz='Asia/Seoul')
    if clock is not None and abs((pd.Timestamp(clock)-started).total_seconds())>5:raise ValueError('Cannot backdate or fabricate a live prediction')
    bound=origin(started);target=bound.normalize()+pd.Timedelta(days=2);days=[(target+pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in [0,1]]
    run=dict(started_at=started.isoformat(),reference_origin=bound.isoformat(),primary_target_date=days[0],status='skipped')
    selection=HOME/'study/selection_before_audit.json';modelpath=HOME/'study/winter2025.joblib'
    if not selection.exists() or not modelpath.exists():run['reason']='Study not frozen before issue; no synthetic forecast'
    else:
        digest=sha(modelpath);dest=HOME/'forecasts'/(days[0]+'_'+digest[:12]+'.json')
        if dest.exists():run.update(status='existing_immutable_record',file=dest.name)
        else:
            evidence=ensure(days);clock=pd.Timestamp.now(tz='Asia/Seoul');db=connect();current=latest(db,'land');past=latest(db,'land',as_of=bound.tz_convert('UTC').isoformat());db.close()
            if current.timestamp.dt.strftime('%Y-%m-%d').eq(days[0]).any():run['reason']='Official target price already acquired; use official mode, no AI issue'
            elif origin(clock)!=bound:run['reason']='Origin changed during I/O; no backdated issue'
            elif not publication_eligible(evidence,days[0],clock)[0]:run['reason']='Missing, stale, failed or positive primary public-price evidence; absence not invented'
            else:
                past=past[past.timestamp<=bound].copy();frame=matrix(past,False,target)
                if len(frame)!=48:run['reason']='Incomplete true D2/D3 eligible features; no next-day cloning'
                else:
                    auxiliary_ok=publication_eligible(evidence,days[1],clock)[0] and not current.timestamp.dt.strftime('%Y-%m-%d').eq(days[1]).any()
                    bundle=joblib.load(modelpath);point=frame.daily.to_numpy()+bundle['regressor'].predict(frame[bundle['price_features']]);ev=EV(degradation=5);decisions={}
                    for pattern,hour,length in PATTERNS:
                        if length<30/(.9*7):decisions[pattern]=dict(status='infeasible',reason='Target SOC infeasible at fixed power');continue
                        if hour+length>24 and not auxiliary_ok:decisions[pattern]=dict(status='awaiting_eligible_D3',reason='No eligible genuine next-date forecast; no cloned prices');continue
                        q=query(frame,hour,length);p,scores=infer(bundle,q);times=pd.DatetimeIndex(q.timestamp);plans={}
                        for name in MODELS[:4]:
                            s=scores[name];planning=p if name=='joint_price' else np.sort(p)[pd.Series(s).rank(method='first').astype(int).to_numpy()-1]
                            plans[name]=dict(G2V=pack(priority_plan(planning,times,s,ev)),V2G_shadow=pack(dispatch('general',planning,times,None,ev)),priority_scores=s.tolist(),decision_prices_are_proxy=name!='joint_price')
                        decisions[pattern]=dict(status='frozen',features=q[INPUTS].to_dict('records'),plans=plans)
                    completed=pd.Timestamp.now(tz='Asia/Seoul')
                    if origin(completed)!=bound:raise ValueError('Origin changed before publication')
                    db=connect();check=latest(db,'land');db.close()
                    if check.timestamp.dt.strftime('%Y-%m-%d').eq(days[0]).any():run['reason']='Official prices acquired during generation; do not publish as future AI'
                    else:
                        if check.timestamp.dt.strftime('%Y-%m-%d').eq(days[1]).any():auxiliary_ok=False
                        if not auxiliary_ok:
                            for pattern,hour,length in PATTERNS:
                                if hour+length>24:decisions[pattern]=dict(status='awaiting_eligible_D3',reason='Next-date publication unverified or already official; no AI D3 plan')
                            frame=frame.iloc[:24].copy();point=point[:24]
                        snapshot=past[past.timestamp>=bound-pd.Timedelta(days=8)]
                        record=dict(schema=17,issued_at=completed.isoformat(),generation_started_at=started.isoformat(),reference_origin=bound.isoformat(),primary_target_date=days[0],auxiliary_target_date=days[1],actual_delay_minutes=(completed-bound).total_seconds()/60,exact_23h_run=abs((completed-bound).total_seconds())<60,model_sha256=digest,trained_through=bundle['train_end'],model_trained_at=bundle['trained_at'],selection_locked_at=json.loads(selection.read_text())['at'],model_locked_after_reference_origin=pd.Timestamp(json.loads(selection.read_text())['at'])>bound,mode='EXPERIMENTAL_SHADOW_NOT_OPERATING_REPLACEMENT',official_known_at_issue=False,publication_evidence=[r for r in evidence if r['target_date'] in days],publication_caveat='Fresh negative checks on two queried sources are not universal first-publication proof',source_prices=snapshot[['timestamp','smp','first_seen','snapshot']].to_dict('records'),input_first_seen_max=snapshot.first_seen.max(),price_features=frame[bundle['price_features']].to_dict('records'),price_rows=[dict(timestamp=t.isoformat(),predicted_smp=float(p)) for t,p in zip(frame.timestamp,point)],decisions=decisions,settlement=ev.__dict__,recommendation='G2V; ranking models and V2G are fixed shadow comparisons only',uncertainty='Uncalibrated ranking/class scores, no promised savings or conditional coverage; historical model selection previously inspected',weather_forecast_used=False,demand_forecast_used=False)
                        record.update(coverage_hours=len(frame),auxiliary_prediction_published=auxiliary_ok,auxiliary_caveat='Cross-date plans withheld unless genuine D3 publication checks pass')
                        write_once(dest,record);run.update(status='created',file=dest.name,issued_at=completed.isoformat(),delay_minutes=record['actual_delay_minutes'])
    run['completed_at']=now();write_once(HOME/'forward_runs'/(pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%f')+'.json'),run);print(json.dumps(run));return run

def reconcile():
    db=connect();prices=latest(db,'land');db.close();idx=prices.set_index('timestamp');summaries=[]
    for path in (HOME/'forecasts').glob('*.json'):
        r=json.loads(path.read_text());f=pd.DataFrame(r['price_rows']);f.timestamp=pd.to_datetime(f.timestamp,utc=True).dt.tz_convert('Asia/Seoul');f['actual']=idx.smp.reindex(f.timestamp).to_numpy();ev=EV(**r['settlement']);dates=[];decisions=[]
        for day,g in f.groupby(f.timestamp.dt.strftime('%Y-%m-%d')):
            if g.actual.notna().sum()!=24:dates.append(dict(day=day,status='awaiting_actual',hours=int(g.actual.notna().sum())));continue
            a=g.actual.to_numpy();p=g.predicted_smp.to_numpy();peak=a>=150;pred=p>=150;tp=int((peak&pred).sum());precision=tp/max(1,int(pred.sum()));recall=tp/max(1,int(peak.sum()));dates.append(dict(day=day,status='price_comparison_complete',hours=24,MAE=float(abs(a-p).mean()),RMSE=float(np.sqrt(np.mean((a-p)**2))),peak_hours=int(peak.sum()),peak_MAE=float(abs(a[peak]-p[peak]).mean()) if peak.any() else None,peak_F1=2*precision*recall/max(1e-9,precision+recall),target_day_elapsed=pd.Timestamp(day,tz='Asia/Seoul')+pd.Timedelta(days=1)<=pd.Timestamp.now(tz='Asia/Seoul')))
        for row in dates:
            if row['status']!='price_comparison_complete':continue
            g=f[f.timestamp.dt.strftime('%Y-%m-%d').eq(row['day'])];a=g.actual.to_numpy();p=g.predicted_smp.to_numpy();da=np.diff(a);dp=np.diff(p)
            row.update(peak_price_absolute_error=float(abs(a.max()-p.max())),peak_hour_absolute_error=int(abs(np.argmax(a)-np.argmax(p))),up_ramp_MAE=float(abs(da[da>0]-dp[da>0]).mean()) if (da>0).any() else None,down_ramp_MAE=float(abs(da[da<0]-dp[da<0]).mean()) if (da<0).any() else None,ramp_boundary='Within target day only; no actual boundary supplied to prediction')
        for pattern,entry in r['decisions'].items():
            if entry['status']!='frozen':decisions.append(dict(pattern=pattern,status=entry['status'],reason=entry.get('reason')));continue
            for name,plans in entry['plans'].items():
                gs=pd.DataFrame(plans['G2V']['schedule']);vs=pd.DataFrame(plans['V2G_shadow']['schedule']);times=pd.DatetimeIndex(pd.to_datetime(gs.timestamp,utc=True)).tz_convert('Asia/Seoul');a=idx.smp.reindex(times).to_numpy()
                if not np.isfinite(a).all():decisions.append(dict(pattern=pattern,model=name,status='awaiting_actual'));continue
                gc=score(a,gs.charge_kw.to_numpy(),gs.discharge_kw.to_numpy(),gs.connected_hours.to_numpy(),ev);vc=score(a,vs.charge_kw.to_numpy(),vs.discharge_kw.to_numpy(),vs.connected_hours.to_numpy(),ev);oracle=priority_plan(a,times,ev=ev);best=oracle['schedule'].charge_kw.to_numpy()>1e-8;chosen=gs.charge_kw.to_numpy()>1e-8
                decisions.append(dict(pattern=pattern,model=name,status='evaluated',G2V_actual=gc['net_cost'],G2V_regret=gc['net_cost']-oracle['net_cost'],cheap_hour_overlap=float((best&chosen).sum()/max(1,best.sum())),saving_immediate=baseline(a,times,ev)['net_cost']-gc['net_cost'],saving_uniform=baseline(a,times,ev,uniform=True)['net_cost']-gc['net_cost'],V2G_actual_gain=gc['net_cost']-vc['net_cost'],V2G_export=vc['grid_export_kwh'],V2G_loss=vc['net_cost']>gc['net_cost']+1e-6))
        actual=idx[['smp','first_seen']].reindex(f.timestamp);actualhash=hashlib.sha256(actual.to_csv().encode()).hexdigest();result=dict(forecast=path.name,forecast_sha256=sha(path),primary_target_date=r['primary_target_date'],evaluated_at=now(),actual_sha256=actualhash,price_dates=dates,decisions=decisions)
        result['evaluation_schema']=2
        dest=HOME/'evaluations'/(path.stem+'_v2_'+actualhash[:12]+'.json')
        if not dest.exists():write_once(dest,result)
        if any(d['status']=='price_comparison_complete' for d in dates):
            figure=HOME/'evaluation_figures'/(dest.stem+'.png')
            if not figure.exists():
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                figure.parent.mkdir(exist_ok=True,parents=True)
                fig,ax=plt.subplots(figsize=(12,4));ax.plot(f.timestamp,f.predicted_smp,label='Frozen prior prediction');ax.plot(f.timestamp,f.actual,label='Current official SMP');ax.set_ylabel('KRW/kWh');ax.set_title('Actual receipt comparison; issued '+r['issued_at']);ax.legend();fig.autofmt_xdate();fig.tight_layout();fig.savefig(figure,dpi=140);plt.close(fig)
        summaries.append(result)
    write(HOME/'evaluation_index.json',summaries);return summaries

def status():
    targets=set();done=set();spikes=set();pending=set();new=list((HOME/'forecasts').glob('*.json'))
    for version in ['v12','v14','v15']:
        base=HOME.parent/version
        targets.update(json.loads(p.read_text())['target_date'] for p in (base/'forecasts').glob('*.json'))
        file=base/'evaluation_index.json'
        for r in json.loads(file.read_text()) if file.exists() else []:
            if r['status']=='evaluated':done.add(r['target_date']);spikes.update([r['target_date']] if r.get('peak_hours',0) else [])
    evaluated=json.loads((HOME/'evaluation_index.json').read_text()) if (HOME/'evaluation_index.json').exists() else []
    for p in new:targets.add(json.loads(p.read_text())['primary_target_date'])
    for r in evaluated:
        for day in r['price_dates']:
            if day['day']==r['primary_target_date'] and day['status']=='price_comparison_complete':done.add(day['day']);spikes.update([day['day']] if day.get('peak_hours',0) else [])
    value=dict(at=now(),independent_primary_target_days=len(targets),primary_dates=sorted(targets),price_comparison_complete_days=len(done),awaiting_primary_days=len(targets-done),spike_days=len(spikes),elapsed_evaluated_days=sum(pd.Timestamp(d,tz='Asia/Seoul')+pd.Timedelta(days=1)<=pd.Timestamp.now(tz='Asia/Seoul') for d in done),V17_joint48_records=len(new),V17_independent_origins=len({json.loads(p.read_text())['reference_origin'] for p in new}),auxiliary_D3_dates_are_not_extra_independent_origins=True,default_recommendation='G2V',operating_model_replaced=False)
    write(HOME/'prospective_status.json',value);return value

if __name__=='__main__':issue();reconcile();print(status())
