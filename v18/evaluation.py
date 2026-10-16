"""Evaluate original AI and frozen EV decisions without rewriting either."""
import hashlib,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from v11.store import connect,latest
from v12.ev import EV,score,baseline
from v15.policies import dispatch
from v17.charging import priority_plan,require_prices
from v18.common import HOME,ROOT,read,write,write_once,sha,now
from v18.publication import all_rows,POSITIVE

EV_DEFAULT=EV(degradation=5)

def timestamp(values):return pd.to_datetime(values,utc=True,format='mixed').dt.tz_convert('Asia/Seoul')

def inventory():
    output=[]
    for version in range(12,19):
        for path in sorted((ROOT/'artifacts'/('v'+str(version))/'forecasts').glob('*.json')):
            r=read(path);rows=r.get('price_rows',r.get('rows',[]))
            if not rows:continue
            f=pd.DataFrame(rows);f['timestamp']=timestamp(f.timestamp)
            if f.timestamp.duplicated().any() or not np.isfinite(f.predicted_smp).all():raise ValueError('Malformed saved forecast')
            issued=pd.Timestamp(r['issued_at']);origin=pd.Timestamp(r['reference_origin']);source=r.get('source_prices',[]);input_max=r.get('input_first_seen_max',r.get('inputs_first_seen_max'));known=r.get('official_known_at_issue',r.get('official_smp_known_at_issue'))
            receipts_ok=None if not source else bool((pd.to_datetime(pd.Series([s['first_seen'] for s in source]),utc=True,format='mixed')<=origin).all() and all(pd.Timestamp(s['timestamp'])<=origin for s in source))
            locked=r.get('selection_locked_at',r.get('model_registered_at'));model_before=None if locked is None else pd.Timestamp(locked)<=origin
            if r.get('model_locked_after_reference_origin'):model_before=False
            delay=(issued-origin).total_seconds()/60
            primary=r.get('primary_target_date',r.get('target_date'))
            evidence=[e for e in all_rows() if e['target_date']==primary and e['market']=='land' and e['status']==POSITIVE and pd.Timestamp(e['checked_at'])<=issued]
            known_observation=bool(evidence) or known is True
            meta=dict(file=path.name,version=version,model=r.get('model','V17_joint_price'),sha256=sha(path),issued_at=issued.isoformat(),reference_origin=origin.isoformat(),primary_target_date=primary,delay_minutes=delay,timing='on_time_within_60s' if 0<=delay<=1 else 'late' if delay>1 else 'invalid_clock',model_locked_before_origin=model_before,input_receipts_eligible=receipts_ok,input_first_seen_max=input_max,official_known_at_issue_record=known,official_positive_observed_by_issue=known_observation,universal_prepublication_certified=False,strict_on_time_eligible=bool(0<=delay<=1 and model_before is True and receipts_ok is True and known is False and not known_observation),final_statistics_verified=False)
            output.append((path,r,f,meta))
    return output

def reference():
    db=connect();f=latest(db,'land')[['timestamp','smp','first_seen','snapshot']];db.close();f=f.assign(source='legacy_official_store')
    # Current complete public website > current EPSIS > current official API.
    groups={}
    for row in all_rows():
        if row['market']=='land' and row['status']==POSITIVE and row.get('prices'):groups.setdefault(row['target_date'],[]).append(row)
    conflicts=[]
    for day,rows in groups.items():
        priority={'general_public':3,'epsis':2,'kpx_api':1};r=max(rows,key=lambda x:(priority.get(x['source'],0),pd.Timestamp(x['checked_at'])))
        a=pd.DataFrame(r['prices']);a.timestamp=timestamp(a.timestamp);a['first_seen']=r['checked_at'];a['snapshot']=r['sha256'];a['source']=r['source']
        for other in rows:
            if other['source']==r['source']:continue
            b=pd.DataFrame(other['prices']);b.timestamp=timestamp(b.timestamp);pair=a[['timestamp','smp']].merge(b,on='timestamp',suffixes=('_a','_b'))
            if len(pair)==24 and not np.allclose(pair.smp_a,pair.smp_b,atol=.001,rtol=0):conflicts.append(dict(target_date=day,preferred_source=r['source'],other_source=other['source'],max_absolute_difference=float(abs(pair.smp_a-pair.smp_b).max())))
        f=f[~f.timestamp.dt.strftime('%Y-%m-%d').eq(day)];f=pd.concat([f,a],ignore_index=True)
    return f.sort_values('timestamp').reset_index(drop=True),conflicts

def price_metrics(a,p):
    a=np.asarray(a,float);p=np.asarray(p,float);peak=a>=150;pred=p>=150;tp=int((peak&pred).sum());precision=tp/max(1,int(pred.sum()));recall=tp/max(1,int(peak.sum()));da=np.diff(a);dp=np.diff(p)
    return dict(MAE=float(abs(a-p).mean()),RMSE=float(np.sqrt(np.mean((a-p)**2))),peak_hours=int(peak.sum()),peak_MAE=float(abs(a[peak]-p[peak]).mean()) if peak.any() else None,peak_bias=float((p[peak]-a[peak]).mean()) if peak.any() else None,peak_F1=2*precision*recall/max(1e-9,precision+recall) if peak.any() or pred.any() else None,peak_hour_error=int(abs(np.argmax(a)-np.argmax(p))),rank_spearman=float(spearmanr(a,p).statistic) if np.ptp(a)>0 and np.ptp(p)>0 else None,cheap5_overlap=len(set(np.argsort(a,kind='stable')[:5])&set(np.argsort(p,kind='stable')[:5]))/5,up_ramp_MAE=float(abs(da[da>0]-dp[da>0]).mean()) if (da>0).any() else None,down_ramp_MAE=float(abs(da[da<0]-dp[da<0]).mean()) if (da<0).any() else None)

def compare_ev(pred,actual,times,ev=None,start=None,end=None):
    ev=ev or EV_DEFAULT;times=pd.DatetimeIndex(times);p=np.asarray(pred,float);a=np.asarray(actual,float)
    g=priority_plan(p,times,ev=ev,start=start,end=end);v=dispatch('general',p,times,None,ev,start,end);og=priority_plan(a,times,ev=ev,start=start,end=end);ov=dispatch('general',a,times,None,ev,start,end)
    def realized(plan):
        s=plan['schedule'];return score(a,s.charge_kw.to_numpy(),s.discharge_kw.to_numpy(),s.connected_hours.to_numpy(),ev)
    gc,vc=realized(g),realized(v);im=baseline(a,times,ev,start,end)['net_cost'];un=baseline(a,times,ev,start,end,True)['net_cost']
    return dict(G2V_expected=g['net_cost'],G2V_actual=gc['net_cost'],official_G2V_optimum=og['net_cost'],G2V_regret=gc['net_cost']-og['net_cost'],immediate_actual=im,uniform_actual=un,saving_immediate=im-gc['net_cost'],saving_uniform=un-gc['net_cost'],V2G_expected_gain=g['net_cost']-v['net_cost'],V2G_actual=vc['net_cost'],V2G_regret=vc['net_cost']-ov['net_cost'],V2G_actual_gain=gc['net_cost']-vc['net_cost'],V2G_export=vc['grid_export_kwh'],V2G_degradation_cost=vc['degradation_cost'],G2V=g,V2G=v,official_G2V=og,official_V2G=ov)

def frozen_plans(path,r):
    if r.get('schema')==17:
        for pattern,entry in r['decisions'].items():
            if entry['status']=='frozen':
                for model,plans in entry['plans'].items():yield pattern,model,plans['G2V'],plans['V2G_shadow'],r['settlement'],r['issued_at']
    elif r.get('decisions'):
        plans=r['decisions'];g=plans.get('G2V');v=plans.get('V2G',plans.get('general'))
        if g and v:yield 'full24',r.get('model','hgb_none'),g,v,r.get('settlement_assumptions',g['assumptions']),r['issued_at']
    else:
        file=ROOT/'artifacts/v12/decisions'/path.name;d=read(file)
        if d and d.get('source_sha256')==sha(path):yield 'full24',r['model'],d['plans']['G2V'],d['plans']['V2G'],d['plans']['G2V']['assumptions'],d['at']

def evaluate():
    f,conflicts=reference();idx=f.set_index('timestamp');rows=[];frozen=[];clock=pd.Timestamp.now(tz='Asia/Seoul');items=inventory()
    for path,r,forecast,meta in items:
        reference_columns=[]
        for day,g in forecast.groupby(forecast.timestamp.dt.strftime('%Y-%m-%d')):
            a=idx.smp.reindex(g.timestamp).to_numpy();row=dict(**meta,target_date=day,auxiliary_target=day!=meta['primary_target_date'],official_hours=int(np.isfinite(a).sum()),target_day_elapsed=pd.Timestamp(day,tz='Asia/Seoul')+pd.Timedelta(days=1)<=clock,comparison_basis='Official provided prices; not certified final corrected statistics',EV_comparison='Retrospective replay of saved price vector under identical generic EV; not newly issued prior decision')
            if len(g)!=24 or not np.isfinite(a).all():row['status']='awaiting_official_prices'
            else:
                row.update(status='official_price_comparison_complete',**price_metrics(a,g.predicted_smp));result=compare_ev(g.predicted_smp,a,g.timestamp);row.update({k:v for k,v in result.items() if k not in ['G2V','V2G','official_G2V','official_V2G']});reference_columns.extend(idx[['smp','source']].reindex(g.timestamp).to_dict('records'))
            rows.append(row)
        for pattern,model,g,v,assumptions,decision_at in frozen_plans(path,r):
            gs=pd.DataFrame(g['schedule']);vs=pd.DataFrame(v['schedule']);times=pd.DatetimeIndex(timestamp(gs.timestamp));a=idx.smp.reindex(times).to_numpy();row=dict(forecast=path.name,forecast_sha256=meta['sha256'],version=meta['version'],model=model,pattern=pattern,decision_generated_at=decision_at,issued_at=meta['issued_at'],decision_frozen_by_issue=pd.Timestamp(decision_at)<=pd.Timestamp(meta['issued_at']),primary_target_date=meta['primary_target_date'])
            if not np.isfinite(a).all():row['status']='awaiting_official_prices'
            else:
                ev=EV(**assumptions);gc=score(a,gs.charge_kw.to_numpy(),gs.discharge_kw.to_numpy(),gs.connected_hours.to_numpy(),ev);vc=score(a,vs.charge_kw.to_numpy(),vs.discharge_kw.to_numpy(),vs.connected_hours.to_numpy(),ev);oracle=priority_plan(a,times,ev=ev);row.update(status='frozen_decision_comparison_complete',G2V_actual=gc['net_cost'],G2V_regret=gc['net_cost']-oracle['net_cost'],V2G_actual_gain=gc['net_cost']-vc['net_cost'],V2G_export=vc['grid_export_kwh'],G2V_expected=g['net_cost'],V2G_expected_gain=g['net_cost']-v['net_cost'],cheap_hour_overlap=float(((gs.charge_kw.to_numpy()>1e-8)&(oracle['schedule'].charge_kw.to_numpy()>1e-8)).sum()/max(1,(oracle['schedule'].charge_kw.to_numpy()>1e-8).sum())))
            frozen.append(row)
        own=[x for x in rows if x['file']==path.name];own_frozen=[x for x in frozen if x['forecast']==path.name];version=hashlib.sha256(json.dumps(dict(rows=own,frozen=own_frozen,reference=reference_columns),sort_keys=True,default=str).encode()).hexdigest();dest=HOME/'evaluations'/(path.stem+'_'+version[:16]+'.json')
        if not dest.exists():write_once(dest,dict(schema=18,at=now(),forecast_sha256=sha(path),rows=own,frozen_decisions=own_frozen,reference_version=version))
    pd.DataFrame(rows).to_csv(HOME/'comparison_index.csv',index=False);pd.DataFrame(frozen).to_csv(HOME/'frozen_decision_index.csv',index=False)
    primary=[r for r in rows if not r['auxiliary_target']];done={r['target_date'] for r in primary if r['status']=='official_price_comparison_complete'};targets={r['primary_target_date'] for r in primary}
    status=dict(at=now(),forecast_files=len(items),independent_reference_origins=len({r['reference_origin'] for r in primary}),independent_primary_target_days=len(targets),primary_dates=sorted(targets),official_price_comparison_complete_days=len(done),target_day_elapsed_days=len({r['target_date'] for r in primary if r['target_day_elapsed'] and r['target_date'] in done}),final_corrected_statistics_verified_days=0,spike_target_days=len({r['target_date'] for r in primary if r.get('peak_hours',0)>0}),awaiting_official_target_days=len(targets-done),on_time_forecast_records=sum(m['timing']=='on_time_within_60s' for _,_,_,m in items),late_forecast_records=sum(m['timing']=='late' for _,_,_,m in items),strict_on_time_eligible_records=sum(m['strict_on_time_eligible'] for _,_,_,m in items),official_known_by_issue_records=sum(m['official_positive_observed_by_issue'] for _,_,_,m in items),unknown_model_lock_records=sum(m['model_locked_before_origin'] is None for _,_,_,m in items),reference_conflicts=conflicts,operating_model_changed=False,research_purpose='Prepublication immutable AI, later official comparison and EV regret; unchanged')
    status.update(on_time_reference_origins=len({m['reference_origin'] for _,_,_,m in items if m['timing']=='on_time_within_60s'}),late_reference_origins=len({m['reference_origin'] for _,_,_,m in items if m['timing']=='late'}),strict_eligible_reference_origins=len({m['reference_origin'] for _,_,_,m in items if m['strict_on_time_eligible']}))
    write(HOME/'status.json',status);write(HOME/'inventory.json',[m for _,_,_,m in items]);return status

if __name__=='__main__':print(evaluate())
