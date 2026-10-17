"""Day/horizon aligned comparison of original predictions and frozen EV plans."""
import hashlib,json
import numpy as np,pandas as pd
from v12.ev import EV,score,baseline
from v15.policies import dispatch
from v17.charging import priority_plan
from v18.evaluation import inventory as old_inventory,reference,price_metrics,compare_ev,frozen_plans,timestamp
from v19.common import HOME,ROOT,read,write,write_once,write_csv,sha,now
ALIASES={'hgb_none':'V12_HGB','lightgbm_peak':'V12_LightGBM_peak','shape_level_blend':'V14_shape_level_blend','two_stage_shrink_0.25':'V15_two_stage_025','two_stage_shrink_0.5':'V15_two_stage_05','V17_joint_price':'V17_joint'}

def inventory():
    output=[]
    for path,r,f,m in old_inventory():output.append((path,r,f,dict(m,original_model=m['model'],model=ALIASES.get(m['model'],m['model']))))
    for path in (HOME/'forecasts').glob('*.json'):
        r=read(path)
        if r.get('mode')=='SIMULATION_ONLY':continue
        f=pd.DataFrame(r['rows']);f.timestamp=timestamp(f.timestamp);commit=read(HOME/'commits'/(path.stem+'.json'));bound=pd.Timestamp(r['reference_origin']);issued=pd.Timestamp(r['issued_at'])
        verified=bool(commit and commit['sha256']==sha(path));meta=dict(file=path.name,version=19,model=r['model'],sha256=sha(path),issued_at=r['issued_at'],reference_origin=r['reference_origin'],primary_target_date=r['target_date'],delay_minutes=(issued-bound).total_seconds()/60,timing=commit['timing'] if verified else 'commit_time_unverified',model_locked_before_origin=pd.Timestamp(r['model_registered_at'])<bound,input_receipts_eligible=pd.Timestamp(r['input_first_seen_max'])<=pd.Timestamp(r['input_snapshot_at'])<bound,official_positive_observed_by_issue=r['official_known_at_issue'],strict_on_time_eligible=verified and commit['timing']=='on_time_within_60s',commit_verified=verified)
        output.append((path,r,f,meta))
    return output

def evaluate():
    ref,conflicts=reference();idx=ref.set_index('timestamp');rows=[];fixed=[];clock=pd.Timestamp.now(tz='Asia/Seoul');items=inventory()
    for path,r,f,meta in items:
        for day,g in f.groupby(f.timestamp.dt.strftime('%Y-%m-%d')):
            horizon=(pd.Timestamp(day).date()-pd.Timestamp(meta['reference_origin']).date()).days
            row=dict(**meta,target_date=day,horizon_days=horizon,cohort='V19_actual_publication' if meta['version']==19 else 'legacy_actual_record',comparison_group=r.get('comparison_group','legacy_D'+str(horizon)+'_different_issue_times'),target_day_elapsed=pd.Timestamp(day,tz='Asia/Seoul')+pd.Timedelta(days=1)<=clock);row['final_statistics_verified']=False;row['model_sha256']=r.get('model_sha256',r.get('model_version'))
            a=idx.smp.reindex(g.timestamp).to_numpy();row['official_hours']=int(np.isfinite(a).sum())
            if len(g)!=24 or not np.isfinite(a).all():row['status']='awaiting_official'
            else:
                result=compare_ev(g.predicted_smp,a,g.timestamp);row.update(status='official_provided_price_comparison',**price_metrics(a,g.predicted_smp),**{k:v for k,v in result.items() if k not in ['G2V','V2G','official_G2V','official_V2G']})
                row['replay_note']='Same-EV retrospective price replay; original frozen plan is a separate table'
            rows.append(row)
        plans=list(frozen_plans(path,r))
        for pattern,model,g,v,assumptions,decision_at in plans:
            gs=pd.DataFrame(g['schedule']);vs=pd.DataFrame(v['schedule']);times=pd.DatetimeIndex(timestamp(gs.timestamp));a=idx.smp.reindex(times).to_numpy();row=dict(file=path.name,forecast_sha256=sha(path),model=model,pattern=pattern,reference_origin=meta['reference_origin'],target_date=meta['primary_target_date'],horizon_days=(pd.Timestamp(meta['primary_target_date']).date()-pd.Timestamp(meta['reference_origin']).date()).days,decision_at=decision_at,decision_frozen_by_issue=pd.Timestamp(decision_at)<=pd.Timestamp(meta['issued_at']),status='awaiting_official',terminal_SOC=g['ending_soc'])
            if np.isfinite(a).all():
                ev=EV(**assumptions);gc=score(a,gs.charge_kw.to_numpy(),gs.discharge_kw.to_numpy(),gs.connected_hours.to_numpy(),ev);vc=score(a,vs.charge_kw.to_numpy(),vs.discharge_kw.to_numpy(),vs.connected_hours.to_numpy(),ev);og=priority_plan(a,times,ev=ev);ov=dispatch('general',a,times,None,ev);row.update(status='frozen_decision_evaluated',G2V_expected=g['net_cost'],G2V_actual=gc['net_cost'],official_G2V_optimum=og['net_cost'],G2V_regret=gc['net_cost']-og['net_cost'],V2G_expected=v['net_cost'],V2G_actual=vc['net_cost'],official_V2G_optimum=ov['net_cost'],V2G_regret=vc['net_cost']-ov['net_cost'],V2G_expected_gain=g['net_cost']-v['net_cost'],V2G_actual_gain=gc['net_cost']-vc['net_cost'],V2G_export=vc['grid_export_kwh'],immediate_actual=baseline(a,times,ev)['net_cost'],uniform_actual=baseline(a,times,ev,uniform=True)['net_cost'])
            fixed.append(row)
        subset=[x for x in rows if x['file']==path.name];decisions=[x for x in fixed if x['file']==path.name];digest=hashlib.sha256(json.dumps([subset,decisions],sort_keys=True,default=str).encode()).hexdigest();dest=HOME/'evaluations'/(path.stem+'_'+digest[:16]+'.json')
        if not dest.exists():
            try:write_once(dest,dict(at=now(),forecast_sha256=sha(path),rows=subset,frozen_decisions=decisions,evaluation_stage='official_provided_price_not_final_statistics'))
            except FileExistsError:pass
    frame=pd.DataFrame(rows);write_csv(HOME/'comparison.csv',frame);write_csv(HOME/'frozen_costs.csv',pd.DataFrame(fixed))
    paired=[]
    for (day,model,digest),g in frame.groupby(['target_date','model','model_sha256'],dropna=False):
        d2=g[g.horizon_days==2];d3=g[g.horizon_days==3]
        pair=dict(target_date=day,model=model,model_sha256=digest,D2_count=len(d2),D3_count=len(d3),paired_actual_records=not d2.empty and not d3.empty)
        if pair['paired_actual_records']:
            a=d2.iloc[-1];b=d3.iloc[-1];pair.update(D3_minus_D2_MAE=float(b.MAE-a.MAE) if pd.notna(a.get('MAE')) and pd.notna(b.get('MAE')) else None,D3_minus_D2_G2V_regret=float(b.G2V_regret-a.G2V_regret) if pd.notna(a.get('G2V_regret')) and pd.notna(b.get('G2V_regret')) else None)
        paired.append(pair)
    write_csv(HOME/'paired_horizons.csv',pd.DataFrame(paired));new=[m for _,_,_,m in items if m['version']==19];origins={m['reference_origin'] for _,_,_,m in items};targets={m['primary_target_date'] for _,_,_,m in items};status=dict(at=now(),actual_record_files=len(items),V19_new_actual_records=len(new),V19_new_target_dates=sorted({m['primary_target_date'] for m in new}),independent_origins=len(origins),independent_primary_target_days=len(targets),paired_same_model_actual_target_count=sum(p['paired_actual_records'] for p in paired),V19_on_time_eligible=sum(m.get('strict_on_time_eligible',False) for m in new),V19_late_records=sum(m.get('timing')=='late' for m in new),official_comparison_target_dates=sorted(frame.loc[frame.status=='official_provided_price_comparison','target_date'].unique()),final_corrected_statistics_verified_days=0,spike_target_days=len(set(frame.loc[frame.get('peak_hours',pd.Series(index=frame.index,dtype=float))>0,'target_date'])),target_day_elapsed_days=len(set(frame.loc[frame.target_day_elapsed,'target_date'])),reference_conflicts=conflicts,operating_model_replaced=False)
    write(HOME/'status.json',status);return status

if __name__=='__main__':print(evaluate())
