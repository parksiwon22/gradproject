"""Fixed-price-model scenario policies with activity and realized incremental profit."""
import hashlib,json
import joblib,numpy as np,pandas as pd
from v11.store import ROOT,now
from v14.research import predict as oldpredict
from v12.ev import EV,score,optimize
from v15.research import HOME,write
from v15.policies import POLICIES,dispatch,residual_templates

def summary(rows):
    f=pd.DataFrame(rows); active=f.export_kwh>=.05
    return dict(days=len(f),mean_export_kwh=float(f.export_kwh.mean()),active_days=int(active.sum()),mean_actual_additional_profit=float(f.actual_gain.mean()),total_actual_additional_profit=float(f.actual_gain.sum()),mean_expected_additional_profit=float(f.expected_gain.mean()),loss_days=int((f.actual_gain< -1e-6).sum()),loss_frequency=float((f.actual_gain< -1e-6).mean()),active_loss_rate=float((f.loc[active,'actual_gain']< -1e-6).mean()) if active.any() else None,mean_gain_when_active=float(f.loc[active,'actual_gain'].mean()) if active.any() else None,mean_degradation_cost=float(f.degradation_cost.mean()),mean_actual_cost=float(f.actual_cost.mean()),mean_G2V_regret=float(f.G2V_regret.mean()),mean_V2G_regret=float(f.V2G_regret.mean()),fallback_days=int(f.fallback.sum()),scenario_hour_coverage=float(f.scenario_hour_coverage.mean()),scenario_day_coverage=float(f.scenario_day_coverage.mean()))

def run():
    folder=HOME/'policy_study'; folder.mkdir(exist_ok=True)
    design=dict(at=now(),models=['V12_hgb_baseline','shape_level_blend'],calibration=['2025-07-01','2025-09-30'],selection=['2025-10-01','2025-12-31'],audit=['2026-01-01','2026-06-30'],policies=POLICIES,scenario_source='Point plus6 calibration residual cluster centers and2 extreme complete-day residuals; equal stress weights, not calibrated probabilities',qualification='Selection mean actual gain vs same-model G2V >=10virtual KRW/day AND mean export >=.5kWh/day AND loss frequency no higher than general AND >=10active days; choose maximum gain; else G2V. Never select by2026 audit.',EV=EV(degradation=5).__dict__,common_terminal_SOC=True,scenario_parameters=dict(CVaR_alpha=.8,risk_weights=[.25,1],finite_scenario_cap=50,profit_gate=100),historical_not_independent=True)
    if not (folder/'design_before_scores.json').exists(): write(folder/'design_before_scores.json',design)
    f=pd.read_csv(HOME/'model_study/frozen_matrix.csv', float_precision='round_trip'); f.timestamp=pd.to_datetime(f.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
    cal=f[(f.target_date>='2025-07-01')&(f.target_date<='2025-09-30')].reset_index(drop=True); ev=EV(degradation=5)
    models={name:joblib.load(ROOT/'artifacts/v14/study'/(name+'.joblib')) for name in design['models']}
    templates={}; frames={}; predictions={}
    for name,b in models.items():
        errors=(cal.actual_smp.to_numpy()-oldpredict(b,cal)).reshape(-1,24); templates[name]=residual_templates(errors); np.save(folder/(name+'_templates.npy'),templates[name])
    totals=[]; oracle={}
    for cohort,limits in [('selection',design['selection']),('audit',design['audit'])]:
        data=f[(f.target_date>=limits[0])&(f.target_date<=limits[1])].reset_index(drop=True)
        for name,b in models.items():
            p=oldpredict(b,data)
            for policy in POLICIES:
                dest=folder/(cohort+'_'+name+'_'+policy+'.csv'); rows=pd.read_csv(dest, float_precision='round_trip').to_dict('records') if dest.exists() else []; done={r['day'] for r in rows}
                for day,ids in data.groupby('target_date',sort=True).indices.items():
                    if day in done: continue
                    g=data.iloc[ids]; times=pd.DatetimeIndex(g.timestamp); point=p[ids]; actual=g.actual_smp.to_numpy(); temp=templates[name]; scenarios=point[None,:]+temp
                    base=dispatch('G2V',point,times,temp,ev); bs=base['schedule']; base_actual=score(actual,bs.charge_kw.to_numpy(),bs.discharge_kw.to_numpy(),bs.connected_hours.to_numpy(),ev)['net_cost']; fallback=False
                    try: plan=dispatch(policy,point,times,temp,ev)
                    except ValueError: plan=base; fallback=True
                    s=plan['schedule']; expected=score(point,s.charge_kw.to_numpy(),s.discharge_kw.to_numpy(),s.connected_hours.to_numpy(),ev); realized=score(actual,s.charge_kw.to_numpy(),s.discharge_kw.to_numpy(),s.connected_hours.to_numpy(),ev)
                    if day not in oracle:
                        oracle[day]=[optimize(actual,times,ev,v2g=v)['net_cost'] for v in (False,True)]
                    cover=(actual>=scenarios.min(axis=0))&(actual<=scenarios.max(axis=0))
                    record=dict(day=day,model=name,policy=policy,expected_gain=base['net_cost']-expected['net_cost'],actual_gain=base_actual-realized['net_cost'],actual_cost=realized['net_cost'],G2V_actual_cost=base_actual,export_kwh=realized['grid_export_kwh'],degradation_cost=realized['degradation_cost'],G2V_regret=base_actual-oracle[day][0],V2G_regret=realized['net_cost']-oracle[day][1],fallback=fallback,scenario_hour_coverage=float(cover.mean()),scenario_day_coverage=bool(cover.all()),ending_soc=plan['ending_soc'])
                    rows.append(record)
                    path=folder/'plans'/cohort/name/policy/(day+'.json')
                    write(path,dict(mode='RETROSPECTIVE_SIMULATION_NOT_FORWARD',computed_at=now(),**record,settlement=ev.__dict__,schedule=s.to_dict('records'),scenario_excess_losses=plan.get('scenario_excess_losses')))
                    if len(rows)%7==0: pd.DataFrame(rows).to_csv(dest,index=False)
                pd.DataFrame(rows).to_csv(dest,index=False); result=dict(cohort=cohort,model=name,policy=policy,**summary(rows)); totals.append(result)
                pd.DataFrame(totals).to_csv(folder/'policy_summary.csv',index=False); print(cohort,name,policy,'export',round(result['mean_export_kwh'],3),'actual gain',round(result['mean_actual_additional_profit'],2),'loss days',result['loss_days'],flush=True)
        if cohort=='selection':
            selected=[]
            for name in models:
                group=pd.DataFrame(totals); group=group[(group.cohort=='selection')&(group.model==name)]; general=group[group.policy=='general'].iloc[0]
                ok=group[(group.mean_actual_additional_profit>=10)&(group.mean_export_kwh>=.5)&(group.active_days>=10)&(group.loss_frequency<=general.loss_frequency)]
                choice=ok.sort_values('mean_actual_additional_profit',ascending=False).iloc[0].policy if len(ok) else 'G2V'
                selected.append(dict(model=name,selected_policy=choice,eligible=ok.policy.tolist()))
            frozen=folder/'selection_before_audit.json'
            if frozen.exists(): assert json.loads(frozen.read_text())['choices']==selected,'Selection changed; no silent audit reselection'
            else: write(frozen,dict(at=now(),choices=selected,operational_promotion=False))
    print('Policy evaluation complete',flush=True)

if __name__=='__main__': run()
