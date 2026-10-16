"""Fixed forecasts, physical sensitivities and explicitly nondeployable oracles."""
import itertools,json
import numpy as np,pandas as pd
from v12.ev import EV,score
from v15.policies import dispatch
from v15.research import HOME as V15
from v16.common import HOME,ROOT,write,csv

def cycle_margin(buy,sell,charge_eff=.9,discharge_eff=.9,wear=5,sell_ratio=1):
    if not np.isfinite([buy,sell,charge_eff,discharge_eff,wear,sell_ratio]).all() or not 0<charge_eff<=1 or not 0<discharge_eff<=1 or wear<0 or not 0<=sell_ratio<=1: raise ValueError('Invalid economic assumptions')
    recharge=buy/(charge_eff*discharge_eff); degradation=2*wear/discharge_eff
    return dict(margin_per_export_kwh=sell_ratio*sell-recharge-degradation,break_even_sell_price=(recharge+degradation)/sell_ratio if sell_ratio>0 else None,recharge_cost=recharge,extra_degradation=degradation,warning='One compensated cycle, excludes SOC/power/connection constraints and prediction risk')

def cost(p,plan,ev):
    s=plan['schedule'];return score(np.asarray(p),s.charge_kw.to_numpy(),s.discharge_kw.to_numpy(),s.connected_hours.to_numpy(),ev)

def source_days(cohort):
    f=csv(V15/'model_study/frozen_matrix.csv');f.timestamp=pd.to_datetime(f.timestamp,utc=True).dt.tz_convert('Asia/Seoul');idx=f.set_index('timestamp').actual_smp
    for path in sorted((V15/'policy_study/plans'/cohort/'shape_level_blend'/'G2V').glob('*.json')):
        saved=json.loads(path.read_text());s=pd.DataFrame(saved['schedule']);times=pd.DatetimeIndex(pd.to_datetime(s.timestamp,utc=True)).tz_convert('Asia/Seoul');actual=idx.reindex(times).to_numpy()
        if len(actual)!=24 or not np.isfinite(actual).all():raise ValueError('Complete frozen price day required')
        yield path.stem,times,s.price.to_numpy(),actual

def summarize(f):
    return dict(days=len(f),mean_export=float(f.export.mean()),active_days=int((f.export>=.05).sum()),mean_gain=float(f.gain.mean()),loss_days=int((f.gain< -1e-6).sum()),loss_frequency=float((f.gain< -1e-6).mean()),mean_expected_gain=float(f.expected_gain.mean()),mean_wear=float(f.wear.mean()),mean_G2V_cost=float(f.G2V_actual.mean()),mean_V2G_cost=float(f.V2G_actual.mean()),mean_G2V_regret=float(f.G2V_regret.mean()),mean_V2G_regret=float(f.V2G_regret.mean()))

def sensitivity():
    folder=HOME/'sensitivity';folder.mkdir(parents=True,exist_ok=True);summaries=[]; choices=[]
    for cohort in ['selection','audit']:
        days=list(source_days(cohort))
        for eff,wear,sell in itertools.product([.8,.9,.95],[0,5,15],[.8,.9,1]):
            name=f'eff{eff}_wear{wear}_sell{sell}';path=folder/(cohort+'_'+name+'.csv');rows=csv(path).to_dict('records') if path.exists() else [];done={r['day'] for r in rows}
            ev=EV(charge_eff=eff,discharge_eff=eff,degradation=wear,sell_ratio=sell)
            for day,times,p,a in days:
                if day in done:continue
                # All schedules have the same exact terminal SOC, including oracle.
                g=dispatch('G2V',p,times,None,ev);v=dispatch('general',p,times,None,ev)
                og=dispatch('G2V',a,times,None,ev);ov=dispatch('general',a,times,None,ev)
                gc,vc=cost(a,g,ev),cost(a,v,ev)
                rows.append(dict(day=day,expected_gain=g['net_cost']-v['net_cost'],gain=gc['net_cost']-vc['net_cost'],G2V_actual=gc['net_cost'],V2G_actual=vc['net_cost'],export=vc['grid_export_kwh'],wear=vc['degradation_cost'],G2V_regret=gc['net_cost']-og['net_cost'],V2G_regret=vc['net_cost']-ov['net_cost'],oracle_incremental_benefit=og['net_cost']-ov['net_cost'],ending_soc=v['ending_soc']))
                if len(rows)%14==0:pd.DataFrame(rows).to_csv(path,index=False)
            f=pd.DataFrame(rows);f.to_csv(path,index=False);result=dict(cohort=cohort,efficiency=eff,wear=wear,sell_ratio=sell,**summarize(f));summaries.append(result)
            pd.DataFrame(summaries).to_csv(ROOT/'docs/V16_SENSITIVITY.csv',index=False)
            if cohort=='selection':
                ok=result['mean_gain']>=10 and result['mean_export']>=.5 and result['active_days']>=10 and result['loss_frequency']<=.1
                choices.append(dict(efficiency=eff,wear=wear,sell_ratio=sell,qualified=ok,recommendation='V2G_shadow_only' if ok else 'G2V'))
            print(cohort,name,'actual gain',round(result['mean_gain'],2),'export',round(result['mean_export'],3),flush=True)
        if cohort=='selection':
            frozen=folder/'selection_before_audit.json'
            if frozen.exists():assert json.loads(frozen.read_text())['choices']==choices,'No audit-dependent selection changes'
            else:write(frozen,dict(choices=choices,not_operational_promotion=True))
    print('Sensitivity completed:27 assumptions x273 days',flush=True)

def information_oracles():
    folder=HOME/'information';folder.mkdir(parents=True,exist_ok=True);rows=[]
    for cohort in ['selection','audit']:
        dest=folder/(cohort+'.csv');old=csv(dest).to_dict('records') if dest.exists() else [];done={(r['day'],r['information']) for r in old};rows=old
        for day,times,p,a in source_days(cohort):
            rank=np.empty(24);rank[np.argsort(a,kind='stable')]=np.sort(p)
            variants=dict(point_baseline=p,perfect_daily_mean=p+a.mean()-p.mean(),perfect_order_only=rank,perfect_shape=a-a.mean()+p.mean(),perfect_prices=a)
            ev=EV(degradation=5);og=dispatch('G2V',a,times,None,ev);ov=dispatch('general',a,times,None,ev)
            for name,price in variants.items():
                if (day,name) in done:continue
                g=dispatch('G2V',price,times,None,ev);v=dispatch('general',price,times,None,ev);gc=cost(a,g,ev);vc=cost(a,v,ev)
                rows.append(dict(cohort=cohort,day=day,information=name,is_future_prediction=False,actual_G2V_cost=gc['net_cost'],actual_V2G_cost=vc['net_cost'],G2V_regret=gc['net_cost']-og['net_cost'],V2G_regret=vc['net_cost']-ov['net_cost'],export=vc['grid_export_kwh'],gain=gc['net_cost']-vc['net_cost'],ending_soc=v['ending_soc']))
            if len(rows)%70==0:pd.DataFrame(rows).to_csv(dest,index=False)
        pd.DataFrame(rows).to_csv(dest,index=False)
    f=pd.concat([csv(folder/(cohort+'.csv')) for cohort in ['selection','audit']]);f.groupby(['cohort','information']).agg(days=('day','size'),G2V_cost=('actual_G2V_cost','mean'),V2G_cost=('actual_V2G_cost','mean'),G2V_regret=('G2V_regret','mean'),V2G_regret=('V2G_regret','mean'),export=('export','mean'),additional_gain=('gain','mean')).reset_index().to_csv(ROOT/'docs/V16_INFORMATION_ORACLES.csv',index=False)
    print('Information upper bounds completed; not forecasts',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--information-only',action='store_true');a=p.parse_args()
    if not a.information_only:sensitivity()
    information_oracles()
