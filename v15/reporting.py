"""Frozen candidate audits, scenario diagnostics and public aggregate tables."""
import json,hashlib
from pathlib import Path
import joblib,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v11.store import ROOT,now
from v14.research import predict as predict14,ev_metrics,scored
from v12.ev import EV,score
from v15.research import HOME,predict,write
from v15.economics import summary

def run():
    folder=HOME/'diagnostics'; folder.mkdir(exist_ok=True); figs=ROOT/'docs/v15_figures'; figs.mkdir(exist_ok=True)
    cv=pd.concat([pd.read_csv(HOME/'model_study/cross_validation.csv', float_precision='round_trip'),pd.read_csv(HOME/'shrinkage_followup/cross_validation.csv', float_precision='round_trip')],ignore_index=True)
    cv.to_csv(ROOT/'docs/V15_MODEL_CV.csv',index=False)
    qualifications=[]
    for path in [HOME/'model_study/qualification.json',HOME/'shrinkage_followup/qualification.json']: qualifications.extend(json.loads(path.read_text())['results'])
    pd.DataFrame(qualifications).to_csv(ROOT/'docs/V15_MODEL_QUALIFICATION.csv',index=False)
    f=pd.read_csv(HOME/'model_study/frozen_matrix.csv', float_precision='round_trip'); f.timestamp=pd.to_datetime(f.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
    models={'baseline':joblib.load(HOME/'model_study/winter2025_baseline.joblib')}
    models.update({p.stem:joblib.load(p) for p in (HOME/'candidates').glob('*.joblib')})
    auditrows=[]
    for cohort,dates in [('seen_2026H1',['2026-01-01','2026-06-30']),('known_October_failure',['2026-10-01','2026-10-15'])]:
        data=f[(f.target_date>=dates[0])&(f.target_date<=dates[1])].reset_index(drop=True); out=data[['timestamp','target_date','actual_smp','previous_actual_smp','daily']].copy()
        for name,b in models.items():
            p=predict(b,data); out[name]=p; metric=scored(data,p,241.63)
            if cohort=='seen_2026H1':
                ev,daily=ev_metrics(data,p); metric.update(ev); daily.to_csv(folder/(cohort+'_'+name+'_ev.csv'),index=False)
            auditrows.append(dict(model=name,cohort=cohort,**metric))
        out.to_csv(folder/(cohort+'_predictions.csv'),index=False)
        highday=out.groupby('target_date').actual_smp.max().idxmax(); g=out[out.target_date==highday]; fig,ax=plt.subplots(figsize=(10,4))
        ax.plot(g.timestamp.dt.hour,g.actual_smp,color='black',linewidth=2.5,label='Official actual')
        for name in models: ax.plot(g.timestamp.dt.hour,g[name],label=name)
        ax.axhline(150,color='red',linestyle=':'); ax.set(xlabel='KST hour',ylabel='SMP KRW/kWh',title=cohort+' / '+highday); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(figs/(cohort+'.png'),dpi=150); plt.close(fig)
    pd.DataFrame(auditrows).to_csv(ROOT/'docs/V15_MODEL_AUDIT.csv',index=False)
    # Extra summer stress uses models trained by2024Dec/calibrated by2025Mar.
    summer=f[(f.target_date>='2025-07-01')&(f.target_date<='2025-09-30')].reset_index(drop=True); summerrows=[]
    base=joblib.load(HOME/'model_study/spring2025_baseline.joblib'); event=joblib.load(HOME/'model_study/spring2025_relative_two_stage.joblib')
    for label,weight in [('baseline',0),('two_stage_shrink_0.25',.25),('two_stage_shrink_0.5',.5)]:
        bundle=base if weight==0 else dict(kind='blend',weight=weight,base=base,event=event)
        p=predict(bundle,summer); summerrows.append(dict(model=label,role='post_selection_seasonal_stress_not_new_test',**scored(summer,p,241.63)))
    pd.DataFrame(summerrows).to_csv(ROOT/'docs/V15_SUMMER_STRESS.csv',index=False)
    # Fill comparable point and scenario-mean expected benefits using saved plans;
    # no new policy tuning or altered actual outcomes.
    summaries=[]
    for path in (HOME/'policy_study').glob('*.csv'):
        if not path.name.startswith(('selection_','audit_')): continue
        data=pd.read_csv(path, float_precision='round_trip')
        if data.empty or 'policy' not in data: continue
        cohort=path.name.split('_')[0]; name=data.model.iloc[0]; policy=data.policy.iloc[0]; templates=np.load(HOME/'policy_study'/(name+'_templates.npy'))
        gains=[]
        for _,row in data.iterrows():
            own=json.loads((HOME/'policy_study/plans'/cohort/name/policy/(row.day+'.json')).read_text()); ref=json.loads((HOME/'policy_study/plans'/cohort/name/'G2V'/(row.day+'.json')).read_text())
            s=pd.DataFrame(own['schedule']); b=pd.DataFrame(ref['schedule']); mean=s.price.to_numpy()+templates.mean(axis=0); ev=EV(degradation=5)
            gain=score(mean,b.charge_kw.to_numpy(),b.discharge_kw.to_numpy(),b.connected_hours.to_numpy(),ev)['net_cost']-score(mean,s.charge_kw.to_numpy(),s.discharge_kw.to_numpy(),s.connected_hours.to_numpy(),ev)['net_cost']; gains.append(gain)
        data['scenario_mean_expected_gain']=gains; data.to_csv(path,index=False)
        result=dict(cohort=cohort,model=name,policy=policy,**summary(data.to_dict('records')),scenario_mean_expected_gain=float(data.scenario_mean_expected_gain.mean()),loss_over50_days=int((data.actual_gain< -50-1e-6).sum()),worst_actual_additional_loss=float(max(0,-data.actual_gain.min())),zero_export_mean_gain=float(data.loc[data.export_kwh<.05,'actual_gain'].mean()) if (data.export_kwh<.05).any() else None)
        summaries.append(result)
    pd.DataFrame(summaries).to_csv(ROOT/'docs/V15_POLICY_COMPARISON.csv',index=False); write(HOME/'policy_comparison.json',summaries)
    for filename in ['year_incidence.csv','type_inventory.csv','type_prediction_scores.csv','known_six_analogs.csv']:
        target=ROOT/'docs'/('V15_'+filename.upper()); target.write_bytes((HOME/'events'/filename).read_bytes())
    p=pd.DataFrame(summaries); g=p[(p.cohort=='audit')&(p.model=='V12_hgb_baseline')]
    fig,ax=plt.subplots(figsize=(9,5)); ax.scatter(g.mean_export_kwh,g.mean_actual_additional_profit,s=65)
    for _,r in g.iterrows(): ax.annotate(r.policy,(r.mean_export_kwh,r.mean_actual_additional_profit),fontsize=8)
    ax.axhline(0,color='black'); ax.axvline(.5,color='gray',linestyle=':'); ax.set(xlabel='Mean grid export kWh/day',ylabel='Realized incremental benefit vs G2V (virtual KRW/day)',title='Previously seen2026H1 / same model and terminal SOC'); fig.tight_layout(); fig.savefig(figs/'activity_vs_profit.png',dpi=150); plt.close(fig)
    write(HOME/'report_summary.json',dict(at=now(),new_model_families=4,fit_configurations=12,followup_blend_configurations=2,fold_evaluations=18,qualified_candidates=[r['name'] for r in qualifications if r['qualified']],policy_models=2,policy_count=7,operating_model_replaced=False,policy_recommendation='G2V; no active policy met preregistered validation criteria'))
    print('V15 tables, candidate diagnostics and scenario profit decomposition saved',flush=True)

if __name__=='__main__': run()
