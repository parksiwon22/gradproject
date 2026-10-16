"""Frozen-cohort reporting and a predeclared policy audit; no candidate reselection."""
import json,hashlib
import joblib,pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v11.store import ROOT,connect,latest,now
from v14.research import HOME,ev_metrics,specifications,write,predict,scored
from v14.features import augment

def run():
    folder=HOME/'study'; selection=json.loads((folder/'selection_before_audit.json').read_text()); top=selection['top3']
    val=pd.read_csv(folder/'selection_metrics.csv'); audit=pd.read_csv(folder/'audit_metrics.csv'); policies=pd.read_csv(folder/'selection_policy_metrics.csv')
    choice=policies[policies.name==top[0]].sort_values('V2G_actual').iloc[0]
    frozen=HOME/'policy_selection_before_policy_audit.json'
    selected=dict(at=now(),name=str(choice['name']),policy=str(choice['policy']),criterion='lowest2025Q4 V2G actual cost within frozen best price candidate; no2026 selection',audit_previously_seen=True)
    if not frozen.exists(): write(frozen,selected)
    else: selected=json.loads(frozen.read_text())
    pred=pd.read_csv(folder/'audit_predictions.csv'); pred.timestamp=pd.to_datetime(pred.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
    rows=[]
    for name,policy in [('V12_hgb_baseline',selected['policy']),(selected['name'],selected['policy'])]:
        summary,daily=ev_metrics(pred,pred[name],policy); rows.append(dict(name=name,policy=policy,**summary)); daily.to_csv(folder/(name+'.policy_audit_'+policy+'.csv'),index=False)
    pd.DataFrame(rows).to_csv(folder/'policy_audit_metrics.csv',index=False)
    db=connect(); raw=latest(db,'land'); db.close(); f,groups,provenance=augment(raw)
    # Reproducible local design matrix, never Git tracked.
    f.to_csv(folder/'retrospective_features.csv',index=False)
    diagnosis=f[(f.target_date>='2026-10-01')&(f.target_date<='2026-10-15')].reset_index(drop=True)
    diag=diagnosis[['timestamp','target_date','actual_smp','previous_actual_smp']].copy(); diagrows=[]
    for name in ['V12_hgb_baseline']+top:
        p=predict(joblib.load(folder/(name+'.joblib')),diagnosis); diag[name]=p; diagrows.append(dict(name=name,interpretation='previously_seen_known_failure_diagnosis_only',**scored(diagnosis,p,241.63)))
    diag.to_csv(folder/'known_october_predictions.csv',index=False); pd.DataFrame(diagrows).to_csv(folder/'known_october_metrics.csv',index=False)
    feature_inventory=[]
    for group,cols in groups.items():
        for col in cols: feature_inventory.append(dict(group=group,feature=col,missing_rate=float(f[col].isna().mean()),origin='D-2 23h',operational_eligible=group=='price',historical_receipt_certified=False))
    pd.DataFrame(feature_inventory).to_csv(folder/'feature_inventory.csv',index=False)
    weather=pd.read_csv(ROOT/'artifacts/v11/kma_apihub/observations_period_hourly.csv'); weather.timestamp=pd.to_datetime(weather.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
    inventory=[]
    for (year,station),g in weather.groupby([weather.timestamp.dt.year,'station_id']):
        row=dict(dataset='ASOS_hourly',year=int(year),station=int(station),rows=len(g),historical_issue_vintage=False)
        for c in ['temperature_c','humidity_percent','wind_ms','solar_radiation_mj_m2','sunshine_hours','rainfall_mm']: row[c+'_missing_rate']=float(g[c].isna().mean())
        inventory.append(row)
    for year,g in raw.groupby(raw.timestamp.dt.year): inventory.append(dict(dataset='KPX_land_SMP',year=int(year),station=None,rows=len(g),historical_issue_vintage=False))
    pd.DataFrame(inventory).to_csv(ROOT/'docs/V14_DATA_INVENTORY.csv',index=False)
    detailed=[]
    for p in folder.glob('*.joblib'):
        b=joblib.load(p); params={}
        for key in ('model','low','high'):
            if key in b and hasattr(b[key],'get_params'): params[key]=b[key].get_params()
        detailed.append(dict(name=p.stem,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),spec=b.get('spec'),features=b.get('features'),parameters=params,architecture=b.get('architecture'),seed=42))
    write(folder/'model_parameters.json',detailed)
    base=val[val.name=='V12_hgb_baseline'].iloc[0]
    findings=[]
    for _,r in val.iterrows():
        findings.append(dict(name=r['name'],MAE_improved=bool(r.MAE<base.MAE),peak_MAE_improved=bool(r.peak_MAE<base.peak_MAE),G2V_improved=bool(r.G2V_regret<base.G2V_regret),V2G_improved=bool(r.V2G_regret<base.V2G_regret),F1_improved=bool(r.peak_F1>base.peak_F1),explanation='Tradeoffs retained; no promotion based on retrospectively inspected labels'))
    pd.DataFrame(findings).to_csv(folder/'hypothesis_outcomes.csv',index=False)
    figs=ROOT/'docs/v14_figures'; figs.mkdir(exist_ok=True)
    # Audit peak dates are diagnostic only after candidate freeze.
    peakdays=pred.groupby('target_date').actual_smp.max().nlargest(3).index
    for day in peakdays:
        g=pred[pred.target_date==day]; fig,ax=plt.subplots(figsize=(10,4))
        ax.plot(g.timestamp.dt.hour,g.actual_smp,label='Official actual',color='black',linewidth=2.5)
        for name in ['V12_hgb_baseline']+top: ax.plot(g.timestamp.dt.hour,g[name],label=name)
        ax.axhline(150,color='red',linestyle=':',label='150 KRW/kWh'); ax.set(xlabel='KST hour',ylabel='SMP KRW/kWh',title='Previously seen audit / '+day); ax.legend(fontsize=7,ncol=2); fig.tight_layout(); fig.savefig(figs/('peak_'+day+'.png'),dpi=150); plt.close(fig)
    for day in diag.groupby('target_date').actual_smp.max().nlargest(1).index:
        g=diag[diag.target_date==day]; fig,ax=plt.subplots(figsize=(10,4)); ax.plot(g.timestamp.dt.hour,g.actual_smp,color='black',linewidth=2.5,label='Official actual')
        for name in ['V12_hgb_baseline']+top: ax.plot(g.timestamp.dt.hour,g[name],label=name)
        ax.axhline(150,color='red',linestyle=':'); ax.set(xlabel='KST hour',ylabel='SMP KRW/kWh',title='Known October failure diagnosis only / '+day); ax.legend(fontsize=7,ncol=2); fig.tight_layout(); fig.savefig(figs/'known_october_failure.png',dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5)); ax.scatter(val.MAE,val.V2G_regret,c=np.where(val.operational_candidate,'#276baa','#c77f34'),s=35)
    for name in ['V12_hgb_baseline']+top:
        r=val[val.name==name].iloc[0]; ax.annotate(name,(r.MAE,r.V2G_regret),fontsize=7)
    ax.set(xlabel='MAE KRW/kWh',ylabel='V2G regret virtual KRW/day',title='2025Q4 selection / blue=price-only, orange=retrospective exogenous'); fig.tight_layout(); fig.savefig(figs/'price_vs_decision.png',dpi=150); plt.close(fig)
    for source,target in [('selection_metrics.csv','V14_ALL_SELECTION_METRICS.csv'),('audit_metrics.csv','V14_AUDIT_METRICS.csv'),('selection_policy_metrics.csv','V14_POLICY_SELECTION.csv'),('policy_audit_metrics.csv','V14_POLICY_AUDIT.csv'),('hypothesis_outcomes.csv','V14_HYPOTHESIS_OUTCOMES.csv')]:
        (ROOT/'docs'/target).write_bytes((folder/source).read_bytes())
    summary=dict(new_price_candidates=len(val)-1,trained_configurations=len(specifications()),blends=2,internal_fitted_estimators=len(specifications())+1,deep_learning_pilot='one GRU,6epochs,CPU2; no expansion due worse validation',extra_dispatch_policy_conditions=4,top3=top,selected_policy=selected,operating_model_replaced=False,source_official_lng_values=6,new_forecast_candidates_currently_waiting_for_next_eligible_origin=True)
    follow=HOME/'chronological_followup/metrics.csv'
    if follow.exists():
        extra=pd.read_csv(follow); pd.concat([val,extra],ignore_index=True).to_csv(ROOT/'docs/V14_ALL_SELECTION_METRICS.csv',index=False)
        summary.update(new_price_candidates=len(val),trained_configurations=len(specifications())+1,internal_fitted_estimators=len(specifications())+2,deep_learning_pilot='Two bounded GRU pilots; second uses true chronological daily lags and calendar; both failed validation, no audit/promotion for followup')
    write(HOME/'summary.json',summary)
    print(json.dumps(summary))

if __name__=='__main__': run()
