"""Diagnostic-only peak anatomy and fixed EV backtests; no model selection."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import brier_score_loss,average_precision_score,roc_auc_score
from v11.store import ROOT,connect,latest,now
from v11.features import build,FEATURES
from v11.exogenous_model import read_date
from v12.research import HOME,risk_predict
from v12.ev import EV,optimize,baseline,regret

def run():
    folder=HOME/'analysis'; folder.mkdir(parents=True,exist_ok=True)
    db=connect(); prices=latest(db,'land'); db.close(); series=prices.set_index('timestamp').smp
    old=ROOT/'artifacts/v11'; prior=read_date(old/'kma_apihub/hourly_study/test_predictions.csv','timestamp')
    spikes=prior[prior.actual_smp>=150].copy()
    for hours in (1,24,48,168): spikes['actual_price_lag'+str(hours)]=series.reindex(pd.DatetimeIndex(spikes.timestamp-pd.Timedelta(hours=hours))).to_numpy()
    demand=read_date(old/'exogenous/demand_forecast_hourly.csv','timestamp').set_index('timestamp').day_ahead_demand_mw
    for hours in (0,24): spikes['archived_forecast_demand_lag'+str(hours)]=demand.reindex(pd.DatetimeIndex(spikes.timestamp-pd.Timedelta(hours=hours))).to_numpy()
    weather=read_date(old/'kma_apihub/observations_period_hourly.csv','timestamp')
    w=weather[weather.station_id==108].set_index('timestamp')
    for field in ('temperature_c','humidity_percent','wind_ms','solar_radiation_mj_m2','rainfall_mm'): spikes['DIAGNOSTIC_actual_weather_'+field]=w[field].reindex(pd.DatetimeIndex(spikes.timestamp)).to_numpy()
    power=read_date(old/'exogenous/power_daily_all.csv','date').set_index('date')
    for field in ('daily_max_demand_mw','reserve_mw','reserve_rate_percent'): spikes['DIAGNOSTIC_daily_'+field]=power[field].reindex(pd.DatetimeIndex(spikes.timestamp.dt.normalize())).to_numpy()
    spikes['bias_delayed_weather']=spikes.OBS_SMP_delayed_ASOS_NOT_FORECAST-spikes.actual_smp
    spikes.to_csv(folder/'known_six_peak_hours.csv',index=False)
    yearly=[]
    for (year,quarter),g in prices.groupby([prices.timestamp.dt.year,prices.timestamp.dt.quarter]):
        yearly.append(dict(year=int(year),quarter=int(quarter),rows=len(g),mean=float(g.smp.mean()),q95=float(g.smp.quantile(.95)),peak_hours_150=int((g.smp>=150).sum()),peak_fraction_150=float((g.smp>=150).mean())))
    pd.DataFrame(yearly).to_csv(folder/'seasonal_peak_incidence.csv',index=False)
    base=build(prices); train=base[base.target_date<'2026-04-01']; cases=base[base.timestamp.isin(spikes.timestamp)]
    scaler=StandardScaler().fit(train[FEATURES]); nearest=NearestNeighbors(n_neighbors=10).fit(scaler.transform(train[FEATURES])); distances,indices=nearest.kneighbors(scaler.transform(cases[FEATURES]))
    analogs=[]
    for t,ds,ids in zip(cases.timestamp,distances,indices):
        for distance,i in zip(ds,ids): analogs.append(dict(case_timestamp=t,analog_timestamp=train.iloc[i].timestamp,distance=float(distance),actual_smp=float(train.iloc[i].actual_smp),daily_profile=float(train.iloc[i].daily)))
    pd.DataFrame(analogs).to_csv(folder/'training_price_feature_analogs.csv',index=False)
    risk=joblib.load(HOME/'study/risk.joblib'); alltwo=build(prices,days_ahead=2)
    val=alltwo[(alltwo.target_date>='2025-10-01')&(alltwo.target_date<='2025-12-31')]; prob=risk_predict(risk,val); label=val.actual_smp>=150
    riskmetrics=dict(Brier=brier_score_loss(label,prob),constant_base_rate_Brier=brier_score_loss(label,np.full(len(label),risk['prior'])),average_precision=average_precision_score(label,prob),ROC_AUC=roc_auc_score(label,prob) if label.nunique()==2 else None,warning='retrospective selection period; prospective calibration unverified')
    (folder/'risk_metrics.json').write_text(json.dumps(riskmetrics,indent=2),encoding='utf-8')
    # Fixed hypothetical daily24-hour connection; not an inferred real EV usage.
    ev=EV(degradation=5); evrows=[]
    pred=read_date(HOME/'study/retrospective_audit_predictions.csv','timestamp')
    names=[c for c in pred if c not in ('timestamp','target_date','actual_smp','previous_actual_smp','peak_probability')]
    for day,g in pred.groupby('target_date',sort=True):
        times=pd.DatetimeIndex(g.timestamp); actual=g.actual_smp.to_numpy()
        oracles={v:optimize(actual,times,ev,v2g=v)['net_cost'] for v in (False,True)}
        for name in names:
            for v2g in (False,True):
                plan=optimize(g[name],times,ev,v2g=v2g); f=plan['schedule']
                from v12.ev import score
                realized=score(actual,f.charge_kw.to_numpy(),f.discharge_kw.to_numpy(),f.connected_hours.to_numpy(),ev)['net_cost']
                evrows.append(dict(day=day,model=name,mode='V2G' if v2g else 'G2V',estimated_net_cost=plan['net_cost'],realized_net_cost=realized,oracle_cost=oracles[v2g],regret=realized-oracles[v2g]))
    pd.DataFrame(evrows).to_csv(folder/'ev_daily_backtest.csv',index=False)
    pd.DataFrame(evrows).groupby(['model','mode'])[['estimated_net_cost','realized_net_cost','oracle_cost','regret']].mean().reset_index().to_csv(folder/'ev_summary.csv',index=False)
    # Concrete official-price example is explicitly a known-price optimization.
    day=prices.timestamp.max().normalize(); g=prices[prices.timestamp.dt.normalize()==day]; times=pd.DatetimeIndex(g.timestamp); start=day+pd.Timedelta(hours=18); end=day+pd.Timedelta(days=1)
    example=dict(at=now(),target_date=day.strftime('%Y-%m-%d'),price_kind='already_acquired_official_SMP_not_AI_forecast',assumptions=ev.__dict__,connection='18:00-24:00 KST',results={})
    for name,v in [('G2V',False),('V2G',True)]:
        plan=optimize(g.smp,times,ev,start,end,v); example['results'][name]={k:value for k,value in plan.items() if k!='schedule'}; plan['schedule'].to_csv(folder/('example_'+name+'.csv'),index=False)
    example['results']['immediate']=baseline(g.smp,times,ev,start,end); example['results']['uniform']=baseline(g.smp,times,ev,start,end,True)
    (folder/'official_EV_example.json').write_text(json.dumps(example,indent=2),encoding='utf-8')
    print('Peak anatomy and',len(evrows),'EV dispatch backtests complete',flush=True)

if __name__=='__main__': run()
