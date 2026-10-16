"""Cutoff-safe event-time feature construction; historical receipt limits explicit."""
import numpy as np
import pandas as pd
from v11.features import build,FEATURES
from v11.store import ROOT

PRICE_EXTRA=['profile_mean','profile_std','profile_rank','relative_level','recent_range','recent_cv','weekly_gap','hourly_shape_change','origin_downtrend','origin_week_change','past_negative_fraction','past_high_fraction']

def augment(prices,external=True,targets=None,include_labels=True):
    f=build(prices,days_ahead=2,targets=targets,include_labels=include_labels); s=prices.set_index('timestamp').smp.sort_index()
    profiles=f.groupby('target_date').daily
    f['profile_mean']=profiles.transform('mean'); f['profile_std']=profiles.transform('std')
    f['profile_rank']=profiles.rank(pct=True); f['relative_level']=f.daily-f.profile_mean
    f['recent_range']=f.max24-f.min24; f['recent_cv']=f.std168/(abs(f.mean168)+1)
    f['weekly_gap']=f.daily-f.weekly
    target=pd.DatetimeIndex(f.timestamp); origin=pd.DatetimeIndex(f.origin_timestamp)
    f['hourly_shape_change']=f.daily.to_numpy()-s.reindex(target-pd.Timedelta(hours=49)).to_numpy()
    f['origin_downtrend']=(f.change24<0).astype(int); f['origin_week_change']=f.origin_price-f.origin_lag168
    for name,condition in [('past_negative_fraction',s<0),('past_high_fraction',s>=150)]:
        f[name]=condition.rolling(168,min_periods=168).mean().reindex(origin).to_numpy()
    for lag in range(6,-1,-1):
        f['sequence_origin_lag'+str(lag*24)]=s.reindex(origin-pd.Timedelta(hours=lag*24)).to_numpy()
    groups={'price':FEATURES+PRICE_EXTRA}; provenance=[]
    if external:
        base=ROOT/'artifacts/v11'; demand=pd.read_csv(base/'exogenous/demand_hourly_all.csv')
        demand.timestamp=pd.to_datetime(demand.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
        ds=demand.drop_duplicates('timestamp').set_index('timestamp').demand_mwh
        f['demand_lag168']=ds.reindex(target-pd.Timedelta(days=7)).to_numpy()
        f['demand_origin_lag168']=ds.reindex(origin-pd.Timedelta(days=7)).to_numpy()
        groups['demand']=groups['price']+['demand_lag168','demand_origin_lag168']
        weather=pd.read_csv(base/'kma_apihub/observations_period_hourly.csv')
        weather.timestamp=pd.to_datetime(weather.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
        cols=['temperature_c','humidity_percent','wind_ms','solar_radiation_mj_m2','sunshine_hours','rainfall_mm']
        lagged=weather.groupby('timestamp')[cols].agg(['mean','std','count'])
        lagged.columns=['asos_'+a+'_'+b+'_lag48' for a,b in lagged.columns]
        w=lagged.reindex(target-pd.Timedelta(hours=48)).reset_index(drop=True)
        for c in w: f[c]=w[c].to_numpy()
        groups['weather']=groups['price']+list(w.columns)
        power=pd.read_csv(base/'exogenous/power_daily_all.csv')
        power.date=pd.to_datetime(power.date,utc=True).dt.tz_convert('Asia/Seoul')
        ps=power.set_index('date')[['supply_capacity_mw','reserve_mw','reserve_rate_percent']]
        supply=ps.reindex(origin.normalize()-pd.Timedelta(days=7)).reset_index(drop=True)
        supply.columns=[c+'_lag7days' for c in supply]
        for c in supply: f[c]=supply[c].to_numpy()
        groups['supply']=groups['price']+list(supply.columns)
        groups['combined']=list(dict.fromkeys(groups['weather']+groups['demand']+groups['supply']))
        provenance=[dict(group=g,availability='retrospective event-time lag only; historical publication/receipt vintage not established',operational=False) for g in ['weather','demand','supply']]
    return f,groups,provenance

def safe_event_lags(target,origin):
    return dict(weather=pd.Timestamp(target)-pd.Timedelta(hours=48)<=pd.Timestamp(origin),demand=pd.Timestamp(target)-pd.Timedelta(days=7)<=pd.Timestamp(origin),supply=pd.Timestamp(origin).normalize()-pd.Timedelta(days=7)<=pd.Timestamp(origin))
