"""D-2/D-3 shared-origin price inputs; separate from preserved V11 builder."""
import numpy as np
import pandas as pd

FEATURES=['hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos','weekend','daily','weekly','spread','origin_price','origin_lag1','origin_lag24','origin_lag168','mean24','std24','min24','max24','mean168','std168','change24']

def build(prices,targets=None,days_ahead=1,include_labels=True):
    if days_ahead not in (2,3): raise ValueError('V17 supported 23h horizons:2 or3 days')
    s=prices.set_index('timestamp').smp.sort_index()
    if s.index.tz is None or s.index.duplicated().any(): raise ValueError('timezone and unique hours required')
    if targets is None: targets=s.index
    targets=pd.DatetimeIndex(targets)
    origin=targets.normalize()-pd.Timedelta(days=days_ahead)+pd.Timedelta(hours=23)
    def at(times): return s.reindex(times).to_numpy(dtype=float)
    out=pd.DataFrame({'timestamp':targets,'origin_timestamp':origin,'target_date':targets.strftime('%Y-%m-%d')})
    for name,values,period in [('hour',targets.hour,24),('dow',targets.dayofweek,7),('month',targets.month,12)]:
        out[name+'_sin']=np.sin(2*np.pi*values/period); out[name+'_cos']=np.cos(2*np.pi*values/period)
    out['weekend']=(targets.dayofweek>=5).astype(int)
    out['days_ahead']=days_ahead
    out['daily']=at(targets-pd.Timedelta(days=days_ahead))
    out['weekly']=at(targets-pd.Timedelta(days=7))
    out['spread']=out.daily-out.weekly
    out['origin_price']=at(origin)
    for lag in (1,24,168): out['origin_lag'+str(lag)]=at(origin-pd.Timedelta(hours=lag))
    for window in (24,168):
        for statistic in (['mean','std','min','max'] if window==24 else ['mean','std']):
            out[statistic+str(window)]=getattr(s.rolling(window,min_periods=window),statistic)().reindex(origin).to_numpy()
    out['change24']=out.origin_price-out.origin_lag24
    if include_labels:
        out['actual_smp']=at(targets); out['previous_actual_smp']=at(targets-pd.Timedelta(hours=1))
    needed=FEATURES+(['actual_smp','previous_actual_smp'] if include_labels else [])
    out=out.replace([np.inf,-np.inf],np.nan).dropna(subset=needed)
    complete=out.groupby('target_date').size()
    return out[out.target_date.isin(complete[complete==24].index)].reset_index(drop=True)
