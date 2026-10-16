import numpy as np,pandas as pd
from v14.features import augment
EVENT=['ramp_mean24','ramp_abs_max24','up15_fraction168','down15_fraction168','range_ratio24_168','origin_jump3','origin_jump6','profile_amplitude_ratio','known_tail_gap']

def build(prices,targets=None,include_labels=True):
    f,groups,_=augment(prices,external=False,targets=targets,include_labels=include_labels)
    s=prices.set_index('timestamp').smp.sort_index(); ds=s.diff(); origin=pd.DatetimeIndex(f.origin_timestamp)
    def at(x): return x.reindex(origin).to_numpy()
    f['ramp_mean24']=at(ds.rolling(24).mean()); f['ramp_abs_max24']=at(ds.abs().rolling(24).max())
    f['up15_fraction168']=at((ds>=15).rolling(168).mean()); f['down15_fraction168']=at((ds<=-15).rolling(168).mean())
    f['range_ratio24_168']=f.recent_range/(f.std168+5)
    f['origin_jump3']=f.origin_price.to_numpy()-s.reindex(origin-pd.Timedelta(hours=3)).to_numpy()
    f['origin_jump6']=f.origin_price.to_numpy()-s.reindex(origin-pd.Timedelta(hours=6)).to_numpy()
    f['profile_amplitude_ratio']=f.profile_std/(f.std168+5); f['known_tail_gap']=f.daily-(f.mean168+1.5*f.std168)
    return f,groups['price']+EVENT

def event_labels(f):
    # Fixed relative-uplift target, not tuned to October's actual six hours.
    return ((f.actual_smp-f.daily)>=20).to_numpy()
