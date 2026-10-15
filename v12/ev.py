"""Hourly SMP-linked G2V LP and V2G MILP (research settlement only)."""
from dataclasses import dataclass,asdict
import numpy as np
import pandas as pd
from scipy.optimize import milp,Bounds,LinearConstraint

@dataclass
class EV:
    capacity:float=60
    initial_soc:float=30
    target_soc:float=80
    charge_kw:float=7
    discharge_kw:float=7
    charge_eff:float=.9
    discharge_eff:float=.9
    min_soc:float=20
    max_soc:float=90
    degradation:float=0
    sell_ratio:float=1

def durations(timestamps,start=None,end=None):
    t=pd.DatetimeIndex(timestamps)
    if t.empty or t.tz is None or t.has_duplicates or not t.is_monotonic_increasing: raise ValueError('unique sorted timezone-aware hours required')
    if len(t)>1 and not ((t[1:]-t[:-1])==pd.Timedelta(hours=1)).all(): raise ValueError('contiguous hourly intervals required')
    start=pd.Timestamp(start) if start is not None else t[0]
    end=pd.Timestamp(end) if end is not None else t[-1]+pd.Timedelta(hours=1)
    if start.tzinfo is None or end.tzinfo is None or end<=start: raise ValueError('valid timezone-aware connection window required')
    if start<t[0] or end>t[-1]+pd.Timedelta(hours=1): raise ValueError('prices must cover entire connection window')
    return np.array([max(0,(min(x+pd.Timedelta(hours=1),end)-max(x,start)).total_seconds()/3600) for x in t])

def validate(ev):
    values=list(asdict(ev).values())
    if not np.isfinite(values).all(): raise ValueError('finite EV parameters required')
    if ev.capacity<=0 or ev.charge_kw<=0 or ev.discharge_kw<0 or not 0<ev.charge_eff<=1 or not 0<ev.discharge_eff<=1: raise ValueError('invalid capacity, power or efficiency')
    if not 0<=ev.min_soc<=ev.initial_soc<=ev.max_soc<=100 or not ev.min_soc<=ev.target_soc<=ev.max_soc: raise ValueError('SOC bounds violated')
    if ev.degradation<0 or not 0<=ev.sell_ratio<=1: raise ValueError('invalid settlement assumptions')

def score(prices,c,d,dt,ev):
    buy=float(np.dot(prices,c*dt)); credit=float(np.dot(prices,d*dt)*ev.sell_ratio)
    wear=float(ev.degradation*np.sum((ev.charge_eff*c+d/ev.discharge_eff)*dt))
    return dict(purchase_cost=buy,export_credit=credit,degradation_cost=wear,net_cost=buy-credit+wear,grid_import_kwh=float(np.sum(c*dt)),grid_export_kwh=float(np.sum(d*dt)))

def optimize(prices,timestamps,ev=None,start=None,end=None,v2g=False):
    ev=ev or EV(); validate(ev)
    p=np.asarray(prices,dtype=float); dt=durations(timestamps,start,end); n=len(p)
    if n!=len(dt) or not np.isfinite(p).all(): raise ValueError('complete finite prices required')
    e0=ev.capacity*ev.initial_soc/100; target=ev.capacity*ev.target_soc/100
    if (max(0,target-e0)>ev.charge_kw*ev.charge_eff*dt.sum()+1e-8): raise ValueError('departure SOC infeasible in connection window')
    # Variables: grid charge[n], grid export[n], battery energy[n+1], mode[n].
    size=4*n+1; objective=np.zeros(size)
    objective[:n]=dt*(p+ev.degradation*ev.charge_eff)
    objective[n:2*n]=dt*(-ev.sell_ratio*p+ev.degradation/ev.discharge_eff)
    lower=np.zeros(size); upper=np.full(size,np.inf)
    upper[:n]=ev.charge_kw*(dt>0); upper[n:2*n]=(ev.discharge_kw if v2g else 0)*(dt>0)
    lower[2*n:3*n+1]=ev.capacity*ev.min_soc/100; upper[2*n:3*n+1]=ev.capacity*ev.max_soc/100
    lower[2*n]=upper[2*n]=e0
    lower[3*n]=target if v2g else max(target,e0)
    if not v2g: upper[3*n]=max(target,e0)
    upper[3*n+1:]=1
    integrality=np.zeros(size); integrality[3*n+1:]=1 if v2g else 0
    constraints=[]; lows=[]; highs=[]
    for i in range(n):
        row=np.zeros(size); row[2*n+i+1]=1; row[2*n+i]=-1; row[i]=-ev.charge_eff*dt[i]; row[n+i]=dt[i]/ev.discharge_eff
        constraints.append(row); lows.append(0); highs.append(0)
        row=np.zeros(size); row[i]=1; row[3*n+1+i]=-ev.charge_kw
        constraints.append(row); lows.append(-np.inf); highs.append(0)
        row=np.zeros(size); row[n+i]=1; row[3*n+1+i]=ev.discharge_kw
        constraints.append(row); lows.append(-np.inf); highs.append(ev.discharge_kw)
    result=milp(objective,integrality=integrality,bounds=Bounds(lower,upper),constraints=LinearConstraint(np.array(constraints),lows,highs),options={'time_limit':20,'mip_rel_gap':1e-8})
    if not result.success: raise ValueError('optimization failed: '+str(result.message))
    c=result.x[:n]; d=result.x[n:2*n]; energy=result.x[2*n:3*n+1]
    if np.max(np.minimum(c,d))>1e-6: raise ValueError('simultaneous charging and discharging')
    frame=pd.DataFrame(dict(timestamp=pd.DatetimeIndex(timestamps),connected_hours=dt,price=p,charge_kw=c,discharge_kw=d,soc_start=energy[:-1]/ev.capacity*100,soc_end=energy[1:]/ev.capacity*100))
    return dict(kind='V2G_SIMULATION' if v2g else 'G2V',assumptions=asdict(ev),ending_soc=float(energy[-1]/ev.capacity*100),schedule=frame,**score(p,c,d,dt,ev))

def baseline(prices,timestamps,ev,start=None,end=None,uniform=False):
    validate(ev); dt=durations(timestamps,start,end)
    remaining=max(0,ev.capacity*(ev.target_soc-ev.initial_soc)/100)/ev.charge_eff
    if remaining>ev.charge_kw*dt.sum()+1e-8: raise ValueError('infeasible baseline')
    c=np.zeros(len(dt))
    if uniform:
        c[dt>0]=remaining/dt.sum() if dt.sum() else 0
    else:
        for i,h in enumerate(dt):
            if h: c[i]=min(ev.charge_kw,remaining/h); remaining-=c[i]*h
    return dict(kind='uniform' if uniform else 'immediate',**score(np.asarray(prices),c,np.zeros(len(c)),dt,ev))

def regret(plan,actual,timestamps,ev,start=None,end=None,v2g=False):
    f=plan['schedule']; realized=score(np.asarray(actual),f.charge_kw.to_numpy(),f.discharge_kw.to_numpy(),f.connected_hours.to_numpy(),ev)
    oracle=optimize(actual,timestamps,ev,start,end,v2g)
    return dict(realized_cost=realized['net_cost'],oracle_cost=oracle['net_cost'],regret=realized['net_cost']-oracle['net_cost'])
