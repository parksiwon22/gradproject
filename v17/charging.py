"""Exact fixed-energy G2V priority dispatch; shared V12 physics and settlement."""
from dataclasses import asdict
import numpy as np,pandas as pd
from v12.ev import EV,validate,durations,score
PATTERNS=[('full24',0,24),('night18_08',18,14),('day09_18',9,9),('short3',12,3),('short4',12,4),('short5',12,5),('long8',22,8),('long12',20,12)]

def priority_plan(prices,times,priority=None,ev=None,start=None,end=None):
    ev=ev or EV(degradation=5);validate(ev);p=np.asarray(prices,dtype=float);dt=durations(times,start,end);rank=p if priority is None else np.asarray(priority,dtype=float)
    if len(p)!=len(dt) or rank.shape!=p.shape or not np.isfinite(p).all() or not np.isfinite(rank).all():raise ValueError('Complete finite planning prices and priorities required')
    need=max(0,ev.capacity*(ev.target_soc-ev.initial_soc)/100)/ev.charge_eff
    if need>ev.charge_kw*dt.sum()+1e-8:raise ValueError('충전 가능 시간과 최대전력으로 목표 SOC를 달성할 수 없습니다')
    c=np.zeros(len(p));remaining=need
    for i in np.argsort(rank,kind='stable'):
        if dt[i]<=0:continue
        c[i]=min(ev.charge_kw,remaining/dt[i]);remaining=max(0,remaining-c[i]*dt[i])
    energy=ev.capacity*ev.initial_soc/100+np.r_[0,np.cumsum(c*dt*ev.charge_eff)]
    s=pd.DataFrame(dict(timestamp=pd.DatetimeIndex(times),connected_hours=dt,price=p,priority=rank,charge_kw=c,discharge_kw=0.,soc_start=energy[:-1]/ev.capacity*100,soc_end=energy[1:]/ev.capacity*100))
    if (s.soc_end>ev.max_soc+1e-7).any() or (s.soc_end<ev.min_soc-1e-7).any():raise ValueError('SOC bounds violated')
    return dict(kind='G2V_PRIORITY_SIMULATION',assumptions=asdict(ev),ending_soc=float(s.soc_end.iloc[-1]),schedule=s,**score(p,c,np.zeros(len(c)),dt,ev))

def require_prices(frame,start,end):
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    if start.tzinfo is None or end.tzinfo is None or end<=start:raise ValueError('시간대가 있는 올바른 연결 시작·종료시각이 필요합니다')
    times=pd.date_range(start.floor('h'),end.ceil('h'),freq='h',inclusive='left')
    selected=frame.set_index('timestamp').reindex(times)
    if selected.index.has_duplicates or selected.price.isna().any():raise ValueError('연결기간 전체의 가격이 없습니다. 다음 날짜 가격을 복제하거나 임의 생성하지 않습니다')
    selected=selected.reset_index(names='timestamp');return selected
