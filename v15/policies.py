"""Shared EV constraints, expected cost plus CVaR or finite-scenario loss cap."""
from dataclasses import asdict
import numpy as np,pandas as pd
from scipy.optimize import milp,Bounds,LinearConstraint
from sklearn.cluster import KMeans
from v12.ev import EV,optimize,durations,validate,score
POLICIES=['G2V','general','v14_conservative','profit100','cvar025','cvar1','scenario_cap50']

def residual_templates(errors):
    errors=np.asarray(errors,dtype=float)
    if errors.ndim!=2 or errors.shape[1]!=24 or len(errors)<20 or not np.isfinite(errors).all(): raise ValueError('At least20 complete prior calibration days required')
    centers=KMeans(n_clusters=6,n_init=10,random_state=42).fit(errors).cluster_centers_
    return np.vstack([np.zeros(24),centers,errors[np.argmax(errors.max(axis=1))],errors[np.argmin(errors.min(axis=1))]])

def scenario_optimize(pred,times,scenarios,ev=None,start=None,end=None,risk_weight=.25,alpha=.8,maxloss=None):
    ev=ev or EV(degradation=5); validate(ev); p=np.asarray(pred,dtype=float); a=np.asarray(scenarios,dtype=float)
    dt=durations(times,start,end); n=len(p); k=len(a)
    if a.shape!=(k,n) or k<2 or not np.isfinite(a).all() or not np.isfinite(p).all(): raise ValueError('Complete finite scenario matrix required')
    if risk_weight<0 or not 0<alpha<1 or maxloss is not None and maxloss<0: raise ValueError('invalid risk parameters')
    base=optimize(p,times,ev,start,end); b=base['schedule']
    reference=np.array([score(x,b.charge_kw.to_numpy(),b.discharge_kw.to_numpy(),dt,ev)['net_cost'] for x in a])
    # charge[n], discharge[n], energy[n+1], binary[n], CVaR z[1], u[k]
    core=4*n+1; size=core+1+k; z=core
    lower=np.zeros(size); upper=np.full(size,np.inf); e0=ev.capacity*ev.initial_soc/100
    upper[:n]=ev.charge_kw*(dt>0); upper[n:2*n]=ev.discharge_kw*(dt>0)
    lower[2*n:3*n+1]=ev.capacity*ev.min_soc/100; upper[2*n:3*n+1]=ev.capacity*ev.max_soc/100
    lower[2*n]=upper[2*n]=e0; lower[3*n]=upper[3*n]=ev.capacity*max(ev.target_soc,ev.initial_soc)/100; upper[3*n+1:core]=1
    integrality=np.zeros(size); integrality[3*n+1:core]=1
    objective=np.zeros(size); mean=a.mean(axis=0)
    objective[:n]=dt*(mean+ev.degradation*ev.charge_eff); objective[n:2*n]=dt*(-mean*ev.sell_ratio+ev.degradation/ev.discharge_eff)
    objective[z]=risk_weight; objective[z+1:]=risk_weight/(k*(1-alpha))
    constraints=[]; lows=[]; highs=[]
    def row(values,lo,hi): constraints.append(values); lows.append(lo); highs.append(hi)
    for i in range(n):
        r=np.zeros(size); r[2*n+i+1]=1; r[2*n+i]=-1; r[i]=-ev.charge_eff*dt[i]; r[n+i]=dt[i]/ev.discharge_eff; row(r,0,0)
        r=np.zeros(size); r[i]=1; r[3*n+1+i]=-ev.charge_kw; row(r,-np.inf,0)
        r=np.zeros(size); r[n+i]=1; r[3*n+1+i]=ev.discharge_kw; row(r,-np.inf,ev.discharge_kw)
    costrows=[]
    for j,price in enumerate(a):
        c=np.zeros(size); c[:n]=dt*(price+ev.degradation*ev.charge_eff); c[n:2*n]=dt*(-price*ev.sell_ratio+ev.degradation/ev.discharge_eff); costrows.append(c)
        r=c.copy(); r[z]=-1; r[z+1+j]=-1; row(r,-np.inf,reference[j])
        if maxloss is not None: row(c,-np.inf,reference[j]+maxloss)
    result=milp(objective,integrality=integrality,bounds=Bounds(lower,upper),constraints=LinearConstraint(np.array(constraints),lows,highs),options={'time_limit':12,'mip_rel_gap':1e-7})
    if not result.success: raise ValueError('Scenario optimization failed: '+str(result.message))
    c=result.x[:n]; d=result.x[n:2*n]; energy=result.x[2*n:3*n+1]
    if np.max(np.minimum(c,d))>1e-6: raise ValueError('simultaneous charge/discharge')
    schedule=pd.DataFrame(dict(timestamp=pd.DatetimeIndex(times),connected_hours=dt,price=p,charge_kw=c,discharge_kw=d,soc_start=energy[:-1]/ev.capacity*100,soc_end=energy[1:]/ev.capacity*100))
    losses=np.array([r@result.x-ref for r,ref in zip(costrows,reference)])
    if maxloss is not None and losses.max()>maxloss+1e-4: raise ValueError('scenario loss cap violated')
    return dict(kind='V15_SCENARIO_SIMULATION',assumptions=asdict(ev),ending_soc=float(energy[-1]/ev.capacity*100),schedule=schedule,scenario_excess_losses=losses.tolist(),risk_weight=risk_weight,scenario_cap=maxloss,**score(p,c,d,dt,ev))

def dispatch(policy,p,times,templates,ev=None,start=None,end=None):
    ev=ev or EV(degradation=5)
    def point_plan(planner,flag):
        plan=optimize(p,times,planner,start,end,flag)
        if abs(plan['ending_soc']-max(ev.target_soc,ev.initial_soc))>1e-5:
            plan=scenario_optimize(p,times,np.vstack([p,p]),planner,start,end,0)
        return plan
    if policy=='G2V': return optimize(p,times,ev,start,end)
    if policy=='general': return point_plan(ev,True)
    if policy=='v14_conservative': return point_plan(EV(**{**asdict(ev),'sell_ratio':.85}),True)
    if policy=='profit100':
        g=optimize(p,times,ev,start,end); v=point_plan(ev,True)
        return v if g['net_cost']-v['net_cost']>=100 else g
    if len(p)!=24: raise ValueError('Scenario policy currently requires24 hourly prices; partial connection inside24h supported')
    scenarios=np.asarray(p)[None,:]+np.asarray(templates)
    if policy=='cvar025': return scenario_optimize(p,times,scenarios,ev,start,end,.25)
    if policy=='cvar1': return scenario_optimize(p,times,scenarios,ev,start,end,1)
    if policy=='scenario_cap50': return scenario_optimize(p,times,scenarios,ev,start,end,0,maxloss=50)
    raise ValueError('unknown policy')
