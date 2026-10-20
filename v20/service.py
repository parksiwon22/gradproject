"""Reuse the V19 slow path, then refresh V20 audit even after collector failure."""
from v19.observer import run
import pandas as pd
from v20.core import audit,HOME
from v20.research import evaluate as evaluate_research
from v19.common import write_once,now

if __name__=='__main__':
    HOME.mkdir(parents=True,exist_ok=True)
    stages=[]
    for name,fn in [('v19_observer',run),('v20_audit',audit),('v20_research_evaluation',evaluate_research)]:
        try:fn();stages.append(dict(stage=name,status='success',at=now()))
        except (Exception,SystemExit) as exc:stages.append(dict(stage=name,status='failed',reason=type(exc).__name__,at=now()))
    write_once(HOME/'service_runs'/(pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%f')+'.json'),dict(at=now(),stages=stages,creates_forecasts=False))
    if any(s['status']=='failed' for s in stages):raise SystemExit(1)
