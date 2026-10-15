"""Additive V12 job; existing V11 data collection task stays unchanged."""
import json
from v11.store import now
from v12.research import HOME

def main():
    from v12.publication import monitor
    from v12.forward import issue,reconcile
    from v12.decisions import freeze,evaluate
    HOME.mkdir(parents=True,exist_ok=True); stages=[]
    for name,action in [('publication',monitor),('forward',issue),('reconcile',reconcile),('dispatch',freeze),('dispatch_evaluate',evaluate)]:
        try: action(); stages.append(dict(stage=name,status='success'))
        except Exception as exc: stages.append(dict(stage=name,status='failed',reason=type(exc).__name__))
    (HOME/'pipeline_status.json').write_text(json.dumps(dict(at=now(),stages=stages),indent=2),encoding='utf-8')
    # Enough new completed target dates make a review eligible, never trigger
    # training/model replacement based merely on an accumulating hour counter.
    results=json.loads((HOME/'evaluation_index.json').read_text(encoding='utf-8')) if (HOME/'evaluation_index.json').exists() else []
    days={r['target_date'] for r in results if r['status']=='evaluated' and r['interpretation'].startswith('prospective')}
    (HOME/'retraining_readiness.json').write_text(json.dumps(dict(at=now(),completed_forward_days=len(days),minimum_days=30,ready_for_manual_chronological_candidate_review=len(days)>=30,automatic_promotion=False,command='python -m v12.retrain --as-of <timezone-aware-cutoff>',warning='Design a new future holdout before retraining; fixed historical audit is already seen'),indent=2),encoding='utf-8')
    if any(x['status']=='failed' for x in stages): raise SystemExit(1)

if __name__=='__main__': main()
