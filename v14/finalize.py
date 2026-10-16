"""No refits: add boundary-free ramp diagnostics and verified resume metadata."""
import hashlib,json
import pandas as pd,numpy as np
from sklearn.metrics import f1_score
from v11.store import connect,latest,now,ROOT
from v14.research import HOME,write,ev_metrics

def run():
    folder=HOME/'study'
    policies=pd.read_csv(folder/'policy_audit_metrics.csv'); audit=pd.read_csv(folder/'audit_predictions.csv'); audit.timestamp=pd.to_datetime(audit.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
    selected=json.loads((HOME/'policy_selection_before_policy_audit.json').read_text())['name']
    for name in ['V12_hgb_baseline',selected]:
        if not ((policies.name==name)&(policies.policy=='nominal')).any():
            summary,daily=ev_metrics(audit,audit[name],'nominal'); policies=pd.concat([policies,pd.DataFrame([dict(name=name,policy='nominal',**summary)])],ignore_index=True)
    policies.to_csv(folder/'policy_audit_metrics.csv',index=False); policies.to_csv(ROOT/'docs/V14_POLICY_AUDIT.csv',index=False)
    for predname,metricname in [('selection_predictions.csv','selection_metrics.csv'),('audit_predictions.csv','audit_metrics.csv'),('known_october_predictions.csv','known_october_metrics.csv')]:
        pred=pd.read_csv(folder/predname); metrics=pd.read_csv(folder/metricname)
        for i,r in metrics.iterrows():
            if r['name'] not in pred: continue
            changes=[]
            for _,g in pred.groupby('target_date'):
                changes.append((np.diff(g.actual_smp),np.diff(g[r['name']])))
            a=np.concatenate([x for x,_ in changes]); p=np.concatenate([y for _,y in changes]); up=a>=15; down=a<=-15
            metrics.loc[i,'within_day_up_ramp_MAE']=float(np.mean(abs(p[up]-a[up]))) if up.any() else np.nan
            metrics.loc[i,'within_day_down_ramp_MAE']=float(np.mean(abs(p[down]-a[down]))) if down.any() else np.nan
            metrics.loc[i,'within_day_ramp_F1']=f1_score(up,p>=15,zero_division=0)
        metrics.to_csv(folder/metricname,index=False)
    val=pd.read_csv(folder/'selection_metrics.csv'); follow=pd.read_csv(HOME/'chronological_followup/metrics.csv')
    pd.concat([val,follow],ignore_index=True).to_csv(ROOT/'docs/V14_ALL_SELECTION_METRICS.csv',index=False)
    outcomes=pd.read_csv(folder/'hypothesis_outcomes.csv')
    if not outcomes.name.eq('gru_chronological7day_pilot').any():
        outcomes=pd.concat([outcomes,pd.DataFrame([dict(name='gru_chronological7day_pilot',MAE_improved=False,peak_MAE_improved=False,G2V_improved=False,V2G_improved=False,F1_improved=False,explanation='Chronological GRU failed validation; no2026 audit or promotion')])],ignore_index=True)
    outcomes.to_csv(folder/'hypothesis_outcomes.csv',index=False); outcomes.to_csv(ROOT/'docs/V14_HYPOTHESIS_OUTCOMES.csv',index=False)
    for source,target in [('audit_metrics.csv','V14_AUDIT_METRICS.csv'),('known_october_metrics.csv','V14_KNOWN_OCTOBER_DIAGNOSIS.csv')]: (ROOT/'docs'/target).write_bytes((folder/source).read_bytes())
    design=json.loads((folder/'design_before_scores.json').read_text()); db=connect(); raw=latest(db,'land'); db.close()
    current=hashlib.sha256(raw[['timestamp','smp']].to_csv(index=False).encode()).hexdigest()
    if current==design['source_sha256']:
        source=folder/'retrospective_features.csv'; snapshot=folder/'frozen_design_matrix.csv'
        if not snapshot.exists(): snapshot.write_bytes(source.read_bytes())
        write(folder/'frozen_snapshot.json',dict(captured_at=now(),sha256=hashlib.file_digest(snapshot.open('rb'),'sha256').hexdigest(),source_prices_match_original_design=True,original_design_source_sha256=design['source_sha256'],notes='Resume only from this verified design matrix; new prices do not silently change old experiment'))
    else: write(folder/'frozen_snapshot_unavailable.json',dict(at=now(),reason='Current prices differ from registered source; no retrospectively asserted equivalence'))
    print('Final diagnostics and frozen matrix verified:',current==design['source_sha256'])

if __name__=='__main__': run()
