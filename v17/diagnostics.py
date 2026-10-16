"""Read completed experiments only: no fitting or candidate reselection."""
import json
import joblib,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score
from v17.common import ROOT,HOME,csv,write,sha
from v17.study import INPUTS,PRICE,CONNECTION,FOLDS,query,infer
from v17.charging import priority_plan
from v12.ev import EV

def run():
    figures=ROOT/'docs/v17_figures';figures.mkdir(exist_ok=True)
    old=csv(ROOT/'artifacts/v14/study/selection_predictions.csv');metrics=csv(ROOT/'docs/V14_ALL_SELECTION_METRICS.csv');rows=[]
    for model in ['V12_hgb_baseline','lgb_hour_rank','rank_level_blend']:
        low=[];high=[];ndcg=[]
        for day,g in old.groupby('target_date'):
            a=g.actual_smp.to_numpy();p=g[model].to_numpy();low.append(len(set(np.argsort(a,kind='stable')[:5])&set(np.argsort(p,kind='stable')[:5]))/5);high.append(len(set(np.argsort(a,kind='stable')[-5:])&set(np.argsort(p,kind='stable')[-5:]))/5)
            ndcg.append(ndcg_score([pd.Series(a).rank(method='first').to_numpy()-1],[p],k=5))
        m=metrics[metrics.name==model].iloc[0]
        rows.append(dict(model=model,days=len(low),low5_overlap=np.mean(low),high5_overlap=np.mean(high),expensive_end_NDCG5=np.mean(ndcg),MAE=m.MAE,G2V_regret=m.G2V_regret,V2G_regret=m.V2G_regret))
    pd.DataFrame(rows).to_csv(ROOT/'docs/V17_V14_RANK_AUDIT.csv',index=False)
    f=csv(ROOT/'docs/V17_MODEL_PATTERN_COMPARISON.csv');macro=f.groupby(['role','model'])[['price_MAE','G2V_regret','cheap_hour_overlap','charge_energy_overlap','saving_immediate','saving_uniform','V2G_actual_gain','V2G_export']].mean().reset_index();macro.to_csv(ROOT/'docs/V17_MODEL_MACRO_COMPARISON.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,role,title in zip(axes,['validation','seen_diagnostic'],['2025 chronological validation (seen history)','2026 H1 diagnostic only (not model selection)']):
        s=f[f.role==role].groupby(['pattern','model']).G2V_regret.mean().unstack();s.plot.bar(ax=ax);ax.set_title(title);ax.set_ylabel('G2V regret (virtual KRW/day)');ax.set_xlabel('Genuine connection window');ax.tick_params(axis='x',rotation=35)
    fig.tight_layout();fig.savefig(figures/'connection_regret.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5));val=f[f.role=='validation'].groupby('model')[['V2G_actual_gain','V2G_export']].mean();val.V2G_actual_gain.plot.bar(ax=axes[0],color='#a03c40');val.V2G_export.plot.bar(ax=axes[1],color='#316c96');axes[0].set_ylabel('Additional V2G gain vs G2V (virtual KRW/day)');axes[1].set_ylabel('Grid export (kWh/day)')
    for ax in axes:ax.tick_params(axis='x',rotation=30)
    fig.suptitle('V2G: both realized benefit and discharge activity matter');fig.tight_layout();fig.savefig(figures/'v2g_activity_and_gain.png',dpi=150);plt.close(fig)
    # Fixed first winter validation date; illustration, never a selected best case.
    matrix=csv(HOME/'study/matrix.csv');matrix.timestamp=pd.to_datetime(matrix.timestamp,utc=True).dt.tz_convert('Asia/Seoul');g=matrix[matrix.base_day=='2025-10-01'];q=query(g,18,14);bundle=joblib.load(HOME/'study/winter2025.joblib');point,scores=infer(bundle,q);a=q.actual_smp.to_numpy();ev=EV(degradation=5)
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True);x=np.arange(len(q));axes[0].plot(x,a,label='Official SMP');axes[0].plot(x,point,label='Joint D2/D3 price');axes[0].set_ylabel('KRW/kWh');axes[0].legend()
    for name in ['joint_price','rank_price_blend']:
        plan=priority_plan(point,q.timestamp,scores[name],ev);s=plan['schedule'];axes[1].step(x,s.charge_kw,where='mid',label=name);axes[2].plot(x,s.soc_end,label=name)
    axes[1].set_ylabel('G2V kW');axes[2].set_ylabel('SOC (%)');axes[2].set_xticks(x,[t.strftime('%m/%d %Hh') for t in q.timestamp],rotation=45);axes[1].legend();axes[2].legend();fig.suptitle('Fixed historical example: 2025-10-01 18h to next date 08h (not live)');fig.tight_layout();fig.savefig(figures/'fixed_overnight_example.png',dpi=150);plt.close(fig)
    write(ROOT/'docs/V17_FEATURES.json',dict(price_features=PRICE,connection_features=CONNECTION,total_connection_features=len(INPUTS),folds=FOLDS,external_groups_used=[],seed=42,score_units='Classifier/rank scores are priorities, not SMP prices'))
    write(HOME/'study/model_manifest.json',{p.name:dict(sha256=sha(p),parameters={k:v.get_params() for k,v in joblib.load(p).items() if hasattr(v,'get_params')}) for p in (HOME/'study').glob('*.joblib')})
    print(macro.to_string(index=False));print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':run()
