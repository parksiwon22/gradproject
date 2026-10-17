"""Reuse fixed families; only one preregistered separate-horizon HGB fit."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import time,joblib,numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from v11.store import connect,latest
from v11.features import FEATURES
from v17.features import build
from v17.study import PRICE,query,infer
from v18.evaluation import price_metrics,compare_ev
from v19.common import HOME,ROOT,read,write,write_once,write_csv,sha,now

SPECS={
 'V12_HGB':('artifacts/v12/study/hgb_none.joblib',[2],'v12'),
 'V12_LightGBM_peak':('artifacts/v12/study/lightgbm_peak.joblib',[2],'v12'),
 'V14_shape_level_blend':('artifacts/v14/study/shape_level_blend.joblib',[2],'v14'),
 'V15_two_stage_025':('artifacts/v15/candidates/two_stage_shrink_0.25.joblib',[2],'v15'),
 'V15_two_stage_05':('artifacts/v15/candidates/two_stage_shrink_0.5.joblib',[2],'v15'),
 'V17_joint':('artifacts/v17/study/winter2025.joblib',[2,3],'v17'),
 'V17_charge_rank_blend':('artifacts/v17/study/winter2025.joblib',[2],'v17_rank'),
 'V19_D3_HGB':('artifacts/v19/study/d3_hgb.joblib',[3],'d3')}

def predict_model(name,prices,target,ahead,bundle=None):
    path,horizons,kind=SPECS[name]
    if ahead not in horizons:raise ValueError('Model not registered for this horizon')
    b=bundle if bundle is not None else joblib.load(ROOT/path);times=pd.date_range(pd.Timestamp(target,tz='Asia/Seoul'),periods=24,freq='h');f=build(prices,targets=times,days_ahead=ahead,include_labels=False);priority=None
    if kind=='v12':
        from v12.research import predict
        point=predict(b,f)
    elif kind=='v14':
        from v14.features import augment
        from v14.research import predict
        f,_,_=augment(prices,external=False,targets=times,include_labels=False);point=predict(b,f)
    elif kind=='v15':
        from v15.features import build as extra
        from v15.research import predict
        f,_=extra(prices,targets=times,include_labels=False);point=predict(b,f)
    elif kind in ['v17','v17_rank']:
        point=f.daily.to_numpy()+b['regressor'].predict(f[PRICE])
        if kind=='v17_rank':
            f['base_day']=target;q=query(f,0,24);_,scores=infer(b,q);priority=scores['rank_price_blend']
    else:point=f.daily.to_numpy()+b['model'].predict(f[FEATURES])
    if len(f)!=24 or not np.isfinite(point).all():raise ValueError('Incomplete genuine target features')
    planning=np.asarray(point) if priority is None else np.sort(point)[pd.Series(priority).rank(method='first').astype(int).to_numpy()-1]
    return f,np.asarray(point),priority,planning

def study():
    folder=HOME/'study';folder.mkdir(exist_ok=True,parents=True);design=folder/'design_before_scores.json'
    if not design.exists():write_once(design,dict(at=now(),hypothesis='Separate D3 residual fit may avoid negative transfer from joint D2/D3; one configuration only',new_fits=1,train=['2023-01-01','2025-06-30'],validation=['2025-10-01','2025-12-31'],known_historical_validation=True,october2026_excluded=True,qualification='D3 MAE<=jointD3*1.05 AND G2V regret<=jointD3*.95; no operating replacement',seed=42,params=dict(max_iter=200,max_leaf_nodes=31,learning_rate=.05,early_stopping=False)))
    db=connect();prices=latest(db,'land');db.close();matrix=build(prices,days_ahead=3);train=matrix[matrix.target_date.between('2023-01-01','2025-06-30')];val=matrix[matrix.target_date.between('2025-10-01','2025-12-31')];path=folder/'d3_hgb.joblib'
    if not path.exists():
        m=HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=31,learning_rate=.05,early_stopping=False,random_state=42);m.fit(train[FEATURES],train.actual_smp-train.daily);joblib.dump(dict(model=m,trained_at=now(),trained_through='2025-06-30',train_rows=len(train),features=FEATURES,horizon_days=3,kind='SEPARATE_D3_HGB_RESEARCH_NOT_OPERATING_REPLACEMENT'),path)
    result=folder/'daily.csv'
    needs_counts=result.exists() and 'peak_TP' not in pd.read_csv(result,nrows=1).columns
    if needs_counts:
        import shutil
        backup=folder/'daily_before_metric_aggregation_fix.csv'
        if not backup.exists():shutil.copyfile(result,backup)
    if not result.exists() or needs_counts:
        b=joblib.load(path);joint=joblib.load(ROOT/SPECS['V17_joint'][0]);rows=[]
        for day,g in val.groupby('target_date'):
            for name,ahead in [('V19_D3_HGB',3),('V17_joint_D3',3),('V17_joint_D2',2)]:
                base=g if ahead==3 else build(prices,targets=pd.DatetimeIndex(g.timestamp),days_ahead=2)
                p=base.daily.to_numpy()+(b['model'].predict(base[FEATURES]) if name=='V19_D3_HGB' else joint['regressor'].predict(base[PRICE]));a=base.actual_smp.to_numpy();metrics=price_metrics(a,p);ev=compare_ev(p,a,base.timestamp)
                actual_peak=a>=150;pred_peak=p>=150;rows.append(dict(day=day,model=name,days_ahead=ahead,peak_TP=int((actual_peak&pred_peak).sum()),peak_FP=int((~actual_peak&pred_peak).sum()),peak_FN=int((actual_peak&~pred_peak).sum()),**metrics,**{k:v for k,v in ev.items() if k not in ['G2V','V2G','official_G2V','official_V2G']}))
        write_csv(result,pd.DataFrame(rows))
    f=pd.read_csv(result);summary=f.groupby(['model','days_ahead']).mean(numeric_only=True).reset_index()
    for i,row in summary.iterrows():
        g=f[f.model==row['model']];positive=g[g.peak_hours>0];tp,fp,fn=g.peak_TP.sum(),g.peak_FP.sum(),g.peak_FN.sum();summary.loc[i,'validation_days']=len(g);summary.loc[i,'peak_hours']=g.peak_hours.sum();summary.loc[i,'peak_F1']=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan
        for col in ['peak_MAE','peak_bias']:summary.loc[i,col]=np.average(positive[col],weights=positive.peak_hours) if len(positive) else np.nan
        for col in ['peak_TP','peak_FP','peak_FN']:summary.loc[i,col]=g[col].sum()
    write_csv(ROOT/'docs/V19_D3_HISTORICAL_COMPARISON.csv',summary);j=summary[summary.model=='V17_joint_D3'].iloc[0];candidate=summary[summary.model=='V19_D3_HGB'].iloc[0];qualified=bool(candidate.MAE<=j.MAE*1.05 and candidate.G2V_regret<=j.G2V_regret*.95)
    dest=folder/'selection_before_live.json'
    if not dest.exists():write_once(dest,dict(at=now(),qualified=qualified,operating_model_replaced=False,shadow_comparison_only=True,MAE=float(candidate.MAE),G2V_regret=float(candidate.G2V_regret)))
    print(summary[['model','days_ahead','MAE','G2V_regret','V2G_actual_gain']].to_string(index=False));return summary

def register():
    path=HOME/'model_registry.json'
    if not path.exists():
        entries={name:dict(path=p,horizons=h,kind=k,sha256=sha(ROOT/p),group='D3_separate_horizon_candidate' if k=='d3' else 'D2_price_priority_proxy' if k=='v17_rank' else 'joint_same_weights_horizon_comparison' if k=='v17' else 'D2_preserved_price_family',trained_through='2025-06-30') for name,(p,h,k) in SPECS.items()}
        write_once(path,dict(locked_at=now(),entries=entries,selection_sha256=sha(HOME/'study/selection_before_live.json'),operating_model_replaced=False,default_recommendation='G2V'))
    return read(path)

if __name__=='__main__':study();print('V19 model registry locked at',register()['locked_at'])
