"""Small connection-aware cheap-hour pilot; all choices fixed before scores."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import json,joblib,numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMClassifier,LGBMRanker
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from v11.store import connect,latest,now
from v12.ev import EV,score,baseline
from v15.policies import dispatch
from v17.features import build,FEATURES
from v17.common import HOME,ROOT,write,csv,sha
from v17.charging import PATTERNS,priority_plan

PRICE=FEATURES+['days_ahead']
CONNECTION=['connection_hour','window_hours','hours_from_connect','required_hours','known_window_rank','window_profile_gap','window_weekly_rank']
INPUTS=PRICE+CONNECTION
MODELS=['joint_price','cheap_classifier','window_cost_rank','rank_price_blend','known_profile']
FOLDS=[dict(name='spring2025',train_end='2024-12-31',cal_start='2025-01-01',cal_end='2025-03-31',start='2025-04-01',end='2025-06-30'),dict(name='winter2025',train_end='2025-06-30',cal_start='2025-07-01',cal_end='2025-09-30',start='2025-10-01',end='2025-12-31')]

def register():
    file=HOME/'study/design_before_scores.json'
    if file.exists():return
    write(file,dict(at=now(),folds=FOLDS,EV=EV(degradation=5).__dict__,patterns=PATTERNS,new_hypotheses=dict(joint_price='Train D2/D3 jointly for same-origin overnight coverage',cheap_classifier='Predict charge-positive hours conditional on actual connection window; weight wrong choices by economic margin',window_cost_rank='Reverse old high-price NDCG objective; linear savings relevance, cheap end, window queries',rank_price_blend='Fixed half rank percentiles of regression and cheap rank; no additional fit'),new_fits=6,seed=42,operating_model_changed=False,qualification='Mean fold-pattern G2V regret ratio<=.95 AND every pattern mean ratio<=1.10 AND average realized cost no higher; both folds required. Never choose by2026 audit.',ranking_labels='7 bins linear actual within-query savings, label_gain0..7; training labels only',historical_periods_previously_seen=True,year2026_only_post_selection_audit=True,shadow_models=MODELS[:4],no_new_external_training_without_original_issue_proof=True))

def matrix(prices,labels=True,target_start=None):
    frames=[]
    for ahead in [2,3]:
        times=pd.date_range(pd.Timestamp(target_start)+pd.Timedelta(days=ahead-2),periods=24,freq='h') if target_start is not None else None
        f=build(prices,days_ahead=ahead,targets=times,include_labels=labels);f['base_day']=(pd.DatetimeIndex(f.origin_timestamp).normalize()+pd.Timedelta(days=2)).strftime('%Y-%m-%d');frames.append(f)
    f=pd.concat(frames,ignore_index=True).sort_values(['base_day','timestamp']);count=f.groupby('base_day').size();return f[f.base_day.isin(count[count==48].index)].reset_index(drop=True)

def query(group,hour,length,ev=None):
    ev=ev or EV(degradation=5);start=pd.Timestamp(group.base_day.iloc[0],tz='Asia/Seoul')+pd.Timedelta(hours=hour);end=start+pd.Timedelta(hours=length)
    q=group[(group.timestamp>=start)&(group.timestamp<end)].copy().reset_index(drop=True)
    if len(q)!=length or not ((pd.DatetimeIndex(q.timestamp)[1:]-pd.DatetimeIndex(q.timestamp)[:-1])==pd.Timedelta(hours=1)).all():raise ValueError('Missing genuine cross-date price/features')
    q['connection_hour']=hour;q['window_hours']=length;q['hours_from_connect']=np.arange(length);q['required_hours']=max(0,ev.capacity*(ev.target_soc-ev.initial_soc)/100)/(ev.charge_eff*ev.charge_kw)
    q['known_window_rank']=q.daily.rank(method='first',pct=True);q['window_profile_gap']=q.daily-q.daily.mean();q['window_weekly_rank']=q.weekly.rank(method='first',pct=True)
    return q

def queries(f,end=None):
    output=[];groups=[];binary=[];relevance=[];weights=[]
    for day,g in f.groupby('base_day',sort=True):
        for name,hour,length in PATTERNS:
            if length<30/(.9*7):continue
            q=query(g,hour,length)
            if end and q.target_date.max()>end:continue
            a=q.actual_smp.to_numpy();plan=priority_plan(a,q.timestamp);label=(plan['schedule'].charge_kw.to_numpy()>1e-8).astype(int)
            gains=(a.max()-a)/(np.ptp(a)+1e-9);rel=np.rint(7*gains).astype(int);marginal=np.sort(a)[min(len(a)-1,4)];w=1+np.minimum(abs(a-marginal),50)/10
            output.append(q);groups.append(len(q));binary.extend(label);relevance.extend(rel);weights.extend(w)
    return pd.concat(output,ignore_index=True),groups,np.asarray(binary),np.asarray(relevance),np.asarray(weights)

def fit(train,fold):
    folder=HOME/'study';path=folder/(fold['name']+'.joblib')
    if path.exists():return joblib.load(path)
    p=train[train.target_date<=fold['train_end']];reg=HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=31,learning_rate=.05,early_stopping=False,random_state=42);reg.fit(p[PRICE],p.actual_smp-p.daily)
    q,groups,binary,rel,w=queries(train,fold['train_end'])
    classifier=LGBMClassifier(n_estimators=140,num_leaves=15,min_child_samples=40,learning_rate=.04,n_jobs=4,verbosity=-1,random_state=42);classifier.fit(q[INPUTS],binary,sample_weight=w)
    rank=LGBMRanker(n_estimators=140,num_leaves=15,min_child_samples=40,learning_rate=.04,n_jobs=4,verbosity=-1,random_state=42,label_gain=list(range(8)),lambdarank_truncation_level=10);rank.fit(q[INPUTS],rel,group=groups)
    bundle=dict(regressor=reg,classifier=classifier,ranker=rank,price_features=PRICE,connection_features=INPUTS,train_end=fold['train_end'],trained_at=now(),query_rows=len(q),train_rows=len(p),kind='V17_SHARED_ORIGIN_D2_D3_SHADOW_NOT_OPERATIONAL');joblib.dump(bundle,path);return bundle

def infer(bundle,q):
    price=q.daily.to_numpy()+bundle['regressor'].predict(q[PRICE]);cheap=-bundle['classifier'].predict_proba(q[INPUTS])[:,1];rank=-bundle['ranker'].predict(q[INPUTS]);percent=lambda a:pd.Series(a).rank(method='first',pct=True).to_numpy()
    scores=dict(joint_price=price,cheap_classifier=cheap,window_cost_rank=rank,rank_price_blend=.5*percent(price)+.5*percent(rank),known_profile=.69*q.daily.to_numpy()+.31*q.weekly.to_numpy())
    return price,scores

def evaluate(bundle,f,fold,role):
    folder=HOME/'study';dest=folder/(fold+'_'+role+'_daily.csv');rows=csv(dest).to_dict('records') if dest.exists() else [];done={(r['day'],r['pattern'],r['model']) for r in rows};ev=EV(degradation=5)
    for day,g in f.groupby('base_day',sort=True):
        for pattern,hour,length in PATTERNS:
            if length<30/(.9*7):
                if (day,pattern,'all_models') not in done:rows.append(dict(day=day,pattern=pattern,model='all_models',status='infeasible',reason='Target30 battery kWh exceeds connection energy capacity'))
                continue
            q=query(g,hour,length)
            # No endpoint labels crossing the declared split boundary.
            end={'spring2025':'2025-06-30','winter2025':'2025-12-31','audit2026H1':'2026-06-30'}[fold]
            if q.target_date.max()>end:continue
            a=q.actual_smp.to_numpy();times=pd.DatetimeIndex(q.timestamp);point,scores=infer(bundle,q);oracle=priority_plan(a,times,ev=ev);v_oracle=dispatch('general',a,times,None,ev)
            immediate=baseline(a,times,ev)['net_cost'];uniform=baseline(a,times,ev,uniform=True)['net_cost']
            for name,s in scores.items():
                if (day,pattern,name) in done:continue
                # Ranking scores are priorities, NOT a claimed SMP price forecast.
                planning=point if name=='joint_price' else np.sort(point)[pd.Series(s).rank(method='first').astype(int).to_numpy()-1]
                if name=='known_profile':planning=s
                plan=priority_plan(planning,times,s,ev);vplan=dispatch('general',planning,times,None,ev)
                gs=plan['schedule'];vs=vplan['schedule'];gc=score(a,gs.charge_kw.to_numpy(),gs.discharge_kw.to_numpy(),gs.connected_hours.to_numpy(),ev);vc=score(a,vs.charge_kw.to_numpy(),vs.discharge_kw.to_numpy(),vs.connected_hours.to_numpy(),ev)
                charge=gs.charge_kw.to_numpy()*gs.connected_hours.to_numpy();optimal=oracle['schedule'].charge_kw.to_numpy();selected=set(np.flatnonzero(charge>1e-8));best=set(np.flatnonzero(optimal>1e-8));overlap=len(selected&best)/max(1,len(best))
                rows.append(dict(day=day,pattern=pattern,model=name,status='evaluated',hours=length,origin=str(q.origin_timestamp.iloc[0]),last_price_date=q.target_date.max(),price_MAE=float(abs(point-a).mean()),decision_proxy_MAE=float(abs(planning-a).mean()),rank_spearman=float(spearmanr(s,a).statistic) if np.ptp(a)>0 and np.ptp(s)>0 else 0,cheap_hour_overlap=overlap,charge_energy_overlap=float(np.minimum(gs.charge_kw.to_numpy(),optimal).sum()/max(1e-8,optimal.sum())),G2V_expected=plan['net_cost'],G2V_actual=gc['net_cost'],G2V_regret=gc['net_cost']-oracle['net_cost'],saving_immediate=immediate-gc['net_cost'],saving_uniform=uniform-gc['net_cost'],V2G_expected_gain=plan['net_cost']-vplan['net_cost'],V2G_actual_gain=gc['net_cost']-vc['net_cost'],V2G_export=vc['grid_export_kwh'],V2G_regret=vc['net_cost']-v_oracle['net_cost'],V2G_loss=vc['net_cost']>gc['net_cost']+1e-6,degradation=vc['degradation_cost'],ending_soc=plan['ending_soc']))
        if len(rows)%80<42:pd.DataFrame(rows).to_csv(dest,index=False)
    pd.DataFrame(rows).to_csv(dest,index=False);return pd.DataFrame(rows)

def summaries(f,fold,role):
    rows=[]
    for (model,pattern),g in f[f.status=='evaluated'].groupby(['model','pattern']):
        keys=['price_MAE','decision_proxy_MAE','rank_spearman','cheap_hour_overlap','charge_energy_overlap','G2V_actual','G2V_regret','saving_immediate','saving_uniform','V2G_expected_gain','V2G_actual_gain','V2G_export','V2G_regret','degradation','ending_soc']
        rows.append(dict(fold=fold,role=role,model=model,pattern=pattern,days=len(g),V2G_loss_days=int(g.V2G_loss.sum()),**g[keys].mean().to_dict()))
    return rows

def run():
    register();folder=HOME/'study';path=folder/'matrix.csv'
    if path.exists():f=csv(path);f.timestamp=pd.to_datetime(f.timestamp,utc=True).dt.tz_convert('Asia/Seoul');f.origin_timestamp=pd.to_datetime(f.origin_timestamp,utc=True).dt.tz_convert('Asia/Seoul')
    else:
        db=connect();prices=latest(db,'land');db.close();f=matrix(prices);f.to_csv(path,index=False)
    result=[]
    for fold in FOLDS:
        train=f[f.base_day<=fold['train_end']].copy();b=fit(train,fold);val=f[(f.base_day>=fold['start'])&(f.base_day<=fold['end'])].copy();r=evaluate(b,val,fold['name'],'validation');result.extend(summaries(r,fold['name'],'validation'));pd.DataFrame(result).to_csv(ROOT/'docs/V17_MODEL_PATTERN_COMPARISON.csv',index=False);print(fold['name'],'complete',flush=True)
    total=pd.DataFrame(result);base=total[total.model=='joint_price'].set_index(['fold','pattern']);qualified=[];scores=[]
    for name in MODELS[1:]:
        candidate=total[total.model==name].set_index(['fold','pattern']);ratio=candidate.G2V_regret/base.G2V_regret.clip(lower=1);pattern=ratio.groupby('pattern').mean();ok=ratio.mean()<=.95 and pattern.max()<=1.10 and candidate.G2V_actual.mean()<=base.G2V_actual.mean()
        scores.append(dict(model=name,mean_regret_ratio=float(ratio.mean()),worst_pattern_mean_ratio=float(pattern.max()),qualified=bool(ok),not_promoted=True))
        if ok:qualified.append(name)
    file=folder/'selection_before_audit.json';selection=dict(at=now(),qualified=qualified,scores=scores,shadow_models=MODELS[:4],operating_model_replaced=False)
    if file.exists():assert json.loads(file.read_text())['qualified']==qualified,'No audit-dependent reselection'
    else:write(file,selection)
    b=joblib.load(folder/'winter2025.joblib');audit=f[(f.base_day>='2026-01-01')&(f.base_day<='2026-06-30')];r=evaluate(b,audit,'audit2026H1','seen_diagnostic');result.extend(summaries(r,'audit2026H1','seen_diagnostic'));pd.DataFrame(result).to_csv(ROOT/'docs/V17_MODEL_PATTERN_COMPARISON.csv',index=False)
    print('V17 small study completed; no October tuning and no operating promotion',flush=True)

if __name__=='__main__':run()
