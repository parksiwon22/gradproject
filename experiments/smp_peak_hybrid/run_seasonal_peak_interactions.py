"""Season x hour peak experiments using existing V11/V17 features unchanged."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

from v11.store import ROOT
from experiments.smp_performance.train_ml_candidates import (
    FEATURES_BY_H, FOLDS, allocate, day_vectors, features_for, fit_model, load_prices,
)
from experiments.smp_peak_hybrid.run_peak_hybrid_study import PEAK, daily_metrics, aggregate, anchor_for

OUT=ROOT/'docs'/'smp_peak_hybrid'
OLD_HOURLY=ROOT/'docs'/'smp_improvement_ml'/'hourly_predictions.csv'
ET_HOURLY=ROOT/'docs'/'smp_improvement_ml_lgbm'/'d1_v11_extratrees_fold_hourly.csv'
LGB_HOURLY=ROOT/'docs'/'smp_improvement_ml_lgbm'/'d2_d3_hourly_predictions.csv'
CALENDAR_GROUPS={
    'hour':['hour_sin','hour_cos'],
    'month':['month_sin','month_cos'],
    'weekday':['dow_sin','dow_cos','weekend'],
}
FIXED_SEASONS={1:'winter',2:'winter',12:'winter',3:'spring',4:'spring',5:'spring',
               6:'summer',7:'summer',8:'summer',9:'autumn',10:'autumn',11:'autumn'}


def fit_gate_models(x,y,peak,seed=42):
    clf=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=7,learning_rate=.05,
        min_samples_leaf=20,l2_regularization=3.0,early_stopping=False,random_state=seed,class_weight='balanced')
    clf.fit(x,peak.astype(int))
    reg=HistGradientBoostingRegressor(max_iter=180,max_leaf_nodes=7,learning_rate=.05,
        min_samples_leaf=12,l2_regularization=5.0,early_stopping=False,random_state=seed)
    reg.fit(x[peak],y[peak])
    return clf,reg


def fixed_season(month):
    return FIXED_SEASONS[int(month)]


def make_month_clusters(train_frame, k=4):
    # Cluster on training-only monthly profiles: 24 peak rates and the month's
    # total peak rate. Existing month/hour features are not augmented.
    pivot=train_frame.pivot_table(index='month',columns='hour',values='is_peak',aggfunc='mean')
    pivot=pivot.reindex(range(1,13)).fillna(train_frame.is_peak.mean())
    totals=train_frame.groupby('month').is_peak.mean().reindex(range(1,13)).fillna(train_frame.is_peak.mean())
    matrix=np.column_stack([pivot.to_numpy(float),totals.to_numpy(float)])
    scaled=StandardScaler().fit_transform(matrix)
    cluster=KMeans(n_clusters=k,random_state=42,n_init=20).fit(scaled)
    month_map={month:int(label) for month,label in zip(range(1,13),cluster.labels_)}
    return month_map,pivot


def build_train_frame(train_days,days):
    rows=[]
    for day in train_days:
        item=days[day]
        for i,(timestamp,actual,base) in enumerate(zip(item['timestamps'],item['actual'],item['baseline'])):
            ts=pd.Timestamp(timestamp)
            rows.append(dict(target_date=day,hour=ts.hour,month=ts.month,weekday=ts.dayofweek,
                actual_smp=float(actual),baseline_smp=float(base),is_peak=bool(actual>=PEAK),
                peak_uplift=float(actual-base) if actual>=PEAK else np.nan,
                x=item['X'][i]))
    return rows


def fit_subgroups(train_days,days,train_frame,season_type,month_map=None):
    # Every subgroup estimator sees the same existing features; the calendar
    # month only routes a forecast to a fold-trained seasonal model.
    def label(month): return fixed_season(month) if season_type=='fixed' else month_map[int(month)]
    x=np.concatenate([days[d]['X'] for d in train_days]); y=np.concatenate([days[d]['actual'] for d in train_days]); peak=y>=PEAK
    global_models=fit_gate_models(x,y,peak)
    frame=pd.DataFrame([{k:v for k,v in r.items() if k!='x'} for r in train_frame])
    frame['group']=frame.month.map(label)
    models={}; fallbacks=[]
    for group,g in frame.groupby('group'):
        days_rows=[r for r in train_frame if label(r['month'])==group]
        gx=np.vstack([r['x'] for r in days_rows]); gy=np.array([r['actual_smp'] for r in days_rows]); gp=gy>=PEAK
        if gp.sum()<40 or (~gp).sum()<40:
            models[group]=global_models; fallbacks.append(group)
        else:
            models[group]=fit_gate_models(gx,gy,gp,seed=42+int(group) if isinstance(group,(int,np.integer)) else 42+['winter','spring','summer','autumn'].index(group))
    return models,global_models,label,fallbacks


def infer_grouped(items,models,global_models,label,kind):
    output=[]
    for item in items:
        prob=np.empty(24); price=np.empty(24)
        for month in [pd.Timestamp(item['day']).month]:
            key=label(month); clf,reg=models.get(key,global_models)
            prob=clf.predict_proba(item['X'])[:,1];price=reg.predict(item['X'])
        output.append(dict(day=item['day'],actual=item['actual'],base_pred=item['base_pred'],
            prob=prob,peak_pred=price,hours=item['hours'],season=label(pd.Timestamp(item['day']).month),kind=kind))
    return output


def tune_gate(cal_rows,guard_mae=.1,guard_rmse=.25,guard_normal=.1,guard_regret=2.):
    ref=[daily_metrics(r['actual'],r['base_pred']) for r in cal_rows]
    baseline=dict(MAE=float(np.mean([v['MAE'] for v in ref])),
        RMSE=float(np.sqrt(np.mean([v['RMSE']**2 for v in ref]))),
        normal_MAE=float(np.mean([v['normal_MAE'] for v in ref])),
        peak_MAE=float(np.nanmean([v['peak_MAE'] for v in ref])),
        G2V_regret=float(np.mean([v['G2V_regret'] for v in ref])))
    thresholds=sorted(set([.05,.1,.15,.2,.25,.3,.4,.5,.6,.7,.8,.9]+[
        float(np.quantile(np.concatenate([r['prob'] for r in cal_rows]),q)) for q in (.5,.75,.9,.95)]))
    rows=[]
    for t in thresholds:
        for scale in (.1,.2,.3,.4,.5,.65,.8,1.,1.25):
            ds=[]
            for r in cal_rows:
                p=r['base_pred']+np.where(r['prob']>=t,scale*np.maximum(0,r['peak_pred']-r['base_pred']),0.)
                ds.append(daily_metrics(r['actual'],p))
            rows.append(dict(threshold=t,scale=scale,
                MAE=float(np.mean([d['MAE'] for d in ds])),RMSE=float(np.sqrt(np.mean([d['RMSE']**2 for d in ds]))),
                normal_MAE=float(np.mean([d['normal_MAE'] for d in ds])),peak_MAE=float(np.nanmean([d['peak_MAE'] for d in ds])),
                G2V_regret=float(np.mean([d['G2V_regret'] for d in ds]))))
    eligible=[r for r in rows if r['peak_MAE']<baseline['peak_MAE']
        and r['MAE']<=baseline['MAE']+guard_mae and r['RMSE']<=baseline['RMSE']+guard_rmse
        and r['normal_MAE']<=baseline['normal_MAE']+guard_normal and r['G2V_regret']<=baseline['G2V_regret']+guard_regret]
    if eligible: chosen=min(eligible,key=lambda r:(r['peak_MAE'],r['MAE'],r['G2V_regret'],r['RMSE'])); failed=False
    else:
        chosen=dict(threshold=np.nan,scale=0.,MAE=baseline['MAE'],RMSE=baseline['RMSE'],normal_MAE=baseline['normal_MAE'],peak_MAE=baseline['peak_MAE'],G2V_regret=baseline['G2V_regret']);failed=True
    return chosen,baseline,failed,pd.DataFrame(rows)


def classifier_stats(rows,threshold):
    a=np.concatenate([r['actual'] for r in rows]);p=np.concatenate([r['prob'] for r in rows]); y=a>=PEAK
    return dict(calibration_hours=len(a),positive_hours=int(y.sum()),positive_rate=float(y.mean()),
        average_precision=float(average_precision_score(y,p)),brier=float(brier_score_loss(y,p)),
        threshold=float(threshold),false_alarm_rate=float((((p>=threshold)&~y).sum())/(~y).sum()),
        false_discovery_rate=float((((p>=threshold)&~y).sum())/max(1,(p>=threshold).sum())))


def feature_ablation(x_train,y_train,peak_train,x_cal,y_cal,peak_cal,h,fold):
    full=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=7,learning_rate=.05,min_samples_leaf=20,
        l2_regularization=3.,early_stopping=False,random_state=42,class_weight='balanced').fit(x_train,peak_train.astype(int))
    rows=[]
    def metric(model,feature_names):
        p=model.predict_proba(x_cal[:,feature_names])[:,1] if isinstance(feature_names,list) else model.predict_proba(x_cal)[:,1]
        return dict(average_precision=float(average_precision_score(peak_cal,p)),brier=float(brier_score_loss(peak_cal,p)))
    base=metric(full,list(range(x_cal.shape[1])))
    names=FEATURES_BY_H[h]
    rows.append(dict(horizon_days=h,fold=fold,variant='full_existing_features',removed_group='',**base,
        delta_average_precision=0.,delta_brier=0.,data_partition='calibration only'))
    for group,columns in CALENDAR_GROUPS.items():
        removed=[names.index(c) for c in columns]
        keep=[i for i in range(len(names)) if i not in removed]
        model=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=7,learning_rate=.05,min_samples_leaf=20,
            l2_regularization=3.,early_stopping=False,random_state=42,class_weight='balanced').fit(x_train[:,keep],peak_train.astype(int))
        result=metric(model,keep)
        rows.append(dict(horizon_days=h,fold=fold,variant='drop_group_refit',removed_group=group,**result,
            delta_average_precision=result['average_precision']-base['average_precision'],
            delta_brier=result['brier']-base['brier'],data_partition='calibration only'))
        # Group-wise permutation of the calibration rows (columns moved together)
        # measures reliance of the already-trained full model on existing variables.
        rng=np.random.default_rng(123+removed[0]); shuffled=x_cal.copy(); shuffled[:,removed]=shuffled[rng.permutation(len(shuffled))][:,removed]
        p=full.predict_proba(shuffled)[:,1]
        ap=float(average_precision_score(peak_cal,p));brier=float(brier_score_loss(peak_cal,p))
        rows.append(dict(horizon_days=h,fold=fold,variant='group_permutation',removed_group=group,
            average_precision=ap,brier=brier,delta_average_precision=ap-base['average_precision'],
            delta_brier=brier-base['brier'],data_partition='calibration only'))
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    prices=load_prices(); series=prices.set_index('timestamp').smp.sort_index()
    maps={h:day_vectors(features_for(prices,h),h) for h in (1,2,3)}
    baseline_log=pd.read_csv(ROOT/'docs'/'smp_improvement_ml'/'experiment_log.csv')
    old=pd.read_csv(OLD_HOURLY); et=pd.read_csv(ET_HOURLY);lgb=pd.read_csv(LGB_HOURLY)
    seasonal_stats=[]; month_clusters=[]; calibration=[];calcurves=[];season_pred_rows=[];feature_rows=[];cal_model_rows=[]
    all_train_dist=[]; season_stability=[]; month_cluster_profiles=[]

    for h in (1,2,3):
        days=maps[h];feature_names=FEATURES_BY_H[h]
        for fold in FOLDS:
            name=fold['name'];train_days=[d for d in sorted(days) if d<=fold['train_end']]
            cal_days=[d for d in sorted(days) if fold['cal_start']<=d<=fold['cal_end']]
            val_days=[d for d in sorted(days) if fold['val_start']<=d<=fold['val_end']]
            train_rows=[]
            for d in train_days:
                item=days[d]
                for i,(timestamp,actual,base) in enumerate(zip(item['timestamps'],item['actual'],item['baseline'])):
                    ts=pd.Timestamp(timestamp)
                    train_rows.append(dict(target_date=d,hour=ts.hour,month=ts.month,weekday=ts.dayofweek,
                        actual_smp=float(actual),baseline_smp=float(base),is_peak=bool(actual>=PEAK),
                        peak_uplift=float(actual-base) if actual>=PEAK else np.nan,x=item['X'][i]))
            train_frame=pd.DataFrame([{k:v for k,v in r.items() if k!='x'} for r in train_rows])
            train_frame['fixed_season']=train_frame.month.map(FIXED_SEASONS)
            month_map,monthly_profiles=make_month_clusters(train_frame)
            train_frame['data_season']=train_frame.month.map(month_map)
            # Include training-only occurrence rate and peak price distribution for
            # month x hour, fixed season x hour and learned-cluster x hour.
            for rule,col in [('month','month'),('fixed_season','fixed_season'),('data_cluster','data_season')]:
                for key,g0 in train_frame.groupby([col,'hour'],sort=True):
                    g=train_frame[(train_frame[col]==key[0])&(train_frame.hour==key[1])]
                    peaks=g[g.is_peak]
                    all_train_dist.append(dict(horizon_days=h,fold=name,train_end=fold['train_end'],rule=rule,
                        group=str(key[0]),month=int(key[0]) if rule=='month' else np.nan,hour=int(key[1]),n_samples=len(g),
                        peak_hours=int(g.is_peak.sum()),peak_rate=float(g.is_peak.mean()),
                        mean_smp=float(g.actual_smp.mean()),median_smp=float(g.actual_smp.median()),
                        mean_peak_smp=float(peaks.actual_smp.mean()) if len(peaks) else np.nan,
                        median_peak_smp=float(peaks.actual_smp.median()) if len(peaks) else np.nan,
                        q90_peak_smp=float(peaks.actual_smp.quantile(.9)) if len(peaks) else np.nan,
                        mean_peak_uplift_over_daily=float(peaks.peak_uplift.mean()) if len(peaks) else np.nan,
                        sample_period='training only'))
            for m,cid in month_map.items():
                month_clusters.append(dict(horizon_days=h,fold=name,train_end=fold['train_end'],month=m,data_cluster=cid,
                    selection_source='training monthly-hour peak frequency only'))
            for month,vec in monthly_profiles.iterrows():
                for hour,value in vec.items(): month_cluster_profiles.append(dict(horizon_days=h,fold=name,month=int(month),hour=int(hour),training_peak_rate=float(value),cluster=month_map[int(month)]))
            for rule,col in [('fixed_season','fixed_season'),('data_cluster','data_season')]:
                top=[]
                for group,g in train_frame.groupby(col):
                    rates=g.groupby('hour').is_peak.mean().sort_values(ascending=False,kind='stable')
                    top.append((group,list(map(int,rates.head(3).index))))
                season_stability.append(dict(horizon_days=h,fold=name,rule=rule,group_top3_hours=json.dumps(top),months=json.dumps({str(k):v for k,v in month_map.items()})))

            x_train=np.vstack([r['x'] for r in train_rows]);y_train=train_frame.actual_smp.to_numpy(float);peak_train=y_train>=PEAK
            x_cal=np.concatenate([days[d]['X'] for d in cal_days]);y_cal=np.concatenate([days[d]['actual'] for d in cal_days]);peak_cal=y_cal>=PEAK
            feature_rows.extend(feature_ablation(x_train,y_train,peak_train,x_cal,y_cal,peak_cal,h,name))

            # Recreate the already-selected base model on train-only inputs.
            train_x=np.concatenate([days[d]['X'] for d in train_days]);train_y=np.concatenate([days[d]['actual'] for d in train_days]);train_daily=np.concatenate([days[d]['baseline'] for d in train_days]);res=train_y-train_daily
            sample_dates=np.repeat(train_days,24)
            if h==1:
                base_model=ExtraTreesRegressor(n_estimators=150,min_samples_leaf=10,max_features=.8,n_jobs=4,random_state=42).fit(train_x,res)
                def base_predict(items): return [it['baseline']+base_model.predict(it['X']) for it in items]
            else:
                logrow=baseline_log[(baseline_log.horizon_days==h)&(baseline_log.fold==name)&(baseline_log.model=='ml_anchor_blend')].iloc[0]
                selected_mode=str(logrow.selected_ml_mode);alpha=float(logrow.selected_anchor_weight)
                base_model=fit_model(train_x,res,sample_dates,selected_mode,train_y)
                def base_predict(items): return [(1-alpha)*(it['baseline']+base_model.predict(it['X']))+it['anchor']*alpha for it in items]
            cal_items=[days[d].copy() for d in cal_days];val_items=[days[d].copy() for d in val_days]
            for item in cal_items+val_items: item['anchor']=anchor_for(item,series);item['hours']=item['timestamps'].hour.to_numpy()
            cal_base=base_predict(cal_items);val_base=base_predict(val_items)

            # Existing global gate (first-stage experiment) is retained as a comparator.
            first_gate=pd.read_csv(OUT/'peak_hybrid_hourly_predictions.csv')
            for model_kind in ('global','fixed_season','data_cluster'):
                if model_kind=='global':
                    global_models=fit_gate_models(train_x,train_y,peak_train)
                    models={0:global_models};label=lambda month:0;fallback=[]
                else:
                    group_col='fixed_season' if model_kind=='fixed_season' else 'data_season'
                    row_group=train_frame[group_col].to_numpy()
                    models={};fallback=[]
                    for group in sorted(pd.unique(row_group),key=str):
                        mask=row_group==group;gm=fit_gate_models(train_x[mask],train_y[mask],peak_train[mask],
                            seed=42+(list(FIXED_SEASONS.values()).index(group) if model_kind=='fixed_season' else int(group)))
                        models[group]=gm
                    label=(lambda month:FIXED_SEASONS[int(month)]) if model_kind=='fixed_season' else (lambda month:month_map[int(month)])
                def predict_rows(items,bases):
                    output=[]
                    for it,bpred in zip(items,bases):
                        group=label(pd.Timestamp(it['day']).month);clf,reg=models.get(group,models.get(0))
                        output.append(dict(day=it['day'],actual=it['actual'],base_pred=bpred,
                            prob=clf.predict_proba(it['X'])[:,1],peak_pred=reg.predict(it['X']),hours=it['hours'],
                            season=label(pd.Timestamp(it['day']).month),kind=model_kind))
                    return output
                cal_rows=predict_rows(cal_items,cal_base);val_rows=predict_rows(val_items,val_base)
                selected,base_cal,failed,curve=tune_gate(cal_rows)
                curve.to_csv(OUT/f'd{h}_{name}_{model_kind}_calibration_curve.csv',index=False)
                prob_stats=classifier_stats(cal_rows,selected['threshold'])
                calibration.append(dict(horizon_days=h,fold=name,method=model_kind,train_end=fold['train_end'],cal_start=fold['cal_start'],cal_end=fold['cal_end'],
                    gate_threshold=selected['threshold'],correction_scale=selected['scale'],selected_in_calibration=not failed,
                    cal_base_peak_MAE=base_cal['peak_MAE'],cal_selected_peak_MAE=selected['peak_MAE'],
                    cal_base_MAE=base_cal['MAE'],cal_selected_MAE=selected['MAE'],cal_base_normal_MAE=base_cal['normal_MAE'],
                    cal_selected_normal_MAE=selected['normal_MAE'],cal_base_G2V=base_cal['G2V_regret'],cal_selected_G2V=selected['G2V_regret'],
                    calibration_classifier_AP=prob_stats['average_precision'],calibration_classifier_brier=prob_stats['brier'],
                    calibration_false_alarm_rate=prob_stats['false_alarm_rate'],fallback_groups=json.dumps(fallback),
                    fixed_seasons=json.dumps(FIXED_SEASONS),data_clusters=json.dumps({str(k):v for k,v in month_map.items()}),
                    feature_columns='existing V11/V17 feature set; no new month/hour/weekday variables',
                    guardrails='MAE+0.10; RMSE+0.25; nonpeak MAE+0.10; G2V+2; peak MAE must improve',vintage_verified=False))
                cal_model_rows.append(dict(horizon_days=h,fold=name,method=model_kind,partition='calibration',**prob_stats))
                for item in val_rows:
                    pred=item['base_pred']+np.where(item['prob']>=selected['threshold'],selected['scale']*np.maximum(0,item['peak_pred']-item['base_pred']),0.)
                    val_audit=first_gate[(first_gate.horizon_days==h)&(first_gate.fold==name)&(first_gate.target_date==item['day'])&(first_gate.model=='gated_peak_hybrid')]
                    for i,(act,b,prob,price,hyb) in enumerate(zip(item['actual'],item['base_pred'],item['prob'],item['peak_pred'],pred)):
                        season_label=label(pd.Timestamp(item['day']).month)
                        season_pred_rows.append(dict(horizon_days=h,fold=name,target_date=item['day'],hour=int(item['hours'][i]),
                            month=int(pd.Timestamp(item['day']).month),season_rule=model_kind,season_group=season_label,
                            actual_smp=float(act),base_refit_prediction=float(b),peak_probability=float(prob),
                            peak_price_prediction=float(price),prediction=float(hyb),selected_threshold=selected['threshold'],
                            correction_scale=selected['scale'],corrected=bool(prob>=selected['threshold'] and selected['scale']>0),
                            actual_peak=bool(act>=PEAK),evaluation_type='RETROSPECTIVE_FOLD_REPLAY_NOT_OPERATIONAL'))
                    cal_model_rows.append(dict(horizon_days=h,fold=name,method=model_kind,partition='validation',
                        classifier_AP=float(average_precision_score(item['actual']>=PEAK,item['prob'])),
                        classifier_brier=float(brier_score_loss(item['actual']>=PEAK,item['prob'])),
                        threshold=selected['threshold'],false_alarm_rate=float((((item['prob']>=selected['threshold'])&(item['actual']<PEAK)).sum())/max(1,(item['actual']<PEAK).sum()))))

    pd.DataFrame(all_train_dist).to_csv(OUT/'training_season_hour_distribution.csv',index=False)
    pd.DataFrame(month_clusters).to_csv(OUT/'training_month_cluster_map.csv',index=False)
    pd.DataFrame(month_cluster_profiles).to_csv(OUT/'training_month_hour_profiles.csv',index=False)
    pd.DataFrame(season_stability).to_csv(OUT/'training_season_hour_top3.csv',index=False)
    pd.DataFrame(feature_rows).to_csv(OUT/'calendar_feature_ablation.csv',index=False)
    pd.DataFrame(calibration).to_csv(OUT/'seasonal_gate_calibration.csv',index=False)
    pd.DataFrame(cal_model_rows).to_csv(OUT/'seasonal_classifier_calibration_scores.csv',index=False)
    season_pred=pd.DataFrame(season_pred_rows);season_pred.to_csv(OUT/'seasonal_hybrid_hourly_predictions.csv',index=False)

    # Unified same-date summaries include stored V5-V20-era candidates, LightGBM
    # peak-weighted variants, first global gate, and the seasonal interactions.
    rows=[]
    def extend_source(frame,h,models):
        for model in models:
            g=frame[(frame.horizon_days==h)&(frame.model==model)]
            for r in g.itertuples():rows.append(dict(horizon_days=h,fold=r.fold,target_date=r.target_date,hour=int(r.hour),
                actual_smp=float(r.actual_smp),predicted_smp=float(r.predicted_smp),model=model))
    for h in (1,2,3):
        extend_source(old,h,['persistence','weekday_median4','hgb','peak_weighted','ml_anchor_blend'])
        if h==1: extend_source(et,h,['V11_ExtraTrees_fold_refit'])
        else: extend_source(lgb,h,['lightgbm_v12','lightgbm_v12_peak','lightgbm_anchor_blend'])
    global_gate=pd.read_csv(OUT/'peak_hybrid_hourly_predictions.csv')
    # The first-stage file also contains uncorrected/reference model rows.
    # Include only the actual gated candidate to prevent duplicate hourly rows.
    global_gate = global_gate[global_gate.model == 'gated_peak_hybrid']
    for r in global_gate.itertuples():
        rows.append(dict(horizon_days=r.horizon_days,fold=r.fold,target_date=r.target_date,hour=int(r.hour),
            actual_smp=float(r.actual_smp),predicted_smp=float(r.gated_hybrid_prediction),model='global_peak_gate'))
    for r in season_pred.itertuples():
        rows.append(dict(horizon_days=r.horizon_days,fold=r.fold,target_date=r.target_date,hour=int(r.hour),
            actual_smp=float(r.actual_smp),predicted_smp=float(r.prediction),model=f'{r.season_rule}_seasonal_peak_gate'))
    allpred=pd.DataFrame(rows)
    summary=[]
    for (h,fold,model),g in allpred.groupby(['horizon_days','fold','model'],sort=True):
        if len(g)%24 or g.target_date.nunique()!=len(g)//24: continue
        summary.append(dict(horizon_days=h,fold=fold,model=model,**aggregate(g),
            evaluation='same chronological fold dates; retrospective; vintage unverified'))
    fold_summary=pd.DataFrame(summary);fold_summary.to_csv(OUT/'final_same_fold_peak_comparison.csv',index=False)
    pooled=[]
    for (h,model),g in allpred.groupby(['horizon_days','model'],sort=True):
        if len(g)%24: continue
        pooled.append(dict(horizon_days=h,model=model,**aggregate(g),evaluation='pooled 275 chronological fold validation days; no independent unseen claim'))
    pooled=pd.DataFrame(pooled);pooled.to_csv(OUT/'final_pooled_peak_comparison.csv',index=False)
    deltas=[]
    for (h,fold),g in fold_summary.groupby(['horizon_days','fold']):
        base_name='V11_ExtraTrees_fold_refit' if h==1 else 'ml_anchor_blend'
        base=g[g.model==base_name]
        if base.empty: continue
        b=base.iloc[0]
        for _,r in g.iterrows():
            if r.model==base_name: continue
            deltas.append(dict(horizon_days=h,fold=fold,baseline=base_name,model=r.model,
                delta_MAE=r.MAE-b.MAE,delta_RMSE=r.RMSE-b.RMSE,delta_peak_MAE=r.peak_MAE-b.peak_MAE,
                delta_normal_MAE=r.normal_MAE-b.normal_MAE,delta_daily_max_error=r.daily_max_abs_error-b.daily_max_abs_error,
                delta_peak_hour_error=r.daily_peak_hour_error-b.daily_peak_hour_error,delta_G2V_regret=r.G2V_regret-b.G2V_regret,
                assessment='positive means worse; selection used calibration only'))
    pd.DataFrame(deltas).to_csv(OUT/'fold_delta_failure_cases.csv',index=False)

    # Held-out seasonal peak shortfall is diagnostic only, never used for routing.
    held=[]
    for r in allpred.itertuples():
        if r.actual_smp>=PEAK:
            dt=pd.Timestamp(r.target_date)
            held.append(dict(horizon_days=r.horizon_days,fold=r.fold,target_date=r.target_date,month=dt.month,
                season=FIXED_SEASONS[dt.month],hour=r.hour,model=r.model,actual_smp=r.actual_smp,
                prediction=r.predicted_smp,underprediction=max(0,r.actual_smp-r.predicted_smp)))
    held=pd.DataFrame(held);held.groupby(['horizon_days','season','hour','model'],as_index=False).agg(
        peak_hours=('actual_smp','size'),peak_MAE=('prediction',lambda s:np.mean(abs(s-held.loc[s.index,'actual_smp']))),
        mean_underprediction=('underprediction','mean'),underprediction_rate=('underprediction',lambda s:(s>0).mean())).to_csv(OUT/'heldout_peak_error_by_season_hour.csv',index=False)

    # Visualization date picked solely for a representative illustration.
    candidates=allpred[(allpred.horizon_days==2)&(allpred.model=='ml_anchor_blend')]
    peak_count=candidates.groupby('target_date').apply(lambda g:int((g.actual_smp>=PEAK).sum()),include_groups=False)
    representative=str(peak_count.sort_values(ascending=False).index[0]);fold=str(candidates[candidates.target_date==representative].fold.iloc[0])
    fig,ax=plt.subplots(figsize=(13,6));day=candidates[candidates.target_date==representative].sort_values('hour')
    ax.plot(day.hour,day.actual_smp,label='Actual SMP',lw=2.4)
    for model,label_text in [('hgb','HGB'),('peak_weighted','Peak-weighted HGB'),('lightgbm_v12_peak','V12 peak-weighted LightGBM'),('ml_anchor_blend','HGB/anchor'),
                             ('fixed_season_seasonal_peak_gate','Fixed-season gate'),('data_cluster_seasonal_peak_gate','Data-cluster gate')]:
        f=allpred[(allpred.horizon_days==2)&(allpred.fold==fold)&(allpred.target_date==representative)&(allpred.model==model)].sort_values('hour')
        if len(f)==24: ax.plot(f.hour,f.predicted_smp,label=label_text,alpha=.85)
    ax.axhline(PEAK,color='gray',ls=':',label='Peak threshold 150');ax.set(title=f'Representative peak day {representative} · D+2 · fold replay',xlabel='Hour (KST)',ylabel='SMP (KRW/kWh)');ax.grid(alpha=.25);ax.legend(ncol=2);fig.tight_layout();fig.savefig(OUT/'representative_peak_day.png',dpi=160);plt.close(fig)

    manifest=dict(peak_threshold=PEAK,folds=FOLDS,feature_set='unchanged V11 D+1 / V17 D+2,D+3 cyclic time/month/weekday plus historical SMP inputs',
        season_models='fixed DJF/MAM/JJA/SON and KMeans monthly hour-peak profiles (k=4), fit separately inside each training fold',
        no_duplicate_calendar_features=True,cluster_definition='train-only 12x24 monthly peak-rate matrix plus monthly peak-rate; standardize then KMeans seed42 n_init20',
        calibration_guardrails='MAE +0.10; RMSE +0.25; normal MAE +0.10; G2V regret +2; peak MAE must improve; calibration only',
        evaluation_not_used_for_selection=True,historical_first_seen_vintage_verified=False,operating_model_replaced=False,
        EV='60kWh; SOC30->80; 7kW; efficiency0.90; full24h',classifier='balanced HistGradientBoostingClassifier',peak_regressor='peak-only HistGradientBoostingRegressor',
        known_limit='past first_seen/publication vintage and model file availability are unverified')
    (OUT/'seasonal_study_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\nFINAL POOLED SAME-FOLD COMPARISON\n',pooled[['horizon_days','model','MAE','RMSE','peak_MAE','normal_MAE','daily_max_abs_error','daily_peak_hour_error','false_alarm_rate','G2V_regret']].to_string(index=False),flush=True)
    print('\nTRAINING SEASON/HOUR SAMPLE\n',pd.DataFrame(all_train_dist).query("rule=='fixed_season'").groupby(['horizon_days','fold','group']).agg(n=('n_samples','sum'),peak_rate=('peak_hours','sum'),peak_smp_q90=('q90_peak_smp','mean'),peak_uplift=('mean_peak_uplift_over_daily','mean')).head(24).to_string(),flush=True)


if __name__=='__main__': main()
