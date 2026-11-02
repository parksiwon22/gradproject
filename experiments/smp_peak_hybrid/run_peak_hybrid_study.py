"""Fold-safe SMP peak-gating study; does not touch the V20 publisher/gate."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

from v11.store import ROOT
from experiments.smp_performance.train_ml_candidates import (
    FEATURES_BY_H, FOLDS, allocate, day_vectors, features_for, fit_model, load_prices,
)

OUT = ROOT / 'docs' / 'smp_peak_hybrid'
PEAK = 150.0
EV_SETUP = '60kWh; SOC 30->80%; charge 7kW; efficiency 0.90; full 24h'


def daily_metrics(actual, predicted, threshold=PEAK):
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    err = p - a
    event, alert = a >= threshold, p >= threshold
    tp = int((event & alert).sum()); fp = int((~event & alert).sum())
    fn = int((event & ~alert).sum()); tn = int((~event & ~alert).sum())
    peak_err = np.abs(err[event]); normal_err = np.abs(err[~event])
    imax, pmax = int(np.argmax(a)), int(np.argmax(p))
    peak_hour_error = abs(imax - pmax); peak_hour_error = min(peak_hour_error, 24 - peak_hour_error)
    return dict(MAE=float(np.mean(np.abs(err))), RMSE=float(np.sqrt(np.mean(err**2))),
        peak_MAE=float(peak_err.mean()) if event.any() else np.nan,
        normal_MAE=float(normal_err.mean()) if normal_err.size else np.nan,
        daily_max_abs_error=float(abs(np.max(p)-np.max(a))),
        daily_peak_hour_error=float(peak_hour_error), actual_peak_hours=int(event.sum()),
        peak_underpredict_rate=float(np.mean(p[event] < a[event])) if event.any() else np.nan,
        peak_mean_underprediction=float(np.mean(np.maximum(0, a[event]-p[event]))) if event.any() else np.nan,
        tp=tp, fp=fp, fn=fn, tn=tn,
        peak_precision=tp/(tp+fp) if tp+fp else 0.0,
        peak_recall=tp/(tp+fn) if tp+fn else np.nan,
        peak_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
        false_alarm_rate=fp/(fp+tn) if fp+tn else np.nan,
        false_discovery_rate=fp/(fp+tp) if fp+tp else 0.0,
        G2V_regret=float(np.dot(a, allocate(p)-allocate(a))))


def aggregate(group):
    a = group.actual_smp.to_numpy(float); p = group.predicted_smp.to_numpy(float)
    per_day = [daily_metrics(g.actual_smp, g.predicted_smp) for _, g in group.groupby('target_date', sort=True) if len(g) == 24]
    err = p-a; peak = a >= PEAK
    event_pred = p >= PEAK
    tp = int((peak & event_pred).sum()); fp = int((~peak & event_pred).sum())
    fn = int((peak & ~event_pred).sum()); tn = int((~peak & ~event_pred).sum())
    return dict(days=len(per_day), hours=len(a), MAE=float(np.mean(abs(err))), RMSE=float(np.sqrt(np.mean(err**2))),
        peak_MAE=float(np.mean(abs(err[peak]))) if peak.any() else np.nan,
        normal_MAE=float(np.mean(abs(err[~peak]))) if (~peak).any() else np.nan,
        daily_max_abs_error=float(np.mean([m['daily_max_abs_error'] for m in per_day])),
        daily_peak_hour_error=float(np.mean([m['daily_peak_hour_error'] for m in per_day])),
        peak_underpredict_rate=float(np.mean(p[peak] < a[peak])) if peak.any() else np.nan,
        peak_mean_underprediction=float(np.mean(np.maximum(0,a[peak]-p[peak]))) if peak.any() else np.nan,
        peak_hours=int(peak.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
        peak_precision=tp/(tp+fp) if tp+fp else 0., peak_recall=tp/(tp+fn) if tp+fn else np.nan,
        peak_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        false_alarm_rate=fp/(fp+tn) if fp+tn else np.nan,
        false_discovery_rate=fp/(fp+tp) if fp+tp else 0.,
        G2V_regret=float(np.mean([m['G2V_regret'] for m in per_day])))


def anchor_for(item, price_series):
    target = item['timestamps']
    weeks = np.vstack([price_series.reindex(target-pd.Timedelta(days=7*k)).to_numpy(float) for k in range(1,5)])
    return np.median(weeks, axis=0)


def peak_frequency(train_days, days):
    rows=[]
    for day in train_days:
        item=days[day]
        for timestamp, actual in zip(item['timestamps'], item['actual']):
            ts=pd.Timestamp(timestamp)
            rows.append(dict(target_date=day,hour=ts.hour,month=ts.month,weekday=ts.dayofweek,actual_smp=float(actual),is_peak=bool(actual>=PEAK)))
    return pd.DataFrame(rows)


def stats_by_group(frame, h, fold_name):
    outputs=[]
    for field, label in [('hour','hour'),('month','month'),('weekday','weekday')]:
        for value, g in frame.groupby(field, sort=True):
            outputs.append(dict(horizon_days=h,fold=fold_name,group=label,value=int(value),samples=len(g),
                peak_hours=int(g.is_peak.sum()),peak_rate=float(g.is_peak.mean()),
                mean_smp=float(g.actual_smp.mean()),median_smp=float(g.actual_smp.median())))
    return outputs


def candidate_stats(cal_rows, threshold, scale, fixed_hour_set=None, offset_by_hour=None):
    forecasts=[]
    for row in cal_rows:
        delta=np.maximum(0.,row['peak_pred']-row['base_pred'])
        if fixed_hour_set is not None:
            correction=np.array([offset_by_hour.get(int(hour),0.) if int(hour) in fixed_hour_set else 0. for hour in row['hours']])
            pred=row['base_pred']+scale*correction
        else:
            pred=row['base_pred']+np.where(row['prob']>=threshold,scale*delta,0.)
        forecasts.append((row['actual'],pred))
    daily=[daily_metrics(a,p) for a,p in forecasts]
    totals=dict(MAE=float(np.mean([x['MAE'] for x in daily])),RMSE=float(np.sqrt(np.mean([x['RMSE']**2 for x in daily]))),
        peak_MAE=float(np.nanmean([x['peak_MAE'] for x in daily])),normal_MAE=float(np.mean([x['normal_MAE'] for x in daily])),
        G2V_regret=float(np.mean([x['G2V_regret'] for x in daily])))
    return totals


def choose_calibrated(cal_rows, base_rows, fixed=False, offset_by_hour=None, fixed_hour_set=None):
    base_day=[daily_metrics(r['actual'],r['base_pred']) for r in base_rows]
    ref=dict(MAE=float(np.mean([x['MAE'] for x in base_day])),
        RMSE=float(np.sqrt(np.mean([x['RMSE']**2 for x in base_day]))),
        peak_MAE=float(np.nanmean([x['peak_MAE'] for x in base_day])),
        normal_MAE=float(np.mean([x['normal_MAE'] for x in base_day])),
        G2V_regret=float(np.mean([x['G2V_regret'] for x in base_day])))
    candidates=[]
    if fixed:
        for scale in np.linspace(0,1.5,16):
            s=candidate_stats(cal_rows,0,scale,fixed_hour_set,offset_by_hour)
            candidates.append(dict(threshold=np.nan,scale=float(scale),**s))
    else:
        thresholds=sorted(set([.05,.1,.15,.2,.25,.3,.4,.5,.6,.7,.8,.9]+[
            float(np.quantile(np.concatenate([r['prob'] for r in cal_rows]),q)) for q in (.5,.75,.9,.95)]))
        for threshold in thresholds:
            for scale in (.25,.5,.75,1.,1.25,1.5):
                s=candidate_stats(cal_rows,threshold,scale)
                candidates.append(dict(threshold=threshold,scale=scale,**s))
    # Peak improvement is pursued only within mild price/regret guardrails.
    eligible=[r for r in candidates if r['peak_MAE'] < ref['peak_MAE']
        and r['MAE'] <= ref['MAE']+.25 and r['RMSE'] <= ref['RMSE']+.5
        and r['normal_MAE'] <= ref['normal_MAE']+.15 and r['G2V_regret'] <= ref['G2V_regret']+5]
    if eligible:
        selected=min(eligible,key=lambda r:(r['peak_MAE'],r['MAE'],r['G2V_regret'],r['RMSE']))
        return selected,ref,False,pd.DataFrame(candidates)
    # No calibration-safe peak improvement: preserve the unmodified base.
    zero=next(r for r in candidates if r['scale']==0)
    return zero,ref,True,pd.DataFrame(candidates)


def model_for_horizon(h, fold_name, train_days, days):
    x=np.concatenate([days[d]['X'] for d in train_days]); actual=np.concatenate([days[d]['actual'] for d in train_days])
    base=np.concatenate([days[d]['baseline'] for d in train_days]); residual=actual-base
    sample_dates=np.repeat(train_days,24)
    if h==1:
        model=ExtraTreesRegressor(n_estimators=150,min_samples_leaf=10,max_features=.8,n_jobs=4,random_state=42)
        model.fit(x,residual)
        return lambda items:[i['baseline']+model.predict(i['X']) for i in items],model
    log=pd.read_csv(ROOT/'docs'/'smp_improvement_ml'/'experiment_log.csv')
    row=log[(log.horizon_days==h)&(log.fold==fold_name)&(log.model=='ml_anchor_blend')].iloc[0]
    mode=str(row.selected_ml_mode); alpha=float(row.selected_anchor_weight)
    model=fit_model(x,residual,sample_dates,mode,actual)
    def predict(items):
        return [(1-alpha)*(i['baseline']+model.predict(i['X']))+alpha*i['anchor'] for i in items]
    return predict,model


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    prices=load_prices(); price_series=prices.set_index('timestamp').smp.sort_index()
    maps={h:day_vectors(features_for(prices,h),h) for h in (1,2,3)}
    base_scores=pd.read_csv(ROOT/'docs'/'smp_improvement_ml_lgbm'/'all_fold_paired_summary.csv')
    old_hour=pd.read_csv(ROOT/'docs'/'smp_improvement_ml'/'hourly_predictions.csv')
    et_hour=pd.read_csv(ROOT/'docs'/'smp_improvement_ml_lgbm'/'d1_v11_extratrees_fold_hourly.csv')
    fold_metrics=[]; freq_rows=[]; stability=[]; decomposition=[]; hourly_preds=[]; calibration_log=[]

    for h in (1,2,3):
        days=maps[h]
        for fold in FOLDS:
            train_days=[d for d in sorted(days) if d<=fold['train_end']]
            cal_days=[d for d in sorted(days) if fold['cal_start']<=d<=fold['cal_end']]
            val_days=[d for d in sorted(days) if fold['val_start']<=d<=fold['val_end']]
            train_frame=peak_frequency(train_days,days)
            freq_rows.extend(stats_by_group(train_frame,h,fold['name']))
            hr=train_frame.groupby('hour').is_peak.mean().sort_values(ascending=False,kind='stable')
            top_hours=set(map(int,hr.head(5).index))
            stability.append(dict(horizon_days=h,fold=fold['name'],train_end=fold['train_end'],
                train_hours=len(train_frame),train_peak_hours=int(train_frame.is_peak.sum()),
                peak_rate=float(train_frame.is_peak.mean()),top3_hours=','.join(map(str,hr.head(3).index)),
                top5_hours=','.join(map(str,hr.head(5).index)),top5_peak_rates=','.join(f'{x:.4f}' for x in hr.head(5).values)))
            # Existing feature builders encode issue-time price history only.
            for group_days in (cal_days,val_days):
                for d in group_days:
                    item=days[d]
                    if item['origin'] != pd.Timestamp(d,tz='Asia/Seoul')-pd.Timedelta(days=h)+pd.Timedelta(hours=23):
                        raise AssertionError(f'origin timestamp mismatch: D+{h} {d}')
                    if (pd.DatetimeIndex(item['timestamps'])-pd.Timedelta(days=h)).max()>item['origin']:
                        raise AssertionError('feature uses SMP beyond 23:00 issue origin')
                    item['anchor']=anchor_for(item,price_series)

            cal_items=[days[d] for d in cal_days]; val_items=[days[d] for d in val_days]
            base_predict,base_model=model_for_horizon(h,fold['name'],train_days,days)
            cal_base=base_predict(cal_items); val_base=base_predict(val_items)
            # Existing validation vectors are the locked comparator. Retain raw
            # HGB, peak-weighted HGB, anchor and final candidate as separate rows.
            comparator_names=['persistence','weekday_median4','hgb','peak_weighted','ml_anchor_blend']
            comp_source=et_hour if h==1 else old_hour
            if h==1: comparator_names+=['V11_ExtraTrees_fold_refit']
            val_records=[]
            for d,item,bpred in zip(val_days,val_items,val_base):
                old_day=comp_source[(comp_source.horizon_days==h)&(comp_source.fold==fold['name'])&(comp_source.target_date==d)]
                if h==1:
                    old_day=old_hour[(old_hour.horizon_days==h)&(old_hour.fold==fold['name'])&(old_hour.target_date==d)]
                for name in comparator_names:
                    if name=='V11_ExtraTrees_fold_refit':
                        src=et_hour[(et_hour.fold==fold['name'])&(et_hour.target_date==d)]
                    else:
                        src=old_day[old_day.model==name]
                    if len(src)!=24: continue
                    val_records.extend(dict(horizon_days=h,fold=fold['name'],target_date=d,hour=int(r.hour),
                        actual_smp=float(r.actual_smp),predicted_smp=float(r.predicted_smp),model=name,
                        evaluation_type='RETROSPECTIVE_FOLD_REPLAY_NOT_OPERATIONAL') for r in src.itertuples())

            # Separate peak classification and peak-only level regression.
            X=np.concatenate([days[d]['X'] for d in train_days]); y=np.concatenate([days[d]['actual'] for d in train_days])
            is_peak=y>=PEAK
            if is_peak.sum()<40: raise RuntimeError(f'too few training peaks for D+{h} {fold["name"]}: {is_peak.sum()}')
            clf=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=7,learning_rate=.05,
                min_samples_leaf=20,l2_regularization=3.0,early_stopping=False,random_state=42,class_weight='balanced')
            clf.fit(X,is_peak.astype(int))
            peak_reg=HistGradientBoostingRegressor(max_iter=180,max_leaf_nodes=7,learning_rate=.05,
                min_samples_leaf=12,l2_regularization=5.0,early_stopping=False,random_state=42)
            peak_reg.fit(X[is_peak],y[is_peak])
            train_peak_model=dict(classifier=clf,peak_regressor=peak_reg,horizon_days=h,threshold=PEAK,
                train_end=fold['train_end'],features=FEATURES_BY_H[h],trained_only_on_pre_validation_fold=True,
                historical_first_seen_vintage_verified=False,operating_model_replacement=False)
            joblib.dump(train_peak_model,OUT/f'd{h}_{fold["name"]}_peak_classifier_regressor.joblib')

            cal_rows=[]
            for item,bpred in zip(cal_items,cal_base):
                prob=clf.predict_proba(item['X'])[:,1];peak_pred=peak_reg.predict(item['X'])
                cal_rows.append(dict(day=item['day'],actual=item['actual'],base_pred=bpred,prob=prob,peak_pred=peak_pred,hours=item['hours']))
            # Candidate fixed-clock hours are derived from training labels only.
            pos=train_frame[train_frame.is_peak]
            # Estimate conditional peak uplift from daily-baseline inputs for each
            # training hour. This is a fixed correction, not a validation-derived hour map.
            train_rows=[]
            for d in train_days:
                item=days[d]
                for hour,(actual,base_value) in enumerate(zip(item['actual'],item['baseline'])):
                    if actual>=PEAK: train_rows.append(dict(hour=hour,uplift=float(actual-base_value)))
            train_uplift=pd.DataFrame(train_rows)
            offset_by_hour={int(k):float(g.uplift.mean()) for k,g in train_uplift.groupby('hour')} if len(train_uplift) else {}

            chosen,cal_ref,failed,curve=choose_calibrated(cal_rows,cal_rows)
            curve.to_csv(OUT/f'd{h}_{fold["name"]}_gated_calibration.csv',index=False)
            calibration_log.append(dict(horizon_days=h,fold=fold['name'],candidate='gated_peak_probability',
                trained_through=fold['train_end'],cal_start=fold['cal_start'],cal_end=fold['cal_end'],
                classifier='HistGradientBoostingClassifier balanced weights',peak_regressor='peak-only HistGradientBoostingRegressor',
                selected_threshold=chosen['threshold'],selected_scale=chosen['scale'],calibration_peak_MAE_before=cal_ref['peak_MAE'],
                calibration_MAE_before=cal_ref['MAE'],calibration_peak_MAE_after=chosen['peak_MAE'],
                calibration_MAE_after=chosen['MAE'],calibration_G2V_before=cal_ref['G2V_regret'],
                calibration_G2V_after=chosen['G2V_regret'],rejected_for_guardrail=failed,
                training_positive_hours=int(is_peak.sum()),candidate_threshold='calibration only',vintage_verified=False))

            # Fixed clock correction: top 3/5/8 hours come only from training peak
            # frequency; correction amount and scale are calibrated on the prior quarter.
            fixed_results=[]
            for k in (3,5,8):
                selected_hours=set(map(int,hr.head(k).index))
                fixed,ref,failed_fixed,fixed_curve=choose_calibrated(cal_rows,cal_rows,fixed=True,
                    offset_by_hour=offset_by_hour,fixed_hour_set=selected_hours)
                fixed_curve.to_csv(OUT/f'd{h}_{fold["name"]}_fixed_top{k}_calibration.csv',index=False)
                fixed_results.append((k,selected_hours,fixed,failed_fixed,ref))
                calibration_log.append(dict(horizon_days=h,fold=fold['name'],candidate=f'fixed_top{k}_hour_correction',
                    trained_through=fold['train_end'],cal_start=fold['cal_start'],cal_end=fold['cal_end'],classifier='none; fixed train-frequency hours',
                    peak_regressor='none; train peak-hour uplift',selected_threshold=np.nan,selected_scale=fixed['scale'],
                    selected_hours=','.join(map(str,sorted(selected_hours))),calibration_peak_MAE_before=ref['peak_MAE'],
                    calibration_MAE_before=ref['MAE'],calibration_peak_MAE_after=fixed['peak_MAE'],calibration_MAE_after=fixed['MAE'],
                    calibration_G2V_before=ref['G2V_regret'],calibration_G2V_after=fixed['G2V_regret'],rejected_for_guardrail=failed_fixed,
                    training_positive_hours=int(is_peak.sum()),candidate_threshold='calibration only',vintage_verified=False))

            # Validation: no threshold/hour/model parameter is chosen here.
            for d,item,bpred in zip(val_days,val_items,val_base):
                prob=clf.predict_proba(item['X'])[:,1];peak_pred=peak_reg.predict(item['X'])
                # Use the preserved baseline candidate forecast for exact paired
                # comparison; correction uses the fold-only classifier/regressor.
                saved_base=old_hour[(old_hour.horizon_days==h)&(old_hour.fold==fold['name'])&
                    (old_hour.target_date==d)&(old_hour.model=='ml_anchor_blend')].sort_values('hour').predicted_smp.to_numpy(float)
                if h==1:
                    saved_base=et_hour[(et_hour.fold==fold['name'])&(et_hour.target_date==d)].sort_values('hour').predicted_smp.to_numpy(float)
                if len(saved_base)!=24: saved_base=bpred
                gate=saved_base+np.where(prob>=chosen['threshold'],chosen['scale']*np.maximum(0.,peak_pred-saved_base),0.)
                hourly_preds.extend(dict(horizon_days=h,fold=fold['name'],target_date=d,hour=int(hour),actual_smp=float(actual),
                    base_prediction=float(base),peak_probability=float(pr),peak_regression_prediction=float(reg),
                    gated_hybrid_prediction=float(hyp),model='gated_peak_hybrid',selected_threshold=chosen['threshold'],
                    selected_scale=chosen['scale'],evaluation_type='RETROSPECTIVE_FOLD_REPLAY_NOT_OPERATIONAL')
                    for hour,actual,base,pr,reg,hyp in zip(item['hours'],item['actual'],saved_base,prob,peak_pred,gate))
                for k,hour_set,fixed,failed_fixed,ref in fixed_results:
                    fixed_pred=saved_base+fixed['scale']*np.array([offset_by_hour.get(int(hour),0.) if int(hour) in hour_set else 0. for hour in item['hours']])
                    hourly_preds.extend(dict(horizon_days=h,fold=fold['name'],target_date=d,hour=int(hour),actual_smp=float(actual),
                        predicted_smp=float(pred),model=f'fixed_top{k}_hour_correction',selected_hours=','.join(map(str,sorted(hour_set))),
                        selected_scale=fixed['scale'],evaluation_type='RETROSPECTIVE_FOLD_REPLAY_NOT_OPERATIONAL')
                        for hour,actual,pred in zip(item['hours'],item['actual'],fixed_pred))
                # Keep training data and component errors distinct for analysis.
                for name, pred in [('base_selected',saved_base),('hgb_or_ET_unblended',bpred),
                                   ('seasonal_anchor',item['anchor']),('peak_weighted_existing',None)]:
                    if name=='peak_weighted_existing':
                        src=old_hour[(old_hour.horizon_days==h)&(old_hour.fold==fold['name'])&
                            (old_hour.target_date==d)&(old_hour.model=='peak_weighted')]
                        pred=src.sort_values('hour').predicted_smp.to_numpy(float)
                    if pred is None or len(pred)!=24: continue
                    for hour,actual,estimate in zip(item['hours'],item['actual'],pred):
                        if actual>=PEAK:
                            decomposition.append(dict(horizon_days=h,fold=fold['name'],target_date=d,hour=int(hour),
                                actual_smp=float(actual),model=name,predicted_smp=float(estimate),
                                error=float(estimate-actual),underprediction=float(max(0.,actual-estimate)),
                                evaluation_type='held-out fold; diagnostic only'))
            print(f'D+{h} {fold["name"]}: train peaks={int(is_peak.sum())}; gated threshold={chosen["threshold"]:.3f}, scale={chosen["scale"]:.2f}, rejected={failed}',flush=True)

            # Save training-only frequency rows for audit. Validation rows must not enter them.
            for topfold in stability:
                if topfold['horizon_days']==h and topfold['fold']==fold['name']:
                    topfold['top_hours_source']='training labels only; no validation scan'

    pd.DataFrame(freq_rows).to_csv(OUT/'training_peak_frequency.csv',index=False)
    stability_df=pd.DataFrame(stability)
    stability_df.to_csv(OUT/'training_peak_hour_stability.csv',index=False)
    pd.DataFrame(decomposition).to_csv(OUT/'heldout_peak_underprediction_decomposition.csv',index=False)
    pd.DataFrame(hourly_preds).to_csv(OUT/'peak_hybrid_hourly_predictions.csv',index=False)
    pd.DataFrame(calibration_log).to_csv(OUT/'peak_hybrid_calibration_log.csv',index=False)

    # Unified held-out scores. Existing variants are read from the preserved
    # predictions; new variants are applied to those same dates and EV assumptions.
    pred_groups=[]
    for h in (1,2,3):
        src=et_hour if h==1 else old_hour
        names=['persistence','weekday_median4','hgb','peak_weighted','ml_anchor_blend'] if h>1 else ['persistence','weekday_median4','hgb','peak_weighted','ml_anchor_blend']
        for name in names:
            f=src[(src.horizon_days==h)&(src.model==name)]
            for r in f.itertuples():
                pred_groups.append(dict(horizon_days=h,fold=r.fold,target_date=r.target_date,hour=r.hour,
                    actual_smp=r.actual_smp,predicted_smp=r.predicted_smp,model=name))
        if h==1:
            f=et_hour[et_hour.model=='V11_ExtraTrees_fold_refit']
            for r in f.itertuples(): pred_groups.append(dict(horizon_days=h,fold=r.fold,target_date=r.target_date,hour=r.hour,actual_smp=r.actual_smp,predicted_smp=r.predicted_smp,model='V11_ExtraTrees_fold_refit'))
    hy=pd.DataFrame(hourly_preds)
    for name in hy.model.unique():
        f=hy[hy.model==name]
        for r in f.itertuples(): pred_groups.append(dict(horizon_days=r.horizon_days,fold=r.fold,target_date=r.target_date,hour=r.hour,actual_smp=r.actual_smp,predicted_smp=r.gated_hybrid_prediction if name=='gated_peak_hybrid' else r.predicted_smp,model=name))
    all_pred=pd.DataFrame(pred_groups)
    summaries=[]
    for (h,fold,name),g in all_pred.groupby(['horizon_days','fold','model'],sort=True):
        if len(g)%24 or g.target_date.nunique()!=len(g)//24: continue
        summaries.append(dict(horizon_days=h,fold=fold,model=name,**aggregate(g),evaluation='same 275-date folds, timestamp-causal retrospective; first_seen vintage unverified'))
    summary=pd.DataFrame(summaries)
    summary.to_csv(OUT/'peak_hybrid_fold_model_comparison.csv',index=False)
    pooled=[]
    for (h,name),g in all_pred.groupby(['horizon_days','model'],sort=True):
        if len(g)%24: continue
        pooled.append(dict(horizon_days=h,model=name,**aggregate(g),evaluation='pooled held-out folds; not an independent unseen test'))
    pooled=pd.DataFrame(pooled); pooled.to_csv(OUT/'peak_hybrid_pooled_comparison.csv',index=False)

    # Relative consistency of training-selected top-three peak hours across folds.
    stab=[]
    for h,g in stability_df.groupby('horizon_days'):
        sets=[set(map(int,str(v).split(','))) for v in g.top3_hours]
        jac=[len(a&b)/len(a|b) for i,a in enumerate(sets) for b in sets[i+1:]]
        stab.append(dict(horizon_days=h,mean_pairwise_top3_jaccard=float(np.mean(jac)) if jac else np.nan,
            fold_top3_overlap_hours=len(set.intersection(*sets)) if sets else 0,
            interpretation='train-only frequency; stability is descriptive, not validation-selected'))
    pd.DataFrame(stab).to_csv(OUT/'training_peak_hour_consistency.csv',index=False)

    # Typical peak case: choose a held-out date with >=2 peak hours for display,
    # never for model selection. Include the hybrid only where it has predictions.
    base_path=ROOT/'docs'/'smp_improvement_ml_lgbm'/'final_fold_demo.csv.gz'
    demo=pd.read_csv(base_path)
    ev=demo[(demo.horizon_days==2)&(demo.series=='실제 SMP')].groupby('target_date').smp.apply(lambda x:int((x>=PEAK).sum()))
    peak_dates=ev[ev>=2]
    representative=str(peak_dates.sort_values(ascending=False).index[0]) if len(peak_dates) else str(ev.sort_values(ascending=False).index[0])
    h=2; fold=next(f['name'] for f in FOLDS if f['val_start']<=representative<=f['val_end'])
    actual=demo[(demo.horizon_days==h)&(demo.target_date==representative)&(demo.series=='실제 SMP')].sort_values('hour')
    hybrid=hy[(hy.horizon_days==h)&(hy.fold==fold)&(hy.target_date==representative)&(hy.model=='gated_peak_hybrid')].sort_values('hour')
    fig,ax=plt.subplots(figsize=(13,5.5));ax.plot(actual.hour,actual.smp,label='실제 SMP',lw=2.2)
    for model,name in [('hgb','HGB'),('peak_weighted','피크 가중 HGB'),('ml_anchor_blend','HGB/anchor')]:
        f=old_hour[(old_hour.horizon_days==h)&(old_hour.fold==fold)&(old_hour.target_date==representative)&(old_hour.model==model)].sort_values('hour')
        if len(f)==24: ax.plot(f.hour,f.predicted_smp,label=name,alpha=.8)
    if len(hybrid)==24: ax.plot(hybrid.hour,hybrid.gated_hybrid_prediction,label='피크 게이트 혼합',lw=2,linestyle='--')
    ax.axhline(PEAK,color='gray',linestyle=':',label='피크 기준 150');ax.set(title=f'피크 시연 · {representative} · D+{h} · retrospective fold replay',xlabel='시각(KST)',ylabel='SMP (원/kWh)');ax.grid(alpha=.25);ax.legend(ncol=3);fig.tight_layout();fig.savefig(OUT/'representative_peak_day.png',dpi=160);plt.close(fig)

    failures=summary.merge(pooled[['horizon_days','model','peak_MAE','MAE','RMSE','normal_MAE','G2V_regret']],on=['horizon_days','model'],suffixes=('_fold','_overall'))
    failures.to_csv(OUT/'peak_hybrid_failure_cases.csv',index=False)
    manifest=dict(peak_threshold=PEAK,folds=FOLDS,EV=EV_SETUP,features='V11 for D+1; V17 shared-origin for D+2/D+3; 23:00 KST cutoff enforced',
        classifier=dict(model='HistGradientBoostingClassifier',max_iter=180,max_leaf_nodes=7,learning_rate=.05,min_samples_leaf=20,class_weight='balanced'),
        peak_regressor=dict(model='HistGradientBoostingRegressor',max_iter=180,max_leaf_nodes=7,learning_rate=.05,min_samples_leaf=12,fit_rows='training SMP >= 150 only'),
        calibration='Only each fold preceding quarter selects probability threshold and correction scale; guardrails keep calibration MAE/RMSE/normal MAE/G2V near base.',
        fixed_hour='Top 3/5/8 hours selected by that fold training peak rates; per-hour uplift from training peaks; scale selected on calibration only.',
        vintage_verified=False,validation_used_for_model_selection=False,operating_model_replaced=False,
        known_limit='historical first_seen publication vintage and checkpoint existence are not verified')
    (OUT/'study_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\nPOOLED COMPARISON\n',pooled.to_string(index=False),flush=True)
    print('\nTRAIN PEAK FREQUENCY\n',stability_df.to_string(index=False),flush=True)


if __name__=='__main__': main()
