"""Fold-safe LightGBM D+2/D+3 and D+1 ExtraTrees paired study.

Run in .venv-smp-research. All new outputs are isolated from V5-V20 artifacts.
Historical price receipts are not vintage-verifiable; this is timestamp-causal
retrospective replay, not a historical operational forecast.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor

from v11.features import FEATURES as D1_FEATURES, build as build_d1
from v11.store import ROOT, connect, latest
from v17.features import FEATURES as DX_FEATURES, build as build_dx

OUT = ROOT / 'docs' / 'smp_improvement_ml_lgbm'
FOLDS = [
    dict(name='fall2024', train_end='2024-06-30', cal_start='2024-07-01', cal_end='2024-09-30', val_start='2024-10-01', val_end='2024-12-31'),
    dict(name='spring2025', train_end='2024-12-31', cal_start='2025-01-01', cal_end='2025-03-31', val_start='2025-04-01', val_end='2025-06-30'),
    dict(name='fall2025', train_end='2025-06-30', cal_start='2025-07-01', cal_end='2025-09-30', val_start='2025-10-01', val_end='2025-12-31'),
]
EV_CAPACITY, EV_INITIAL, EV_TARGET, EV_POWER, EV_EFFICIENCY = 60.0, 30.0, 80.0, 7.0, .9
FEATURES = {1: D1_FEATURES, 2: DX_FEATURES, 3: DX_FEATURES}


def load_prices():
    db = connect()
    try:
        result = latest(db, 'land')
    finally:
        db.close()
    return result.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)


def build_days(prices, horizon):
    frame = build_d1(prices, days_ahead=1) if horizon == 1 else build_dx(prices, days_ahead=horizon)
    days = {}
    for day, group in frame.groupby('target_date', sort=True):
        group = group.sort_values('timestamp')
        if len(group) != 24:
            continue
        origin = pd.Timestamp(group.origin_timestamp.iloc[0])
        expected = pd.Timestamp(day, tz='Asia/Seoul') - pd.Timedelta(days=horizon) + pd.Timedelta(hours=23)
        if origin != expected:
            raise AssertionError(f'origin mismatch {day} D+{horizon}: {origin} != {expected}')
        target = pd.DatetimeIndex(group.timestamp)
        latest_lag = target - pd.Timedelta(days=horizon)
        if latest_lag.max() > origin:
            raise AssertionError(f'future price timestamp in features {day} D+{horizon}')
        days[str(day)] = dict(day=str(day), actual=group.actual_smp.to_numpy(float),
            base=group.daily.to_numpy(float), x=group[FEATURES[horizon]].to_numpy(float),
            target=target, origin=origin, hour=target.hour.to_numpy())
    return days


def charge_schedule(prices):
    needed = EV_CAPACITY * (EV_TARGET - EV_INITIAL) / 100 / EV_EFFICIENCY
    charge = np.zeros(24)
    for h in np.argsort(prices, kind='stable'):
        amount = min(EV_POWER, needed)
        charge[h] = amount
        needed -= amount
        if needed <= 1e-10:
            break
    return charge


def score(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    error = predicted - actual
    peak = actual >= 150
    regret = float(np.dot(actual, charge_schedule(predicted) - charge_schedule(actual)))
    return dict(MAE=float(np.mean(np.abs(error))), RMSE=float(np.sqrt(np.mean(error ** 2))),
                G2V_regret=regret, peak_MAE=float(np.mean(np.abs(error[peak]))) if peak.any() else np.nan,
                peak_hours=int(peak.sum()))


def make_model(kind):
    # Exact V12 LightGBM factory values. The peak variant uses V12's registered
    # 6x weight for actual >=150 KRW/kWh; no validation labels enter fitting.
    return LGBMRegressor(n_estimators=300, learning_rate=.05, num_leaves=31,
        random_state=42, n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True)


def blend_weight(cal, predictions):
    rows = []
    base_mae = float(np.mean([x['base_score']['MAE'] for x in cal]))
    base_regret = float(np.mean([x['base_score']['G2V_regret'] for x in cal]))
    for weight in np.linspace(0, 1, 21):
        stats = [score(item['actual'], (1-weight)*pred+weight*item['anchor'])
                 for item, pred in zip(cal, predictions)]
        rows.append(dict(weight=float(weight), MAE=float(np.mean([s['MAE'] for s in stats])),
            RMSE=float(np.sqrt(np.mean([s['RMSE']**2 for s in stats]))),
            G2V_regret=float(np.mean([s['G2V_regret'] for s in stats])),
            baseline_MAE=base_mae, baseline_G2V_regret=base_regret))
    eligible = [r for r in rows if r['MAE'] <= base_mae + 1e-10 and r['G2V_regret'] <= base_regret + 1e-10]
    best = min(eligible, key=lambda r: (r['MAE'], r['G2V_regret'], r['RMSE'])) if eligible else {'weight': 0.0}
    return float(best['weight']), rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    daymaps = {h: build_days(prices, h) for h in (1, 2, 3)}
    lgb_daily, lgb_hourly, extratrees_daily, extratrees_hourly, log = [], [], [], [], []
    raw_series = prices.set_index('timestamp').smp.sort_index()

    # Refit the V11 ExtraTrees family on each proper historical fold, allowing
    # direct same-date D+1 comparison with the prior HGB experiments.
    for fold in FOLDS:
        h, days = 1, daymaps[1]
        train_days = [d for d in sorted(days) if d <= fold['train_end']]
        val_days = [d for d in sorted(days) if fold['val_start'] <= d <= fold['val_end']]
        x = np.concatenate([days[d]['x'] for d in train_days]); y = np.concatenate([days[d]['actual']-days[d]['base'] for d in train_days])
        model = ExtraTreesRegressor(n_estimators=150, min_samples_leaf=10, max_features=.8, n_jobs=4, random_state=42)
        model.fit(x, y)
        for day in val_days:
            item = days[day]; pred = item['base'] + model.predict(item['x']); s = score(item['actual'], pred)
            extratrees_daily.append(dict(horizon_days=1,fold=fold['name'],target_date=day,model='V11_ExtraTrees_fold_refit',**s,
                train_end=fold['train_end'],validation_start=fold['val_start'],validation_end=fold['val_end'],
                nominal_origin=item['origin'].isoformat(),timestamp_cutoff_verified=True,first_seen_vintage_verified=False,
                evaluation_type='fold-refit retrospective replay; not historical operational forecast'))
            for hour in range(24): extratrees_hourly.append(dict(horizon_days=1,fold=fold['name'],target_date=day,hour=hour,
                actual_smp=float(item['actual'][hour]),predicted_smp=float(pred[hour]),error=float(pred[hour]-item['actual'][hour]),model='V11_ExtraTrees_fold_refit'))

    for h in (2, 3):
        days = daymaps[h]
        for fold in FOLDS:
            train_days = [d for d in sorted(days) if d <= fold['train_end']]
            cal_days = [d for d in sorted(days) if fold['cal_start'] <= d <= fold['cal_end']]
            val_days = [d for d in sorted(days) if fold['val_start'] <= d <= fold['val_end']]
            if min(len(train_days), len(cal_days), len(val_days)) < 80:
                raise RuntimeError(f'incomplete {fold["name"]} D+{h} split')
            train_x = np.concatenate([days[d]['x'] for d in train_days])
            train_residual = np.concatenate([days[d]['actual']-days[d]['base'] for d in train_days])
            train_actual = np.concatenate([days[d]['actual'] for d in train_days])
            cal = [days[d].copy() for d in cal_days]
            val = [days[d] for d in val_days]
            for item in cal + val:
                item['anchor'] = np.median(np.vstack([raw_series.reindex(item['target']-pd.Timedelta(days=7*k)).to_numpy(float)
                                                       for k in (1,2,3,4)]), axis=0)
                if not np.isfinite(item['anchor']).all(): raise RuntimeError(f'missing anchor {item["day"]}')
            for item in cal: item['base_score'] = score(item['actual'], item['base'])
            predictions = {}
            selected_modes = []
            for mode in ('lightgbm_v12', 'lightgbm_v12_peak'):
                est = make_model(mode)
                weights = np.where(train_actual >= 150, 6.0, 1.0) if mode.endswith('_peak') else None
                if weights is None: est.fit(train_x, train_residual)
                else: est.fit(train_x, train_residual, sample_weight=weights)
                cal_preds = [i['base'] + est.predict(i['x']) for i in cal]
                val_preds = [i['base'] + est.predict(i['x']) for i in val]
                predictions[mode] = (cal_preds, val_preds)
                joblib.dump(dict(model=est,horizon_days=h,mode=mode,features=FEATURES[h],train_end=fold['train_end'],
                                 train_days=len(train_days),feature_origin='23:00 KST timestamp-causal price-only',
                                 historical_input_vintage_verified=False), OUT/f'd{h}_{fold["name"]}_{mode}.joblib')
                stats = [score(i['actual'], p) for i,p in zip(cal,cal_preds)]
                selected_modes.append((float(np.mean([s['MAE'] for s in stats])),
                    float(np.mean([s['G2V_regret'] for s in stats])),float(np.sqrt(np.mean([s['RMSE']**2 for s in stats]))),mode))
            base_mae = float(np.mean([i['base_score']['MAE'] for i in cal])); base_regret=float(np.mean([i['base_score']['G2V_regret'] for i in cal]))
            eligible=[r for r in selected_modes if r[0]<=base_mae+1e-10 and r[1]<=base_regret+1e-10]
            selected_mode=min(eligible)[-1] if eligible else 'lightgbm_v12'
            alpha, curve = blend_weight(cal, predictions[selected_mode][0])
            pd.DataFrame(curve).to_csv(OUT/f'd{h}_{fold["name"]}_blend_calibration.csv',index=False)
            for name in ('lightgbm_v12','lightgbm_v12_peak','lightgbm_anchor_blend'):
                byday=[]
                for j,item in enumerate(val):
                    if name=='persistence': pred=item['base']
                    elif name=='weekday_median4': pred=item['anchor']
                    elif name=='lightgbm_v12': pred=predictions['lightgbm_v12'][1][j]
                    elif name=='lightgbm_v12_peak': pred=predictions['lightgbm_v12_peak'][1][j]
                    else: pred=(1-alpha)*predictions[selected_mode][1][j]+alpha*item['anchor']
                    stats=score(item['actual'],pred); byday.append(stats)
                    lgb_daily.append(dict(horizon_days=h,fold=fold['name'],target_date=item['day'],model=name,**stats,
                        selected_mode=selected_mode,anchor_weight=alpha,train_end=fold['train_end'],calibration_start=fold['cal_start'],
                        calibration_end=fold['cal_end'],validation_start=fold['val_start'],validation_end=fold['val_end'],
                        nominal_origin=item['origin'].isoformat(),timestamp_cutoff_verified=True,first_seen_vintage_verified=False,
                        evaluation_type='fold-refit retrospective replay; not historical operational forecast'))
                    for hour in range(24): lgb_hourly.append(dict(horizon_days=h,fold=fold['name'],target_date=item['day'],hour=hour,
                        actual_smp=float(item['actual'][hour]),predicted_smp=float(pred[hour]),error=float(pred[hour]-item['actual'][hour]),model=name))
                log.append(dict(horizon_days=h,fold=fold['name'],status='success',model=name,selected_mode=selected_mode,
                    anchor_weight=alpha,train_days=len(train_days),calibration_days=len(cal),validation_days=len(val),
                    calibration_baseline_MAE=base_mae,calibration_baseline_G2V_regret=base_regret,
                    validation_MAE=float(np.mean([x['MAE'] for x in byday])),
                    validation_RMSE_pooled=float(np.sqrt(np.mean([x['RMSE']**2 for x in byday]))),
                    validation_G2V_regret=float(np.mean([x['G2V_regret'] for x in byday])),
                    vintage_verified=False))
            print(f'D+{h} {fold["name"]}: selected {selected_mode}, blend alpha={alpha:.2f}',flush=True)

    pd.DataFrame(lgb_daily).to_csv(OUT/'d2_d3_daily_scores.csv',index=False)
    pd.DataFrame(lgb_hourly).to_csv(OUT/'d2_d3_hourly_predictions.csv',index=False)
    pd.DataFrame(extratrees_daily).to_csv(OUT/'d1_v11_extratrees_fold_daily.csv',index=False)
    pd.DataFrame(extratrees_hourly).to_csv(OUT/'d1_v11_extratrees_fold_hourly.csv',index=False)
    pd.DataFrame(log).to_csv(OUT/'lightgbm_experiment_log.csv',index=False)

    # All candidate rows become comparable via saved hourly vectors, exact same
    # dates, pooled hourly MAE/RMSE, and mean daily G2V regret.
    vectors=[]
    old_hgb=pd.read_csv(ROOT/'docs/smp_improvement_ml/hourly_predictions.csv')
    for row in old_hgb.itertuples(): vectors.append(dict(horizon_days=row.horizon_days,fold=row.fold,target_date=row.target_date,
        hour=row.hour,actual_smp=row.actual_smp,predicted_smp=row.predicted_smp,error=row.error,model=row.model))
    vectors.extend(lgb_hourly); vectors.extend(extratrees_hourly)
    allh=pd.DataFrame(vectors)
    paired=[]
    for (h,fold,name), group in allh.groupby(['horizon_days','fold','model'],sort=True):
        if len(group) % 24: continue
        day_metrics=[]
        for _,g in group.groupby('target_date',sort=True):
            if len(g)!=24: continue
            day_metrics.append(score(g.actual_smp.to_numpy(float),g.predicted_smp.to_numpy(float)))
        err=group.error.to_numpy(float)
        paired.append(dict(horizon_days=int(h),fold=fold,model=name,days=len(day_metrics),MAE=float(np.mean(np.abs(err))),
            RMSE=float(np.sqrt(np.mean(err**2))),G2V_regret=float(np.mean([x['G2V_regret'] for x in day_metrics])),
            peak_MAE=float(np.mean([x['peak_MAE'] for x in day_metrics if np.isfinite(x['peak_MAE'])])) if any(np.isfinite(x['peak_MAE']) for x in day_metrics) else np.nan,
            evaluation='same fold and target days; timestamp causal replay; receipt vintage unverified'))
    pairedf=pd.DataFrame(paired); pairedf.to_csv(OUT/'all_fold_paired_comparison.csv',index=False)
    pooled=[]
    for (h,name),group in allh.groupby(['horizon_days','model'],sort=True):
        day_stats=[]
        for _,g in group.groupby('target_date',sort=True):
            if len(g)==24: day_stats.append(score(g.actual_smp.to_numpy(float),g.predicted_smp.to_numpy(float)))
        if not day_stats: continue
        errors=group.error.to_numpy(float)
        if not np.isfinite(errors).all(): continue
        pooled.append(dict(horizon_days=int(h),model=name,days=len(day_stats),MAE=float(np.mean(np.abs(errors))),
            RMSE=float(np.sqrt(np.mean(errors**2))),G2V_regret=float(np.mean([x['G2V_regret'] for x in day_stats])),
            peak_MAE=float(np.mean([x['peak_MAE'] for x in day_stats if np.isfinite(x['peak_MAE'])])) if any(np.isfinite(x['peak_MAE']) for x in day_stats) else np.nan,
            evaluation='same hourly pooled metric and same target dates; retrospective timestamp-causal replay'))
    pooledf=pd.DataFrame(pooled); pooledf.to_csv(OUT/'all_fold_paired_summary.csv',index=False)

    manifest=dict(folds=FOLDS,models=dict(lightgbm_base=dict(n_estimators=300,learning_rate=.05,num_leaves=31,min_child_samples=20,random_state=42),
        lightgbm_peak=dict(base='lightgbm_base',sample_weight='6 when actual target SMP>=150, else 1'),
        d1_extratrees=dict(n_estimators=150,min_samples_leaf=10,max_features=.8,random_state=42)),
        datasets='same V11 SQLite land SMP history; features cut at nominal origin timestamp, first_seen not enforced for historical replay',
        evaluation='target-day cohorts identical to earlier 275-day 2024Q4/2025Q2/2025Q4 validation, EV=60kWh SOC30->80 7kW efficiency .9 full day',
        issue_time_policy='23:00 KST; v11/v17 builders drop a day when any required origin/lag SMP is missing; no future imputation; adapter raises on incomplete day',
        historical_first_seen_vintage_verified=False, independently_unseen_test=False,
        runtime_versions=dict(lightgbm=lgb.__version__,numpy=np.__version__,pandas=pd.__version__,scikit_learn=__import__('sklearn').__version__,joblib=joblib.__version__))
    (OUT/'study_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\nCOMPARISON\n',pooledf.to_string(index=False),flush=True)


if __name__ == '__main__': main()
