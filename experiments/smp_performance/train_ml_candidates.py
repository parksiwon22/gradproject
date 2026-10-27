"""Train horizon-specific causal HGB research candidates and replay old models.

All generated artifacts live under docs/smp_improvement_ml; original V5-V20
models and result files are read-only. Historical first-seen vintages are not
available, so scores are timestamp-causal retrospective research only.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from v11.features import FEATURES as D1_FEATURES, build as build_d1
from v11.store import ROOT, connect, latest
from v12.ev import EV
from v17.features import FEATURES as DX_FEATURES, build as build_dx
from experiments.smp_performance.run_study import select_anchor_weight


OUT = ROOT / 'docs' / 'smp_improvement_ml'
FINAL_LABEL_CUTOFF = '2026-10-26'  # task date is 2026-10-27 KST
FOLDS = [
    dict(name='fall2024', train_end='2024-06-30', cal_start='2024-07-01', cal_end='2024-09-30', val_start='2024-10-01', val_end='2024-12-31'),
    dict(name='spring2025', train_end='2024-12-31', cal_start='2025-01-01', cal_end='2025-03-31', val_start='2025-04-01', val_end='2025-06-30'),
    dict(name='fall2025', train_end='2025-06-30', cal_start='2025-07-01', cal_end='2025-09-30', val_start='2025-10-01', val_end='2025-12-31'),
]
EV_RULE = EV(degradation=5)
FEATURES_BY_H = {1: D1_FEATURES, 2: DX_FEATURES, 3: DX_FEATURES}


def load_prices():
    db = connect()
    try:
        p = latest(db, 'land')
    finally:
        db.close()
    return p.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)


def features_for(prices, horizon):
    if horizon == 1:
        return build_d1(prices, days_ahead=1)
    return build_dx(prices, days_ahead=horizon)


def allocate(predicted):
    required = EV_RULE.capacity * (EV_RULE.target_soc - EV_RULE.initial_soc) / 100 / EV_RULE.charge_eff
    charge = np.zeros(24, dtype=float)
    for h in np.argsort(predicted, kind='stable'):
        amount = min(EV_RULE.charge_kw, required)
        charge[h] = amount
        required -= amount
        if required <= 1e-9:
            break
    return charge


def score(actual, predicted):
    err = np.asarray(predicted) - np.asarray(actual)
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    regret = float(np.dot(actual, allocate(predicted) - allocate(actual)))
    peak = actual >= 150
    return dict(MAE=float(np.mean(np.abs(err))), RMSE=float(np.sqrt(np.mean(err ** 2))),
                G2V_regret=regret, peak_MAE=float(np.mean(np.abs(err[peak]))) if peak.any() else np.nan,
                peak_hours=int(peak.sum()))


def day_vectors(frame, horizon):
    rows = {}
    for day, g in frame.groupby('target_date', sort=True):
        g = g.sort_values('timestamp')
        if len(g) != 24:
            continue
        pred_origin = pd.Timestamp(g.origin_timestamp.iloc[0])
        expected_origin = pd.Timestamp(day, tz='Asia/Seoul') - pd.Timedelta(days=horizon) + pd.Timedelta(hours=23)
        if pred_origin != expected_origin:
            raise AssertionError(f'invalid issue origin: D+{horizon} {day}: {pred_origin} != {expected_origin}')
        max_known_lag = pd.DatetimeIndex(g.timestamp) - pd.Timedelta(days=horizon)
        if max_known_lag.max() > pred_origin:
            raise AssertionError(f'baseline feature past origin: D+{horizon} {day}')
        if horizon == 1:
            actual, baseline = g.actual_smp.to_numpy(float), g.daily.to_numpy(float)
        else:
            actual, baseline = g.actual_smp.to_numpy(float), g.daily.to_numpy(float)
        rows[str(day)] = dict(day=str(day), actual=actual, baseline=baseline,
                              X=g[FEATURES_BY_H[horizon]].to_numpy(float),
                              timestamps=pd.DatetimeIndex(g.timestamp), origin=pred_origin,
                              hours=g.timestamp.dt.hour.to_numpy())
    return rows


def fit_model(x, y, dates, mode, actual_prices):
    # Fixed model capacity from existing V11/V17 HGB family; variants test only
    # causal recency and a bounded high-price emphasis.
    model = HistGradientBoostingRegressor(max_iter=250, max_leaf_nodes=31, learning_rate=.05,
                                          early_stopping=False, l2_regularization=2.0, random_state=42)
    y = np.asarray(y)
    actual_prices = np.asarray(actual_prices)
    kwargs = {}
    if mode in ('recency365', 'recency365_peak'):
        parsed_dates = pd.to_datetime(dates)
        age = (parsed_dates.max() - parsed_dates).days.to_numpy()
        weights = np.exp(-np.log(2) * age / 365.0)
        if mode == 'recency365_peak':
            weights *= np.where(actual_prices >= 150, 2.0, 1.0)
        kwargs['sample_weight'] = weights
    elif mode == 'peak_weighted':
        kwargs['sample_weight'] = np.where(actual_prices >= 150, 2.0, 1.0)
    model.fit(x, y, **kwargs)
    return model


def select_blend(calibration, ml_predictions):
    # Choose HGB/anchor weight only on the preceding calendar quarter.
    weights = np.linspace(0, 1, 21)
    rows = []
    for weight in weights:
        stats = [score(item['actual'], (1 - weight) * ml + weight * item['anchor'])
                 for item, ml in zip(calibration, ml_predictions)]
        rows.append(dict(weight=float(weight), mae=float(np.mean([x['MAE'] for x in stats])),
                         rmse=float(np.sqrt(np.mean([x['RMSE'] ** 2 for x in stats]))),
                         regret=float(np.mean([x['G2V_regret'] for x in stats])),
                         base_mae=float(np.mean([x['base']['MAE'] for x in calibration])),
                         base_regret=float(np.mean([x['base']['G2V_regret'] for x in calibration]))))
    eligible = [r for r in rows if r['mae'] <= r['base_mae'] and r['regret'] <= r['base_regret']]
    chosen = min(eligible, key=lambda r: (r['mae'], r['regret'], r['rmse'])) if eligible else dict(weight=0.0)
    return float(chosen['weight']), rows


def load_prior_candidates():
    """Load compatible saved candidates without altering them; report failures."""
    candidates = {
        'D+1_V11_HGB_saved': ROOT / 'artifacts/v11/training/through_2026Q1__hgb.joblib',
        'D+1_V11_ExtraTrees_saved': ROOT / 'artifacts/v11/training/through_2026Q1__extra_trees.joblib',
        'D+2_V12_LightGBM_peak_saved': ROOT / 'artifacts/v12/study/lightgbm_peak.joblib',
        'D+2_V14_shape_level_blend_saved': ROOT / 'artifacts/v14/study/shape_level_blend.joblib',
        'D+2_V15_event_features_lgb_saved': ROOT / 'artifacts/v15/model_study/winter2025_event_features_lgb.joblib',
        'D+2/D+3_V17_joint_saved': ROOT / 'artifacts/v17/study/winter2025.joblib',
        'D+3_V19_HGB_saved': ROOT / 'artifacts/v19/study/d3_hgb.joblib',
        'D+1_V5_HGB_saved': ROOT / 'artifacts/models/day_ahead_hgb_residual_model.joblib',
    }
    status = []
    loaded = {}
    for name, path in candidates.items():
        if not path.exists():
            status.append(dict(model=name, path=str(path.relative_to(ROOT)), load_status='missing'))
            continue
        try:
            obj = joblib.load(path)
            loaded[name] = obj
            meta = ({k: v for k, v in obj.items() if isinstance(v, (str, int, float, bool))}
                    if isinstance(obj, dict) else {'object_type': type(obj).__name__})
            status.append(dict(model=name, path=str(path.relative_to(ROOT)), load_status='loaded', metadata=json.dumps(meta, default=str)))
        except Exception as exc:
            status.append(dict(model=name, path=str(path.relative_to(ROOT)), load_status='unavailable', reason=f'{type(exc).__name__}: {exc}'))
    pd.DataFrame(status).to_csv(OUT / 'legacy_model_load_audit.csv', index=False)
    return loaded


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    daymaps = {h: day_vectors(features_for(prices, h), h) for h in (1, 2, 3)}
    trace, daily, experiments = [], [], []
    for h in (1, 2, 3):
        days = daymaps[h]
        feature_names = FEATURES_BY_H[h]
        for fold in FOLDS:
            train_days = [d for d in sorted(days) if d <= fold['train_end']]
            cal_days = [d for d in sorted(days) if fold['cal_start'] <= d <= fold['cal_end']]
            val_days = [d for d in sorted(days) if fold['val_start'] <= d <= fold['val_end']]
            if len(train_days) < 300 or len(cal_days) < 80 or len(val_days) < 80:
                experiments.append(dict(horizon_days=h, fold=fold['name'], status='skipped', train_days=len(train_days), calibration_days=len(cal_days), validation_days=len(val_days)))
                continue
            X = np.concatenate([days[d]['X'] for d in train_days])
            y = np.concatenate([days[d]['actual'] - days[d]['baseline'] for d in train_days])
            actual_prices = np.concatenate([days[d]['actual'] for d in train_days])
            sample_dates = np.repeat(train_days, 24)
            cal = [days[d] for d in cal_days]
            val = [days[d] for d in val_days]
            # Seasonal anchor identical to prior rolling-anchor method. The blend
            # candidate is calibrated against this fold's prior quarter only.
            s = prices.set_index('timestamp').smp.sort_index()
            for item in cal + val:
                t = item['timestamps']
                item['anchor'] = np.median(np.vstack([s.reindex(t - pd.Timedelta(days=7*k)).to_numpy(float) for k in range(1, 5)]), axis=0)
                if np.any(~np.isfinite(item['anchor'])):
                    raise RuntimeError(f'missing anchor for {item["day"]}')
            for item in cal:
                item['base'] = score(item['actual'], item['baseline'])
            modes = ('hgb', 'recency365', 'peak_weighted', 'recency365_peak')
            model_cal_preds = {}
            model_val_preds = {}
            for mode in modes:
                model = fit_model(X, y, sample_dates, mode, actual_prices)
                cal_preds = [item['baseline'] + model.predict(item['X']) for item in cal]
                val_preds = [item['baseline'] + model.predict(item['X']) for item in val]
                model_cal_preds[mode] = cal_preds
                model_val_preds[mode] = val_preds
                bundle = dict(model=model, horizon_days=h, mode=mode, features=feature_names,
                              trained_through=fold['train_end'], train_start=train_days[0],
                              train_days=len(train_days), feature_origin_policy='23:00 KST causal price-only features',
                              original_vintage_verified=False)
                joblib.dump(bundle, OUT / f'd{h}_{fold["name"]}_{mode}.joblib')
            # Choose base ML variant based on calibration MAE with explicit EV
            # non-regression constraint; do not look at validation to select.
            base_candidates = []
            for mode, preds in model_cal_preds.items():
                stats = [score(it['actual'], p) for it, p in zip(cal, preds)]
                base_candidates.append((float(np.mean([x['MAE'] for x in stats])),
                                        float(np.mean([x['G2V_regret'] for x in stats])),
                                        float(np.sqrt(np.mean([x['RMSE']**2 for x in stats]))), mode))
            base_mae = float(np.mean([x['base']['MAE'] for x in cal]))
            base_regret = float(np.mean([x['base']['G2V_regret'] for x in cal]))
            eligible = [r for r in base_candidates if r[0] <= base_mae and r[1] <= base_regret]
            selected_mode = min(eligible)[-1] if eligible else 'hgb'
            blend_weight, blend_curve = select_blend(cal, model_cal_preds[selected_mode])
            seasonal_weight, _ = select_blend(cal, [item['baseline'] for item in cal])
            pd.DataFrame(blend_curve).to_csv(OUT / f'd{h}_{fold["name"]}_blend_calibration.csv', index=False)
            for model_name in ['persistence', 'weekday_median4', 'rolling_anchor_shadow'] + list(modes) + ['ml_anchor_blend']:
                vals = []
                for item, preds_by_mode in zip(val, [dict((m, model_val_preds[m][i]) for m in modes) for i in range(len(val))]):
                    if model_name == 'persistence': p = item['baseline']
                    elif model_name == 'weekday_median4': p = item['anchor']
                    elif model_name == 'rolling_anchor_shadow': p = (1-seasonal_weight)*item['baseline']+seasonal_weight*item['anchor']
                    elif model_name == 'ml_anchor_blend': p = (1-blend_weight)*preds_by_mode[selected_mode]+blend_weight*item['anchor']
                    else: p = preds_by_mode[model_name]
                    stat = score(item['actual'], p)
                    vals.append(stat)
                    daily.append(dict(horizon_days=h, fold=fold['name'], target_date=item['day'], model=model_name, **stat,
                                      selected_ml_mode=selected_mode, selected_anchor_weight=blend_weight,
                                      evaluation_type='timestamp-causal retrospective walk-forward; first_seen vintage unverified'))
                    for hour, (actual, pred) in enumerate(zip(item['actual'], p)):
                        trace.append(dict(horizon_days=h, fold=fold['name'], target_date=item['day'], hour=hour,
                                          actual_smp=float(actual), predicted_smp=float(pred), error=float(pred-actual), model=model_name))
                errors = np.concatenate([((item['baseline'] if name=='persistence' else item['anchor'] if name=='weekday_median4' else None)) for name in []]) if False else None
                experiments.append(dict(horizon_days=h, fold=fold['name'], status='success', model=model_name,
                                        train_days=len(train_days), calibration_days=len(cal), validation_days=len(val),
                                        selected_ml_mode=selected_mode, selected_anchor_weight=blend_weight,
                                        calibration_MAE_best=min(base_candidates)[0], calibration_regret_best=min(base_candidates)[1],
                                        validation_MAE=float(np.mean([x['MAE'] for x in vals])),
                                        validation_RMSE=float(np.sqrt(np.mean([x['RMSE']**2 for x in vals]))),
                                        validation_G2V_regret=float(np.mean([x['G2V_regret'] for x in vals])),
                                        feature_count=len(feature_names), cutoff_verified=True, vintage_verified=False))
            print(f'D+{h} {fold["name"]}: trained {len(train_days)} days; selected {selected_mode}, anchor weight {blend_weight:.2f}', flush=True)
    pd.DataFrame(daily).to_csv(OUT/'daily_scores.csv', index=False)
    pd.DataFrame(trace).to_csv(OUT/'hourly_predictions.csv', index=False)
    pd.DataFrame(experiments).to_csv(OUT/'experiment_log.csv', index=False)
    score_table = pd.DataFrame(daily)
    summary = score_table.groupby(['horizon_days','model'], as_index=False).agg(
        folds=('fold','nunique'), days=('target_date','nunique'), MAE=('MAE','mean'), RMSE=('RMSE','mean'),
        G2V_regret=('G2V_regret','mean'), peak_MAE=('peak_MAE','mean'))
    hourly_frame = pd.DataFrame(trace)
    pooled_rmse = hourly_frame.groupby(['horizon_days','model']).error.apply(lambda x: float(np.sqrt(np.mean(np.square(x))))).to_dict()
    summary['RMSE'] = [pooled_rmse[(h, model)] for h, model in zip(summary.horizon_days, summary.model)]
    baseline = summary[summary.model.eq('persistence')].set_index('horizon_days')
    for col in ('MAE','RMSE','G2V_regret'):
        summary[col+'_improvement_pct'] = [100*(1-v/baseline.loc[h,col]) for h,v in zip(summary.horizon_days,summary[col])]
    summary['evaluation'] = 'prior-reviewed folds; retrospective, timestamp-causal, vintage unverified'
    summary.to_csv(OUT/'summary.csv', index=False)
    loaded = load_prior_candidates()

    # Direct same-day checks from preserved forecast files. These windows were
    # previously used for model selection, so they remain selection diagnostics.
    legacy_rows = []
    def append_saved_frame(path, horizon, columns, period_start, period_end, source_label):
        f = pd.read_csv(path)
        f = f[(f.target_date >= period_start) & (f.target_date <= period_end)]
        for label, column in columns.items():
            if column not in f:
                continue
            byday = []
            for day, g in f.groupby('target_date', sort=True):
                g = g.sort_values('timestamp')
                if len(g) != 24:
                    continue
                a = g.actual_smp.to_numpy(float)
                p = g[column].to_numpy(float)
                byday.append(score(a, p))
            if byday:
                legacy_rows.append(dict(horizon_days=horizon, model=label, source=source_label,
                    start=period_start, end=period_end, days=len(byday),
                    MAE=float(np.mean([x['MAE'] for x in byday])),
                    RMSE=float(np.sqrt(np.mean([x['RMSE']**2 for x in byday]))),
                    G2V_regret=float(np.mean([x['G2V_regret'] for x in byday])),
                    assessment='previously reviewed/selection window; diagnostic, not independent test'))
    append_saved_frame(ROOT/'artifacts/v12/study/selection_predictions.csv', 2,
        {'V12 LightGBM peak (saved hourly output)':'lightgbm_peak'}, '2025-10-01','2025-12-31','V12 saved selection predictions')
    append_saved_frame(ROOT/'artifacts/v14/study/selection_predictions.csv', 2,
        {'V14 shape-level blend (saved hourly output)':'shape_level_blend'}, '2025-10-01','2025-12-31','V14 saved selection predictions')
    append_saved_frame(ROOT/'artifacts/v15/model_study/winter2025_predictions.csv', 2,
        {'V15 event-features LGB (saved hourly output)':'event_features_lgb'}, '2025-10-01','2025-12-31','V15 saved winter2025 predictions')
    # V17 exposes daily common-day EV and MAE metrics but not stored hourly
    # price predictions in this environment, so RMSE is intentionally blank.
    v17file = ROOT/'artifacts/v17/study/winter2025_validation_daily.csv'
    if v17file.exists():
        old = pd.read_csv(v17file)
        old = old[(old.day >= '2025-10-01') & (old.day <= '2025-12-31') &
                  old.pattern.eq('full24') & old.model.eq('joint_price')]
        if len(old):
            legacy_rows.append(dict(horizon_days=2,model='V17 joint_price (saved daily)',source='V17 saved daily full24 metrics',
                start='2025-10-01',end='2025-12-31',days=int(old.day.nunique()),
                MAE=float(old.price_MAE.mean()),RMSE=np.nan,G2V_regret=float(old.G2V_regret.mean()),
                assessment='previously reviewed fold; same full24 EV window; saved hourly predictions/RMSE unavailable'))
    if legacy_rows:
        legacy = pd.DataFrame(legacy_rows)
        newcommon = summary[(summary.horizon_days.eq(2)) & summary.model.isin(['persistence','hgb','peak_weighted','ml_anchor_blend','rolling_anchor_shadow'])]
        newcommon = newcommon.copy(); newcommon['source'] = 'new horizon-specific HGB validation; folds include 2025Q4'
        newcommon['start'] = '2025-10-01'; newcommon['end'] = '2025-12-31'; newcommon['assessment'] = 'previously reviewed fold; compare only within shared dates'
        # Replace overall three-fold aggregates with exact last-fold rows for fair paired comparison.
        lastfold = score_table[(score_table.horizon_days.eq(2)) & score_table.fold.eq('fall2025') & score_table.model.isin(['persistence','hgb','peak_weighted','ml_anchor_blend','rolling_anchor_shadow'])].groupby('model').agg(days=('target_date','nunique'), MAE=('MAE','mean'), RMSE=('RMSE','mean'), G2V_regret=('G2V_regret','mean')).reset_index()
        lastfold = lastfold.merge(newcommon[['model','source','start','end','assessment']],on='model',how='left')
        pd.concat([legacy,lastfold],ignore_index=True).to_csv(OUT/'d2_2025q4_common_date_comparison.csv',index=False)

    # Existing V19 daily report provides V17/V19 same-day D3 scores. It shares
    # the already-reviewed 2025Q4 window; do not use it for selection.
    v19file = ROOT/'artifacts/v19/study/daily.csv'
    v19rows = []
    if v19file.exists():
        old = pd.read_csv(v19file)
        old = old[(old.day >= '2025-10-01') & (old.day <= '2025-12-31')]
        for name, grp in old[old.model.isin(['V19_D3_HGB','V17_joint_D3'])].groupby('model'):
            if len(grp):
                v19rows.append(dict(horizon_days=3,model=name,days=len(grp),MAE=float(grp.MAE.mean()),
                    RMSE=float(grp.RMSE.mean()),G2V_regret=float(grp.G2V_regret.mean()),
                    source='V19 saved daily metrics', assessment='previously reviewed/selection window; diagnostic'))
    if v19rows:
        lastfold3 = score_table[(score_table.horizon_days.eq(3)) & score_table.fold.eq('fall2025') & score_table.model.isin(['persistence','hgb','peak_weighted','ml_anchor_blend','rolling_anchor_shadow'])].groupby('model').agg(days=('target_date','nunique'),MAE=('MAE','mean'),RMSE=('RMSE','mean'),G2V_regret=('G2V_regret','mean')).reset_index()
        lastfold3['horizon_days']=3; lastfold3['source']='new horizon-specific HGB validation; folds include 2025Q4'; lastfold3['assessment']='previously reviewed fold; paired dates, vintage unverified'
        pd.concat([pd.DataFrame(v19rows),lastfold3],ignore_index=True).to_csv(OUT/'d3_v17_v19_common_date_comparison.csv',index=False)

    # Same-date V11 diagnostic on 2026Q3, using the locked training endpoint
    # before the quarter. This period was previously examined; it is descriptive
    # only and does not affect model or blend selection.
    v11_test_path=ROOT/'artifacts/v11/training/test_predictions.csv'
    if v11_test_path.exists():
        old=pd.read_csv(v11_test_path)
        old=old[(old.target_date>='2026-07-01')&(old.target_date<='2026-09-30')]
        cols={'V11 fixed profile saved':'fixed__profile','V11 HGB saved through Q1':'through_2026Q1__hgb',
              'V11 ExtraTrees saved through Q1':'through_2026Q1__extra_trees'}
        diag=[]
        for name,col in cols.items():
            vals=[]
            if col not in old: continue
            for _,g in old.groupby('target_date',sort=True):
                g=g.sort_values('timestamp')
                if len(g)==24: vals.append(score(g.actual_smp.to_numpy(float),g[col].to_numpy(float)))
            if vals: diag.append(dict(horizon_days=1,model=name,days=len(vals),MAE=np.mean([x['MAE'] for x in vals]),
                RMSE=np.mean([x['RMSE'] for x in vals]),G2V_regret=np.mean([x['G2V_regret'] for x in vals]),
                source='V11 saved 2026Q3 predictions',assessment='previously reviewed audit; diagnostic only'))
        d1map=daymaps[1]; tr=[d for d in sorted(d1map) if d<='2026-03-31']; va=sorted([d for d in d1map if '2026-07-01'<=d<='2026-09-30'])
        if tr and va:
            xx=np.concatenate([d1map[d]['X'] for d in tr]); yy=np.concatenate([d1map[d]['actual']-d1map[d]['baseline'] for d in tr]); aa=np.concatenate([d1map[d]['actual'] for d in tr]); dd=np.repeat(tr,24)
            m=fit_model(xx,yy,dd,'peak_weighted',aa); vals=[]
            for d in va: vals.append(score(d1map[d]['actual'],d1map[d]['baseline']+m.predict(d1map[d]['X'])))
            diag.append(dict(horizon_days=1,model='New peak-weighted HGB refit through Q1',days=len(vals),
                MAE=np.mean([x['MAE'] for x in vals]),RMSE=np.mean([x['RMSE'] for x in vals]),G2V_regret=np.mean([x['G2V_regret'] for x in vals]),
                source='new fixed refit evaluated on same previously reviewed 2026Q3 audit',assessment='diagnostic only; not used for reselection'))
        pd.DataFrame(diag).to_csv(OUT/'d1_v11_2026q3_common_date_diagnostic.csv',index=False)

    # Freeze a non-operating shadow candidate per horizon. Pick the fixed ML
    # variant from the time-ordered validation summary (MAE first, regret no
    # worse than persistence); calibrate only the blend weight on the latest
    # 90 completed targets, then refit on all available historical labels.
    final_modes = {1:'peak_weighted', 2:'peak_weighted', 3:'hgb'}
    prices_series = prices.set_index('timestamp').smp.sort_index()
    for h in (1,2,3):
        daymap = daymaps[h]
        # The archive contains future target dates; exclude current and future
        # labels from every candidate intended for a later forecast origin.
        final_dates = [d for d in sorted(daymap) if d <= FINAL_LABEL_CUTOFF]
        available_end = final_dates[-1]
        cal_dates = final_dates[-90:]
        train_dates = [d for d in final_dates if d < cal_dates[0]]
        cal_items = [daymap[d] for d in cal_dates]
        for item in cal_items:
            t=item['timestamps']
            item['anchor']=np.median(np.vstack([prices_series.reindex(t-pd.Timedelta(days=7*k)).to_numpy(float) for k in range(1,5)]),axis=0)
            item['base']=score(item['actual'],item['baseline'])
        pre_X=np.concatenate([daymap[d]['X'] for d in train_dates]); pre_y=np.concatenate([daymap[d]['actual']-daymap[d]['baseline'] for d in train_dates])
        pre_prices=np.concatenate([daymap[d]['actual'] for d in train_dates]); pre_dates=np.repeat(train_dates,24)
        mode=final_modes[h]
        pre_model=fit_model(pre_X,pre_y,pre_dates,mode,pre_prices)
        pre_cal=[item['baseline']+pre_model.predict(item['X']) for item in cal_items]
        alpha,curve=select_blend(cal_items,pre_cal)
        curve_path=OUT/f'd{h}_final_blend_calibration.csv'; pd.DataFrame(curve).to_csv(curve_path,index=False)
        all_X=np.concatenate([daymap[d]['X'] for d in final_dates]); all_y=np.concatenate([daymap[d]['actual']-daymap[d]['baseline'] for d in final_dates])
        all_prices=np.concatenate([daymap[d]['actual'] for d in final_dates]); all_dates=np.repeat(final_dates,24)
        final_model=fit_model(all_X,all_y,all_dates,mode,all_prices)
        bundle=dict(schema=1,model=final_model,horizon_days=h,mode=mode,features=FEATURES_BY_H[h],
                    anchor='median of same-hour SMP at target-7/14/21/28 days',anchor_weight=alpha,
                    trained_through=available_end,train_start=final_dates[0],train_days=len(final_dates),
                    calibration_start=cal_dates[0],calibration_end=cal_dates[-1],calibration_days=len(cal_dates),
                    training_policy='direct HGB residual; fixed variant chosen from repeated chronological validation; recent 90-day blend calibration; refit all available history',
                    original_vintage_verified=False,operating_replacement=False)
        joblib.dump(bundle, OUT/f'd{h}_recommended_shadow.joblib')
    # Plot fold-level candidate metric comparisons, leaving each horizon on its own axis.
    fig, axes = plt.subplots(1,3,figsize=(17,5))
    chosen=['persistence','weekday_median4','hgb','recency365','peak_weighted','recency365_peak','ml_anchor_blend']
    for h, ax in zip((1,2,3),axes):
        part=summary[(summary.horizon_days==h)&summary.model.isin(chosen)].set_index('model').reindex(chosen).dropna()
        part[['MAE','RMSE','G2V_regret']].plot(kind='bar',ax=ax)
        ax.set_title(f'D+{h}'); ax.set_ylabel('MAE / RMSE / G2V regret'); ax.tick_params(axis='x',rotation=60); ax.grid(axis='y',alpha=.2)
    fig.suptitle('Horizon-specific HGB experiments (retrospective, vintage unverified)'); fig.tight_layout()
    fig.savefig(OUT/'candidate_metric_comparison.png',dpi=170); plt.close(fig)
    # Pick representative actual vs forecasts from the last fold for presentation only.
    for h in (1,2,3):
        sub=pd.DataFrame(trace); sub=sub[(sub.horizon_days==h)&(sub.fold=='fall2025')]
        dates=sorted(sub.target_date.unique())
        day=dates[len(dates)//2]
        p=sub[sub.target_date.eq(day)]
        fig,ax=plt.subplots(figsize=(11,4)); actual=p[p.model.eq('persistence')]
        ax.plot(actual.hour,actual.actual_smp,label='Actual SMP',lw=2,color='#111827')
        for m in ['persistence','hgb','ml_anchor_blend','rolling_anchor_shadow']:
            q=p[p.model.eq(m)]
            if not q.empty: ax.plot(q.hour,q.predicted_smp,label=m)
        ax.set(title=f'D+{h} representative retrospective forecast · {day}',xlabel='Hour (KST)',ylabel='SMP (KRW/kWh)'); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
        fig.savefig(OUT/f'd{h}_representative_forecast.png',dpi=170); plt.close(fig)
    manifest=dict(created_at=pd.Timestamp.now(tz='Asia/Seoul').isoformat(), folds=FOLDS,
                  model='HGB residual direct models per horizon, 4 fixed variants; calibration-only selection/blending',
                  model_parameters=dict(max_iter=250,max_leaf_nodes=31,learning_rate=.05,l2_regularization=2.0,early_stopping=False,random_state=42),
                  runtime_versions=dict(scikit_learn=__import__('sklearn').__version__,numpy=np.__version__,pandas=pd.__version__),
                  features={str(k):v for k,v in FEATURES_BY_H.items()},
                  ev_conditions=dict(capacity_kwh=60,initial_soc_percent=30,target_soc_percent=80,charge_kw=7,efficiency=.9,degradation=5,window='full target day'),
                  train_calibration_validation_temporal_order=True, nominal_origin='23:00 Asia/Seoul',
                  input_columns='SMP-derived price features only', original_first_seen_vintage_verified=False,
                  final_label_cutoff=FINAL_LABEL_CUTOFF,
                  all_validation_periods_previously_seen=True, operating_models_modified=False,
                  original_model_artifacts_preserved=True, loaded_existing_candidate_count=len(loaded))
    (OUT/'study_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\nSUMMARY\n', summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
