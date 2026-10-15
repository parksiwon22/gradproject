"""New local experiments; private outputs stay under ignored artifacts/."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import json
import hashlib
import importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor as HGB, HistGradientBoostingClassifier, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score
from model import v5

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/research_local'
SEED = 42

def metrics(frame, pred, threshold):
    y = frame.actual_smp.to_numpy()
    pred = np.asarray(pred)
    peak = y >= threshold
    alert = pred >= threshold
    result = dict(MAE=mean_absolute_error(y, pred), RMSE=float(np.sqrt(mean_squared_error(y, pred))), R2=r2_score(y, pred), peak_threshold=threshold, peak_count=int(peak.sum()), peak_MAE=float(np.mean(abs(pred[peak]-y[peak]))) if peak.any() else None, peak_Bias=float(np.mean(pred[peak]-y[peak])) if peak.any() else None, peak_Precision=precision_score(peak, alert, zero_division=0), peak_Recall=recall_score(peak, alert, zero_division=0), peak_F1=f1_score(peak, alert, zero_division=0), false_alarms=int((alert & ~peak).sum()))
    costs, optimal, overlaps, errors, maxerr = [], [], [], [], []
    ramp_up, ramp_down = [], []
    for _, indexes in frame.groupby('target_date', sort=True).indices.items():
        a, p = y[indexes], pred[indexes]
        assert len(a) == 24
        maxerr.append(abs(a.max()-p.max()))
        delta = np.diff(np.r_[frame.iloc[indexes[0]].previous_actual_smp, a])
        delta_p = np.diff(np.r_[frame.iloc[indexes[0]].previous_actual_smp, p])
        ramp_up.extend(abs(delta_p[delta>=15]-delta[delta>=15]))
        ramp_down.extend(abs(delta_p[delta<=-15]-delta[delta<=-15]))
        for duration in (2,3,4,5,6,8):
            s, _ = v5.cheapest_window(p, duration)
            oracle, value = v5.cheapest_window(a, duration)
            costs.append(float(a[s:s+duration].sum()))
            optimal.append(value)
            overlaps.append(v5.overlap_rate(s, oracle, duration))
            errors.append(int(np.any((a[s:s+duration]>=threshold)&(p[s:s+duration]<threshold))))
    result.update(EV_cost=float(np.mean(costs)), EV_oracle_cost=float(np.mean(optimal)), EV_regret=float(np.mean(np.array(costs)-optimal)), cheapest_window_overlap=float(np.mean(overlaps)), peak_charging_error_rate=float(np.mean(errors)), daily_max_MAE=float(np.mean(maxerr)), up_ramp_MAE=float(np.mean(ramp_up)) if ramp_up else None, down_ramp_MAE=float(np.mean(ramp_down)) if ramp_down else None)
    return result

def run():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(v5.DATA_PATH)
    prepared = v5.preprocess_hourly(raw)
    data = v5.build_stacked_dataset(prepared)
    features = prepared.model_features
    # Fit through June; July selects residual calibration, August/September select models.
    train = data[data.target_date < '2023-07-01'].reset_index(drop=True)
    cal = data[(data.target_date >= '2023-07-01') & (data.target_date < '2023-08-01')].reset_index(drop=True)
    val = data[(data.target_date >= '2023-08-01') & (data.target_date < '2023-10-01')].reset_index(drop=True)
    test = data[data.target_date >= '2023-10-01'].reset_index(drop=True)
    assert (pd.to_datetime(data.origin_timestamp) < pd.to_datetime(data.timestamp)).all()
    assert pd.to_datetime(data.origin_timestamp).dt.hour.eq(23).all()
    threshold = float(train.actual_smp.quantile(.95))
    manifest = dict(run='local-20261015-new', seed=SEED, data_sha256=hashlib.sha256(v5.DATA_PATH.read_bytes()).hexdigest(), features=features, source='local aligned_hourly_all_features_2023.csv; original feature provenance not independently re-audited', cutoff='previous day 23:00; source publication lags unverified', periods={'train':[train.target_date.min(),train.target_date.max()], 'calibration':[cal.target_date.min(),cal.target_date.max()], 'validation':[val.target_date.min(),val.target_date.max()], 'historical_test':[test.target_date.min(),test.target_date.max()]}, packages={x:importlib.metadata.version(x) for x in ['numpy','pandas','scikit-learn','joblib']}, experiments=[])
    frames = {'calibration':cal,'validation':val,'historical_test':test}
    predictions = {s:{} for s in frames}
    for s,f in frames.items():
        predictions[s]['daily_profile'] = f.same_hour_previous_day_smp.to_numpy()
        predictions[s]['fixed_profile_069'] = .69*f.same_hour_previous_day_smp.to_numpy()+.31*f.same_hour_previous_week_smp.to_numpy()
    candidates = [('hgb',HGB(max_iter=400,learning_rate=.05,max_leaf_nodes=31,l2_regularization=.05,random_state=SEED),False), ('hgb_peak_weight4',HGB(max_iter=400,learning_rate=.05,max_leaf_nodes=31,l2_regularization=.05,random_state=SEED),True), ('hgb_quantile90',HGB(loss='quantile',quantile=.9,max_iter=300,learning_rate=.05,random_state=SEED),False), ('extra_trees',ExtraTreesRegressor(n_estimators=200,min_samples_leaf=10,max_features=.8,n_jobs=4,random_state=SEED),False)]
    for module, cls in [('lightgbm','LGBMRegressor'),('catboost','CatBoostRegressor'),('xgboost','XGBRegressor')]:
        try:
            mod=__import__(module)
            kwargs = dict(random_state=SEED)
            if module=='catboost': kwargs.update(iterations=400,verbose=False,thread_count=4,allow_writing_files=False)
            else: kwargs.update(n_estimators=400,n_jobs=4,learning_rate=.05)
            candidates.append((module,getattr(mod,cls)(**kwargs),False))
        except ImportError:
            manifest['experiments'].append(dict(model=module,status='unavailable',reason='not installed; package download DNS failed'))
    for name, estimator, weighted in candidates:
        try:
            if isinstance(estimator, HGB): estimator.set_params(early_stopping=False)
            kwargs = {'sample_weight':np.where(train.actual_smp>=threshold,4.,1.)} if weighted else {}
            estimator.fit(train[features].to_numpy(),train.target_residual_vs_daily.to_numpy(),**kwargs)
            for s,f in frames.items(): predictions[s][name] = f.same_hour_previous_day_smp.to_numpy()+estimator.predict(f[features].to_numpy())
            joblib.dump(estimator,OUT/(name+'.joblib'))
            manifest['experiments'].append(dict(model=name,status='success',parameters=estimator.get_params(),peak_weight=4 if weighted else 1))
        except Exception as exc:
            manifest['experiments'].append(dict(model=name,status='failed',reason=repr(exc)))
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
        print(name,manifest['experiments'][-1]['status'],flush=True)
    # Calibration uses July only, never model selection or test targets.
    bias = cal.assign(error=cal.actual_smp-predictions['calibration']['hgb']).groupby('target_hour').error.mean()
    classifier=HistGradientBoostingClassifier(max_iter=200,random_state=SEED,early_stopping=False)
    classifier.fit(train[features],(train.actual_smp>=threshold).astype(int))
    correction=float(np.mean(cal.actual_smp-predictions['calibration']['hgb']))
    for s,f in frames.items():
        predictions[s]['hour_residual_correction']=predictions[s]['hgb']+f.target_hour.map(bias).to_numpy()
        predictions[s]['ensemble']=.5*predictions[s]['hgb']+.5*predictions[s]['hgb_peak_weight4']
        probability=classifier.predict_proba(f[features])[:,1]
        predictions[s]['peak_gate']=predictions[s]['hgb']*(1-probability)+predictions[s]['hgb_quantile90']*probability
        pd.DataFrame({'timestamp':f.timestamp,'peak_probability':probability,'q90_price':predictions[s]['hgb_quantile90']}).to_csv(OUT/(s+'_risk.csv'),index=False)
    manifest['calibration']={'hour_bias':bias.to_dict(),'global_bias_diagnostic':correction,'classifier_parameters':classifier.get_params()}
    rows=[]
    for s,f in frames.items():
        for name,p in predictions[s].items(): rows.append(dict(split=s,model=name,**metrics(f,p,threshold)))
    table=pd.DataFrame(rows)
    validation=table[table.split=='validation']
    baseline=validation[validation.model=='fixed_profile_069'].iloc[0]
    eligible=validation[validation.MAE<=baseline.MAE*1.05]
    winner=eligible.sort_values(['peak_MAE','MAE','model']).iloc[0].model
    manifest['selection']={'rule':'validation MAE <= fixed profile MAE * 1.05; minimize peak MAE then MAE','winner':winner,'test_not_used_for_selection':True,'v5_replacement':'deferred: 2024 and publication-lag audit missing'}
    # Frozen V5 is contextual, trained through August and tuned on September.
    reference=pd.read_csv(ROOT/'artifacts/results/v5_day_ahead_predictions.csv')
    rerun=pd.read_csv(OUT/'v5_reproduction/results/v5_day_ahead_predictions.csv')
    np.testing.assert_array_equal(reference.timestamp,rerun.timestamp)
    np.testing.assert_allclose(reference.profile_hgb_blend,rerun.profile_hgb_blend,rtol=0,atol=1e-8)
    np.testing.assert_allclose(reference.actual_smp,test.actual_smp,rtol=0,atol=1e-8)
    predictions['historical_test']['V5_frozen']=reference.profile_hgb_blend.to_numpy()
    frozen=metrics(test,reference.profile_hgb_blend,threshold)
    assert len(test)==2208 and abs(frozen['MAE']-11.093786148024114)<1e-6
    assert abs(frozen['R2']-.6429194848696833)<1e-6
    table=pd.concat([table,pd.DataFrame([dict(split='historical_test',model='V5_frozen',**frozen)])],ignore_index=True)
    table.to_csv(OUT/'metrics.csv',index=False)
    manifest['V5_reproduction']='2208 predictions match at absolute tolerance 1e-8; MAE/R2 match 1e-6'
    manifest['EV_definition']='hypothetical SMP-linked tariff, 1 kW constant, consecutive 2/3/4/5/6/8 hours, full 24h availability; cost in KRW, durations equally weighted; not retail tariff or V2G revenue'
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
    for s,f in frames.items():
        output=f[['timestamp','target_date','actual_smp']].copy()
        for n,p in predictions[s].items(): output[n]=p
        output.to_csv(OUT/(s+'_predictions.csv'),index=False)
    import matplotlib.pyplot as plt
    f=test; p=predictions['historical_test'][winner]
    for period,mask in [('quarter',np.ones(len(f),dtype=bool)),('first_week',np.arange(len(f))<168)]:
        fig,ax=plt.subplots(figsize=(14,4))
        ax.plot(pd.to_datetime(f.timestamp)[mask],f.actual_smp[mask],label='Actual',linewidth=.8)
        ax.plot(pd.to_datetime(f.timestamp)[mask],reference.profile_hgb_blend[mask],label='V5',alpha=.7,linewidth=.7)
        ax.plot(pd.to_datetime(f.timestamp)[mask],p[mask],label=winner,alpha=.7,linewidth=.7)
        ax.set_ylabel('SMP (KRW/kWh)'); ax.legend(); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(OUT/(period+'.png')); plt.close(fig)
    diagnostic=f.assign(error=p-f.actual_smp,abs_error=abs(p-f.actual_smp)).groupby('target_hour').agg(MAE=('abs_error','mean'),Bias=('error','mean'))
    diagnostic.to_csv(OUT/'hourly_errors.csv')
    print(table[table.split=='historical_test'].to_string(index=False),flush=True)
    print('validation-selected:',winner,flush=True)

if __name__=='__main__': run()
