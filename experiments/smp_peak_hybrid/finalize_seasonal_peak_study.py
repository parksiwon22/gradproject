"""Rebuild clean same-fold tables from the already generated peak-study outputs."""
from pathlib import Path
import numpy as np
import pandas as pd

from v11.store import ROOT
from experiments.smp_peak_hybrid.run_peak_hybrid_study import aggregate
from experiments.smp_performance.train_ml_candidates import FOLDS

OUT = ROOT / 'docs' / 'smp_peak_hybrid'


def mdtable(frame, floatfmt='.3f'):
    values=[]
    for row in frame.itertuples(index=False, name=None):
        values.append([format(x,floatfmt) if isinstance(x,(float,np.floating)) and np.isfinite(x)
                       else ('NA' if isinstance(x,(float,np.floating)) else str(x)) for x in row])
    cols=[str(c) for c in frame.columns]
    lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    lines += ['| '+' | '.join(map(str,row))+' |' for row in values]
    return '\n'.join(lines)


def append_source(rows, frame, horizon, models, prediction_col='predicted_smp'):
    for model in models:
        g = frame[(frame.horizon_days == horizon) & (frame.model == model)]
        for r in g.itertuples():
            rows.append(dict(horizon_days=horizon, fold=r.fold,
                target_date=str(r.target_date)[:10], hour=int(r.hour), actual_smp=float(r.actual_smp),
                predicted_smp=float(getattr(r, prediction_col)), model=model))


def main():
    root = ROOT / 'docs'
    old = pd.read_csv(root/'smp_improvement_ml'/'hourly_predictions.csv')
    et = pd.read_csv(root/'smp_improvement_ml_lgbm'/'d1_v11_extratrees_fold_hourly.csv')
    lgb = pd.read_csv(root/'smp_improvement_ml_lgbm'/'d2_d3_hourly_predictions.csv')
    first = pd.read_csv(OUT/'peak_hybrid_hourly_predictions.csv')
    first = first[first.model.eq('gated_peak_hybrid')]
    seasonal = pd.read_csv(OUT/'seasonal_hybrid_hourly_predictions.csv')
    rows=[]
    for h in (1,2,3):
        append_source(rows,old,h,['persistence','weekday_median4','hgb','peak_weighted','ml_anchor_blend'])
        if h == 1:
            append_source(rows,et,h,['V11_ExtraTrees_fold_refit'])
        else:
            append_source(rows,lgb,h,['lightgbm_v12','lightgbm_v12_peak','lightgbm_anchor_blend'])
    for r in first.itertuples():
        rows.append(dict(horizon_days=int(r.horizon_days),fold=r.fold,target_date=str(r.target_date)[:10],
            hour=int(r.hour),actual_smp=float(r.actual_smp),predicted_smp=float(r.gated_hybrid_prediction),model='global_peak_gate'))
    for r in seasonal.itertuples():
        rows.append(dict(horizon_days=int(r.horizon_days),fold=r.fold,target_date=str(r.target_date)[:10],hour=int(r.hour),
            actual_smp=float(r.actual_smp),predicted_smp=float(r.prediction),model=f'{r.season_rule}_seasonal_peak_gate'))
    forecasts=pd.DataFrame(rows)
    fold_rows=[]
    for (h,fold,model),g in forecasts.groupby(['horizon_days','fold','model'],sort=True):
        day_counts=g.groupby('target_date').size()
        if len(g) and (day_counts.eq(24).all()) and g.hour.nunique()==24:
            fold_rows.append(dict(horizon_days=h,fold=fold,model=model,**aggregate(g),
                evaluation_type='chronological fold replay; first_seen vintage unverified'))
    fold_scores=pd.DataFrame(fold_rows)
    pooled_rows=[]
    for (h,model),g in forecasts.groupby(['horizon_days','model'],sort=True):
        per_fold=g.groupby('fold').target_date.nunique()
        # Require 24 hourly values per date and all three validation quarters.
        valid=(g.groupby(['fold','target_date']).size().eq(24).all() and len(per_fold)==3)
        if valid:
            pooled_rows.append(dict(horizon_days=h,model=model,folds=int(per_fold.size),**aggregate(g),
                evaluation_type='same 275 validation dates; repeated historical research; not operational'))
    pooled=pd.DataFrame(pooled_rows)
    fold_scores.to_csv(OUT/'final_same_fold_peak_comparison.csv',index=False)
    pooled.to_csv(OUT/'final_pooled_peak_comparison.csv',index=False)
    baseline={1:'V11_ExtraTrees_fold_refit',2:'ml_anchor_blend',3:'ml_anchor_blend'}
    delta=[]
    for (h,fold),g in fold_scores.groupby(['horizon_days','fold']):
        b=g[g.model.eq(baseline[h])]
        if b.empty: continue
        b=b.iloc[0]
        for _,r in g.iterrows():
            if r.model==baseline[h]: continue
            delta.append(dict(horizon_days=h,fold=fold,baseline=baseline[h],model=r.model,
                delta_MAE=r.MAE-b.MAE,delta_RMSE=r.RMSE-b.RMSE,delta_peak_MAE=r.peak_MAE-b.peak_MAE,
                delta_normal_MAE=r.normal_MAE-b.normal_MAE,delta_daily_max_error=r.daily_max_abs_error-b.daily_max_abs_error,
                delta_peak_hour_error=r.daily_peak_hour_error-b.daily_peak_hour_error,delta_G2V_regret=r.G2V_regret-b.G2V_regret,
                positive_delta_means_worse=True))
    pd.DataFrame(delta).to_csv(OUT/'fold_delta_failure_cases.csv',index=False)

    # Compact training-only fixed-season peak timing and uplift stability summary.
    dist=pd.read_csv(OUT/'training_season_hour_distribution.csv')
    fixed=dist[dist.rule.eq('fixed_season')]
    stable=[]
    for (h,fold,season),g in fixed.groupby(['horizon_days','fold','group']):
        g=g.sort_values(['peak_rate','mean_peak_uplift_over_daily'],ascending=False)
        stable.append(dict(horizon_days=h,fold=fold,season=season,
            top3_hours=','.join(map(str,g.head(3).hour.astype(int))),
            top3_peak_rate=','.join(f'{x:.3f}' for x in g.head(3).peak_rate),
            top3_peak_mean_smp=','.join(f'{x:.1f}' for x in g.head(3).mean_peak_smp),
            top3_peak_uplift=','.join(f'{x:.1f}' for x in g.head(3).mean_peak_uplift_over_daily),
            min_hour_samples=int(g.n_samples.min()),max_hour_samples=int(g.n_samples.max()),
            data_partition='training only'))
    pd.DataFrame(stable).to_csv(OUT/'training_season_hour_stability_summary.csv',index=False)

    latest=dist[(dist.horizon_days.eq(3))&(dist.fold.eq('fall2025'))&(dist.rule.eq('month'))].copy()
    monthly=[]
    for month,g in latest.groupby('month',sort=True):
        n=int(g.n_samples.sum());events=int(g.peak_hours.sum());w=g.peak_hours.to_numpy(float)
        top=g.sort_values(['peak_rate','mean_peak_uplift_over_daily'],ascending=False).iloc[0]
        monthly.append(dict(month=int(month),samples=n,peak_hours=events,peak_rate=events/n if n else np.nan,
            mean_peak_smp=float(np.average(g.mean_peak_smp.fillna(0),weights=w)) if w.sum() else np.nan,
            mean_peak_uplift_over_daily=float(np.average(g.mean_peak_uplift_over_daily.fillna(0),weights=w)) if w.sum() else np.nan,
            highest_peak_rate_hour=int(top.hour),highest_peak_rate=float(top.peak_rate),
            highest_peak_hour_sample_count=int(top.n_samples),training_cutoff='2025-06-30',horizon_days=3))
    monthly=pd.DataFrame(monthly)
    monthly.to_csv(OUT/'training_month_peak_summary_d3_fall2025.csv',index=False)
    clusters=pd.read_csv(OUT/'training_month_cluster_map.csv')
    clusters=clusters[(clusters.horizon_days.eq(3))&(clusters.fold.eq('fall2025'))]
    clusters.to_csv(OUT/'training_month_cluster_map_d3_fall2025.csv',index=False)

    ab=pd.read_csv(OUT/'calendar_feature_ablation.csv')
    ab_summary=ab.groupby(['horizon_days','variant','removed_group'],dropna=False).agg(
        mean_delta_AP=('delta_average_precision','mean'),mean_delta_brier=('delta_brier','mean'),folds=('fold','nunique')).reset_index()
    ab_summary.to_csv(OUT/'calendar_feature_ablation_summary.csv',index=False)
    fold_table=pd.DataFrame(FOLDS)[['name','train_end','cal_start','cal_end','val_start','val_end']]

    lines=['# 계절·시간 상호작용 피크 예측 연구 결과','',
        '## 범위와 방법','',
        '- 기존 D+1 V11 ExtraTrees 및 D+2/D+3 HGB-anchor 후보, 원본 HGB, 피크 가중 HGB, V12 LightGBM 계열을 기존 저장 fold 예측과 비교했습니다.',
        '- 신규 후보는 기존 21개 V11/V17 피처를 그대로 사용했습니다. 월·시간·요일 변수를 추가하지 않았고, 학습 구간에서 고정 계절(DJF/MAM/JJA/SON) 또는 월별 24시간 피크 빈도 프로파일 KMeans(k=4)로 모델을 라우팅했습니다.',
        '- 피크 분류기와 SMP≥150 가격 회귀기를 별도 학습했습니다. 예측에서는 분류 확률과 회귀값만으로 보정 여부를 판단했습니다. 실제 피크 라벨은 보정 입력으로 쓰지 않았습니다.',
        '- 임계값·보정비율 선택은 각 fold의 calibration 구간에만 제한했습니다. MAE +0.10, RMSE +0.25, 비피크 MAE +0.10, G2V regret +2.0 이내를 지키며 피크 MAE를 줄이는 설정이 없으면 보정을 비활성화하도록 했습니다.',
        '- 평가 fold는 fall2024, spring2025, fall2025의 validation 구간(각 horizon 275일)입니다. 반복 검토된 과거 구간이며 독립적인 미사용 테스트가 아닙니다. 원본 수신/공개 first_seen과 당시 모델 가용성은 검증되지 않아 모두 retrospective fold replay입니다.','',
        mdtable(fold_table),'',
        '## 기존 달력 피처의 피크 분류 기여','',
        '아래 차이는 calibration 분기에서 계산한 평균입니다. 피처 제거는 남은 피처로 재학습했고, permutation은 학습된 전체 분류기에서 한 그룹을 함께 섞었습니다. `delta_average_precision`이 음수면 피처 제거/섞기가 AP를 낮췄다는 뜻으로 해당 그룹의 기여와 일치합니다.','',
        mdtable(ab_summary,floatfmt='.4f'),'','## 동일 fold pooled 비교','',
        'MAE/RMSE/피크 MAE는 원/kWh, 일별 최고 가격 오차와 피크시각 오차는 일별 평균(가격: 원/kWh, 시각: 시간), 오경보율은 실제 비피크 시간 중 예측 피크 비율, G2V regret은 EV 60kWh·SOC 30→80%·7kW·효율 0.90·24시간 선택 조건에서 일평균 실제 SMP 비용 대비 oracle 최소 비용 차이(원)입니다.','',
        mdtable(pooled[['horizon_days','model','days','MAE','RMSE','peak_MAE','normal_MAE','daily_max_abs_error','daily_peak_hour_error','false_alarm_rate','G2V_regret']]),'']
    lines += ['## 기준 모델 대비 fold 차이','',
        '`delta`는 후보−기준이며 양수는 악화입니다. 세 fold 가운데 유리한 fold 수를 함께 확인하고, 특정 한 fold의 개선을 일반화로 주장하지 않습니다.','']
    deltas=pd.DataFrame(delta)
    for h in (1,2,3):
        g=deltas[deltas.horizon_days.eq(h)]
        stats=g.groupby('model').agg(
            folds=('fold','nunique'),MAE_delta=('delta_MAE','mean'),RMSE_delta=('delta_RMSE','mean'),
            peak_MAE_delta=('delta_peak_MAE','mean'),normal_MAE_delta=('delta_normal_MAE','mean'),
            daily_max_delta=('delta_daily_max_error','mean'),peak_hour_delta=('delta_peak_hour_error','mean'),
            regret_delta=('delta_G2V_regret','mean')).reset_index()
        better=[]
        for model,mg in g.groupby('model'):
            better.append(dict(model=model,MAE_better_folds=int((mg.delta_MAE<0).sum()),
                peak_MAE_better_folds=int((mg.delta_peak_MAE<0).sum()),
                regret_better_folds=int((mg.delta_G2V_regret<0).sum())))
        stats=stats.merge(pd.DataFrame(better),on='model',how='left')
        lines += [f'### D+{h}','','평균 fold 차이(후보−기준):','',mdtable(stats),'']
    lines += ['## 학습 구간에서 본 계절별 시간대','',
        '시간대 순위와 표본 수는 각 fold 학습 구간에서만 계산했습니다. 날짜가 겹치는 fold/horizon 행은 독립 표본처럼 합산하지 않았습니다. 월별 표는 D+3 fall2025 fold의 2025-06-30 cutoff 학습 구간만 집계했습니다.','',
        '### 월별 발생률 및 피크 가격','',mdtable(monthly),'',
        '### 고정 계절별 시간대 상위 3개','',mdtable(pd.DataFrame(stable).query("horizon_days == 3 and fold == 'fall2025'")),'',
        '### 데이터 기반 월 군집','',mdtable(clusters[['month','data_cluster','train_end','selection_source']]),'',
        '## 결론','',
        '- 계절 분리/데이터 기반 월 군집 게이트는 일부 fold에서 피크 MAE와 regret을 줄였지만, pooled 기준으로 전체 MAE가 높아져 현 단계의 최종 모델 교체 근거가 되지 않습니다.',
        '- 피크 가중 LightGBM은 피크 MAE를 크게 낮추는 대신 전체 MAE, 급등 오경보 및/또는 regret을 악화시켰습니다. 이는 피크 전용 최적화의 trade-off로 유지해야 합니다.',
        '- D+3 LightGBM-anchor는 pooled MAE가 10.396으로 현재 HGB-anchor 10.429보다 0.033 낮고 regret도 0.33 낮지만, MAE 개선은 3개 fold 중 1개에서만 나타나 승격 근거가 약합니다.',
        '- 따라서 D+1 V11 ExtraTrees, D+2/D+3 기존 HGB-anchor를 현 최종 연구 후보로 유지합니다. 이번 결과는 별도 연구 산출물이며 V20·공식 발급 gate·운영 모델에 연결하지 않았습니다.',
        '- 표본의 SMP≥150 발생률이 높고 계절·시각 패턴의 변화가 있어, 임계값 분류와 가격 회귀의 성능은 레짐 변화 및 가격 기준 공개시각 검증에 민감합니다. 실운영 시점 vintage 검증, 더 많은 비중복 기간 검증이 남아 있습니다.','']
    lines += ['## 재현 명령','',
        '저장된 기존 비교 결과를 바탕으로 같은 순서로 다시 계산할 때 repository root에서 실행합니다.','',
        '```powershell',
        '.venv-smp-research\\Scripts\\python.exe -m experiments.smp_peak_hybrid.run_peak_hybrid_study',
        '.venv-smp-research\\Scripts\\python.exe -m experiments.smp_peak_hybrid.run_seasonal_peak_interactions',
        '.venv-smp-research\\Scripts\\python.exe -m experiments.smp_peak_hybrid.finalize_seasonal_peak_study',
        '```','']
    (OUT/'FINAL_SEASONAL_PEAK_STUDY.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Pooled rows:',len(pooled),'per-horizon counts:',pooled.groupby('horizon_days').size().to_dict())
    print(pooled[['horizon_days','model','days','MAE','RMSE','peak_MAE','daily_max_abs_error','daily_peak_hour_error','false_alarm_rate','G2V_regret']].to_string(index=False))
    print('Output:',OUT/'FINAL_SEASONAL_PEAK_STUDY.md')


if __name__=='__main__':
    main()
