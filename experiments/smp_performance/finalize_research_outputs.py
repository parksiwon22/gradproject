"""Create immutable, clearly labeled final research summaries and a demo plot."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs' / 'smp_improvement_ml_lgbm'


def main():
    korean_font = Path(r'C:\Windows\Fonts\malgun.ttf')
    if korean_font.exists():
        font_manager.fontManager.addfont(str(korean_font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(korean_font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    summary = pd.read_csv(OUT / 'all_fold_paired_summary.csv')
    chosen = {1: 'V11_ExtraTrees_fold_refit', 2: 'ml_anchor_blend', 3: 'ml_anchor_blend'}
    rows = []
    for h, name in chosen.items():
        group = summary[summary.horizon_days.eq(h)].set_index('model')
        winner = group.loc[name]
        baseline = group.loc['persistence']
        rows.append(dict(
            horizon_days=h, selected_model=name, days=int(winner.days),
            MAE=winner.MAE, RMSE=winner.RMSE, G2V_regret=winner.G2V_regret,
            baseline='persistence', baseline_MAE=baseline.MAE,
            baseline_RMSE=baseline.RMSE, baseline_G2V_regret=baseline.G2V_regret,
            MAE_reduction_pct=100*(baseline.MAE-winner.MAE)/baseline.MAE,
            RMSE_reduction_pct=100*(baseline.RMSE-winner.RMSE)/baseline.RMSE,
            regret_reduction_pct=100*(baseline.G2V_regret-winner.G2V_regret)/baseline.G2V_regret,
            evaluation='275 retrospective fold validation days; not independent unseen test',
        ))
    final = pd.DataFrame(rows)
    final.to_csv(OUT / 'final_recommended_models.csv', index=False)

    common_rows = []
    for horizon, path in [
        (2, ROOT / 'docs' / 'smp_improvement_ml' / 'd2_2025q4_common_date_comparison.csv'),
        (3, ROOT / 'docs' / 'smp_improvement_ml' / 'd3_v17_v19_common_date_comparison.csv'),
    ]:
        old = pd.read_csv(path)
        for r in old.to_dict('records'):
            common_rows.append(dict(horizon_days=horizon, model=r['model'], days=r['days'],
                MAE=r['MAE'], RMSE=r.get('RMSE', np.nan), G2V_regret=r['G2V_regret'],
                source=r.get('source', 'preserved V5-V20 score'), assessment=r.get('assessment', 'reused 2025Q4 comparison; retrospective')))
    lgb_log = pd.read_csv(OUT / 'lightgbm_experiment_log.csv')
    lgb_log = lgb_log[(lgb_log.fold == 'fall2025') & (lgb_log.model == 'lightgbm_anchor_blend')]
    for r in lgb_log.itertuples():
        common_rows.append(dict(horizon_days=int(r.horizon_days), model='lightgbm_anchor_blend',
            days=int(r.validation_days), MAE=r.validation_MAE, RMSE=r.validation_RMSE_pooled,
            G2V_regret=r.validation_G2V_regret, source='new fold-trained LightGBM/anchor replay',
            assessment='2025Q4 previously reviewed; retrospective, vintage unverified'))
    pd.DataFrame(common_rows).to_csv(OUT / 'common_2025q4_model_summary.csv', index=False)

    hourly = pd.read_csv(OUT / 'd2_d3_hourly_predictions.csv')
    d1 = pd.read_csv(OUT / 'd1_v11_extratrees_fold_hourly.csv')
    all_hourly = pd.concat([hourly, d1], ignore_index=True)

    folds = {f['name']: f for f in json.loads((OUT / 'study_manifest.json').read_text(encoding='utf-8'))['folds']}
    old_hourly = pd.read_csv(ROOT / 'docs' / 'smp_improvement_ml' / 'hourly_predictions.csv')
    parts = []

    def add_series(source, horizon, models, source_name):
        for model in models:
            frame = source[(source.horizon_days == horizon) & (source.model == model)]
            for fold, fg in frame.groupby('fold'):
                meta = folds[fold]
                for target_date, day_frame in fg.groupby('target_date'):
                    series = '기준선 SMP' if model == 'persistence' else model
                    parts.append(pd.DataFrame({
                        'target_date': target_date, 'horizon_days': horizon, 'fold': fold,
                        'hour': day_frame.hour.astype(int), 'series': series,
                        'smp': day_frame.predicted_smp.astype(float), 'train_end': meta['train_end'],
                        'calibration_start': meta['cal_start'], 'calibration_end': meta['cal_end'],
                        'validation_start': meta['val_start'], 'validation_end': meta['val_end'],
                        'evaluation_type': 'RETROSPECTIVE_FOLD_REPLAY_NOT_OPERATIONAL',
                        'input_vintage_verified': False, 'model_available_at_origin': False,
                        'source_file': source_name,
                    }))

    for horizon in (1, 2, 3):
        add_series(old_hourly, horizon, ['persistence'], 'V20 HGB fold persistence baseline')
        for fold, fg in old_hourly[(old_hourly.horizon_days == horizon) & (old_hourly.model == 'persistence')].groupby('fold'):
            meta = folds[fold]
            for target_date, day_frame in fg.groupby('target_date'):
                parts.append(pd.DataFrame({
                    'target_date': target_date, 'horizon_days': horizon, 'fold': fold,
                    'hour': day_frame.hour.astype(int), 'series': '실제 SMP',
                    'smp': day_frame.actual_smp.astype(float), 'train_end': meta['train_end'],
                    'calibration_start': meta['cal_start'], 'calibration_end': meta['cal_end'],
                    'validation_start': meta['val_start'], 'validation_end': meta['val_end'],
                    'evaluation_type': 'RETROSPECTIVE_FOLD_REPLAY_NOT_OPERATIONAL',
                    'input_vintage_verified': False, 'model_available_at_origin': False,
                    'source_file': 'timestamp-causal historical SMP',
                }))
        if horizon == 1:
            add_series(d1, horizon, ['V11_ExtraTrees_fold_refit'], 'D1 ExtraTrees fold refit')
            add_series(old_hourly, horizon, ['hgb', 'ml_anchor_blend'], 'HGB candidate fold forecasts')
        else:
            add_series(old_hourly, horizon, ['hgb', 'ml_anchor_blend'], 'HGB candidate fold forecasts')
            add_series(hourly, horizon, ['lightgbm_v12', 'lightgbm_anchor_blend'], 'LightGBM candidate fold forecasts')
    demo = pd.concat(parts, ignore_index=True).drop_duplicates(['target_date', 'horizon_days', 'hour', 'series'])
    demo.to_csv(OUT / 'final_fold_demo.csv.gz', index=False, compression='gzip')
    chosen_hourly = []
    for h, name in chosen.items():
        frame = all_hourly[(all_hourly.horizon_days == h) & (all_hourly.model == name)].copy()
        frame['abs_error'] = frame.error.abs()
        chosen_hourly.append(frame.groupby('hour').abs_error.mean().rename(f'D+{h} MAE'))
    pd.concat(chosen_hourly, axis=1).rename_axis('hour').reset_index().to_csv(OUT / 'final_hourly_mae.csv', index=False)

    day = '2025-12-31'
    chosen_series = {1: ('V11_ExtraTrees_fold_refit', d1), 2: ('ml_anchor_blend', hourly), 3: ('ml_anchor_blend', hourly)}
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, (h, (model, data)) in zip(axes, chosen_series.items()):
        actual = data[(data.horizon_days == h) & (data.target_date == day)].drop_duplicates('hour').sort_values('hour')
        pred = data[(data.horizon_days == h) & (data.target_date == day) & (data.model == model)].sort_values('hour')
        ax.plot(actual.hour, actual.actual_smp, label='실제 SMP', linewidth=2)
        ax.plot(pred.hour, pred.predicted_smp, label=f'D+{h} {model}', linewidth=1.6)
        ax.set_ylabel('원/kWh'); ax.set_title(f'{day} · D+{h} · retrospective fold replay'); ax.grid(alpha=.25); ax.legend()
    axes[-1].set_xlabel('시간 (KST)')
    fig.suptitle('최종 연구 후보 예측과 실제 SMP · 2025-12-31')
    fig.tight_layout()
    fig.savefig(OUT / 'final_representative_forecast.png', dpi=160)
    plt.close(fig)

    print(final.to_string(index=False))
    print('hourly_rows', len(all_hourly), 'representative date', day)


if __name__ == '__main__':
    main()
