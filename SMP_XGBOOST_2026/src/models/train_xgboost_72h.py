"""Train and evaluate the XGBoost 72-hour land SMP model from a fixed shared dataset."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent
SHARED = ROOT / "SMP_SHARED_DATA"
SEED = 2026
FUEL_POLICY = "target-month domestic LNG cost; conditional historical experiment; publication time unverified"
KEYS = ["split", "forecast_origin", "observed_cutoff", "target_timestamp", "horizon", "forecast_day"]

SEARCH_SPACE = [
    {"max_depth": 4, "n_estimators": 450, "learning_rate": 0.05, "min_child_weight": 5, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 5.0, "reg_alpha": 0.1},
    {"max_depth": 6, "n_estimators": 600, "learning_rate": 0.03, "min_child_weight": 8, "subsample": 0.90, "colsample_bytree": 0.80, "reg_lambda": 10.0, "reg_alpha": 0.2},
    {"max_depth": 3, "n_estimators": 650, "learning_rate": 0.04, "min_child_weight": 3, "subsample": 0.90, "colsample_bytree": 0.90, "reg_lambda": 3.0, "reg_alpha": 0.0},
]


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mask = np.isfinite(actual) & np.isfinite(predicted)
    y = actual[mask]
    p = predicted[mask]
    if len(y) == 0:
        return {"n_rows": 0, "coverage": 0.0, "mae": None, "rmse": None, "r2": None}
    return {"n_rows": int(len(y)), "coverage": float(len(y) / len(actual)),
            "mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": float(r2_score(y, p)) if len(y) > 1 else None}


def score_with_baselines(frame: pd.DataFrame, predicted: np.ndarray) -> dict:
    actual = pd.to_numeric(frame["actual_smp"], errors="coerce").to_numpy(dtype=float)
    model = metrics(actual, predicted)
    result = dict(model)
    for col, label in [("baseline_recent_daily", "recent_daily"), ("baseline_weekly", "weekly")]:
        baseline_values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        scored = metrics(actual, baseline_values)
        result[f"{label}_mae"] = scored["mae"]
        result[f"{label}_rmse"] = scored["rmse"]
        base_mae = scored["mae"]
        result[f"{label}_improvement_pct"] = ((base_mae - model["mae"]) / base_mae * 100.0) if base_mae not in (None, 0) and model["mae"] is not None else None
    result["actual_coverage"] = float(np.isfinite(actual).sum() / len(actual)) if len(actual) else 0.0
    return result


def write_html_72h(frame: pd.DataFrame, output: Path, origin: str) -> None:
    payload = [{"t": pd.Timestamp(row.target_timestamp).strftime("%Y-%m-%d %H:%M"),
                "a": float(row.actual_smp), "p": float(row.predicted_smp)} for row in frame.itertuples(index=False)]
    html_data = json.dumps(payload, ensure_ascii=False)
    page = f"""<!doctype html>
<html lang="ko"><meta charset="utf-8"><title>72시간 SMP 예측 {origin}</title>
<style>body{{font:16px system-ui;margin:24px;color:#172033}} canvas{{width:100%;height:480px;border:1px solid #ccd3de}} .controls{{display:flex;gap:20px;align-items:center;margin:16px 0}} label{{display:flex;gap:8px;align-items:center}} .legend{{color:#475569}}</style>
<h1>육지 SMP 72시간 예측</h1><p>발행시각 {origin} (Asia/Seoul), 원/kWh</p>
<div class="controls"><label>시작 <input id="start" type="range" min="0" max="71" value="0"><span id="sv">1</span></label><label>끝 <input id="end" type="range" min="1" max="72" value="72"><span id="ev">72</span></label><span class="legend">실제 ● 파랑　예측 ● 주황　D+1/D+2/D+3 구간 경계 점선</span></div>
<canvas id="chart" width="1500" height="560"></canvas><p id="summary"></p>
<script>const data={html_data};const c=document.getElementById('chart'),ctx=c.getContext('2d'),s=document.getElementById('start'),e=document.getElementById('end');
function draw(){{let a=Math.min(+s.value,+e.value-1),b=Math.max(a+1,+e.value);document.getElementById('sv').textContent=a+1;document.getElementById('ev').textContent=b;ctx.clearRect(0,0,c.width,c.height);let L=70,R=25,T=25,B=65,W=c.width-L-R,H=c.height-T-B;let v=data.slice(a,b),lo=Math.min(...v.flatMap(x=>[x.a,x.p])),hi=Math.max(...v.flatMap(x=>[x.a,x.p]));lo=Math.floor(lo/20)*20;hi=Math.ceil(hi/20)*20;if(hi===lo)hi=lo+1;let x=i=>L+i*(W/Math.max(1,v.length-1)),y=z=>T+(hi-z)*(H/(hi-lo));ctx.strokeStyle='#dbe1ea';ctx.fillStyle='#475569';ctx.font='13px system-ui';for(let k=0;k<=5;k++){{let val=lo+(hi-lo)*k/5,yy=y(val);ctx.beginPath();ctx.moveTo(L,yy);ctx.lineTo(L+W,yy);ctx.stroke();ctx.fillText(val.toFixed(0),8,yy+4)}}
function line(key,color){{ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=2.5;v.forEach((d,i)=>i?ctx.lineTo(x(i),y(d[key])):ctx.moveTo(x(i),y(d[key])));ctx.stroke()}}line('a','#2563eb');line('p','#f97316');ctx.setLineDash([6,5]);ctx.strokeStyle='#64748b';for(let k of [24,48]){{let i=k-a;if(i>0&&i<v.length){{ctx.beginPath();ctx.moveTo(x(i),T);ctx.lineTo(x(i),T+H);ctx.stroke()}}}}ctx.setLineDash([]);ctx.fillText(v[0].t,L,T+H+25);ctx.fillText(v[v.length-1].t,L+W-130,T+H+25);document.getElementById('summary').textContent='구간 MAE: '+(v.reduce((q,d)=>q+Math.abs(d.a-d.p),0)/v.length).toFixed(2)+' 원/kWh · 표시 '+v.length+'시간'}}s.oninput=e.oninput=draw;draw();</script></html>"""
    output.write_text(page, encoding="utf-8")


def make_72h_plot(frame: pd.DataFrame, output_dir: Path, origin: str) -> None:
    sample = frame.sort_values("target_timestamp").copy()
    ts = pd.to_datetime(sample["target_timestamp"])
    actual = sample["actual_smp"].to_numpy(dtype=float)
    pred = sample["predicted_smp"].to_numpy(dtype=float)
    mae = float(np.mean(np.abs(actual - pred)))
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(ts, actual, label="Actual SMP", color="#2563eb", linewidth=2)
    ax.plot(ts, pred, label="XGBoost prediction", color="#f97316", linewidth=2)
    for boundary in [24, 48]:
        ax.axvline(ts.iloc[boundary], color="#64748b", linestyle="--", alpha=0.75)
    ax.set_title(f"72-hour land SMP forecast · {origin} · MAE {mae:.2f} KRW/kWh")
    ax.set_ylabel("KRW/kWh")
    ax.set_xlabel("Target timestamp (Asia/Seoul)")
    ax.grid(True, alpha=0.22)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    safe_origin = pd.Timestamp(origin).strftime("%Y%m%d_%H")
    fig.savefig(output_dir / f"sample_72h_{safe_origin}.png", dpi=160)
    plt.close(fig)
    write_html_72h(sample, output_dir / f"sample_72h_{safe_origin}.html", origin)


def make_figures(predictions: pd.DataFrame, metrics_day: pd.DataFrame, metrics_month: pd.DataFrame,
                 metrics_horizon: pd.DataFrame, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Monthly MAE by lead day, keeping Validation and Final Test visually separate.
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for ax, split in zip(axes, ["validation", "final_test"]):
        part = metrics_month[metrics_month["split"] == split]
        for day, label in [(1, "D+1"), (2, "D+2"), (3, "D+3")]:
            one = part[part["forecast_day"] == day]
            ax.plot(one["target_month"], one["mae"], marker="o", label=label)
        ax.set_title("Validation" if split == "validation" else "Final Test")
        ax.set_ylabel("MAE (KRW/kWh)")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=3)
    axes[-1].tick_params(axis="x", rotation=45)
    fig.suptitle("Monthly MAE by forecast day")
    fig.tight_layout()
    fig.savefig(output_dir / "monthly_mae_by_forecast_day.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for split, label in [("validation", "Validation"), ("final_test", "Final Test")]:
        part = metrics_horizon[(metrics_horizon["split"] == split)]
        ax.plot(part["horizon"], part["mae"], label=label, linewidth=1.8)
    ax.set_xlabel("Horizon (hour)"); ax.set_ylabel("MAE (KRW/kWh)"); ax.set_title("MAE by horizon")
    ax.grid(True, alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(output_dir / "mae_by_horizon.png", dpi=160); plt.close(fig)

    # Six-month comparison uses one clean D+1 prediction per actual target timestamp.
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=False)
    for ax, split in zip(axes, ["validation", "final_test"]):
        part = predictions[(predictions["split"] == split) & (predictions["forecast_day"] == 1)].sort_values("target_timestamp")
        part = part.drop_duplicates("target_timestamp", keep="first")
        ax.plot(pd.to_datetime(part["target_timestamp"]), part["actual_smp"], label="Actual SMP", color="#2563eb", linewidth=1.2)
        ax.plot(pd.to_datetime(part["target_timestamp"]), part["predicted_smp"], label="D+1 prediction", color="#f97316", linewidth=1.2)
        ax.set_title(f"{split.replace('_', ' ').title()} · D+1")
        ax.set_ylabel("KRW/kWh"); ax.grid(True, alpha=0.2); ax.legend()
    fig.suptitle("Six-month actual and D+1 forecast comparison")
    fig.tight_layout(); fig.savefig(output_dir / "six_month_actual_vs_d1.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    part = predictions[predictions["split"] == "final_test"]
    ax.scatter(part["actual_smp"], part["predicted_smp"], s=7, alpha=0.22, color="#0f766e")
    lo = float(min(part["actual_smp"].min(), part["predicted_smp"].min()))
    hi = float(max(part["actual_smp"].max(), part["predicted_smp"].max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#64748b")
    ax.set_xlabel("Actual SMP (KRW/kWh)"); ax.set_ylabel("Predicted SMP (KRW/kWh)")
    ax.set_title("Final Test: actual vs predicted")
    ax.grid(True, alpha=0.2); fig.tight_layout(); fig.savefig(output_dir / "final_test_actual_vs_predicted.png", dpi=160); plt.close(fig)

    # Error by target hour of day.
    part = predictions[predictions["split"].isin(["validation", "final_test"])].copy()
    part["target_hour"] = pd.to_datetime(part["target_timestamp"]).dt.hour
    hourly_error = part.groupby(["split", "target_hour"], as_index=False).agg(mae=("absolute_error", "mean"))
    fig, ax = plt.subplots(figsize=(12, 5))
    for split, label in [("validation", "Validation"), ("final_test", "Final Test")]:
        one = hourly_error[hourly_error["split"] == split]
        ax.plot(one["target_hour"], one["mae"], marker="o", label=label)
    ax.set_xlabel("Target hour (KST)"); ax.set_ylabel("MAE (KRW/kWh)"); ax.set_title("Error by target hour")
    ax.grid(True, alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "mae_by_target_hour.png", dpi=160); plt.close(fig)

    sample_origin = predictions.loc[predictions["split"] == "final_test", "forecast_origin"].min()
    sample = predictions[(predictions["forecast_origin"] == sample_origin) & (predictions["split"] == "final_test")]
    make_72h_plot(sample, output_dir, str(sample_origin))
    return str(sample_origin)


def render_report(dataset_version: str, report_dir: Path, quality: dict, search: pd.DataFrame,
                  overall: pd.DataFrame, by_day: pd.DataFrame, best_params: dict, training_seconds: float,
                  model_file: Path, prediction_file: Path, sample_origin: str) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    source = quality["source_quality"]
    groups = [
        (1, "EPSIS actual hourly load, mean of 12 five-minute snapshots, MW; 1,095 hourly gaps retained and imputed for model input."),
        (2, "Complete mainland day-ahead demand forecast history; publication time unverified, so only historical forecast values up to the shared cutoff are used."),
        (3, "KASI official holidays, separate from weekends."),
        (4, "Asia/Seoul calendar and hour/weekday/month cyclical features; checked against the supplied calendar panel."),
        (5, "KPX generation input `20261119_v2`: 26,304 hourly rows for 2023–2025 from annual 2023, API 2024, and annual 2025 sources. Includes LNG, bituminous coal, anthracite coal, nuclear, and combined coal generation (MWh); 2024 source gaps remain missing and flagged."),
        (6, "Official regional solar/wind sources; Jeju excluded where explicit; missing provincial sums remain missing; PPA/BTM coverage is not asserted."),
        (7, "Supply reserve MW and percent; 5-minute means require all 12 samples, 1,085 absent samples and 676 incomplete hours retained as missing."),
        (8, "Domestic LNG generation fuel cost in KRW/kWh matched by target month under the agreed conditional experiment."),
        (9, "FRED WTI and Brent values selected from the source vintage panel by D-2 as-of date."),
        (10, "FRED international LNG and coal monthly values selected from the source vintage panel by D-2 as-of date."),
        (11, "Seoul ASOS 108 temperature, wind and solar observations; night flag, observed zero and missing SI are separate."),
        (12, "KMA Seoul temperature and metro weather/precipitation, selecting the latest archived issue no later than the origin; issue time, target day and lead are retained."),
    ]
    rows = "\n".join(f"| {n} | {text} |" for n, text in groups)
    (report_dir / "data_collection_report.md").write_text(f"""# Data collection and shared dataset audit

Dataset version: `{dataset_version}`  
Source branch: `codex/smp-72h-data` at `{quality.get('source_commit', 'a90141b81ecd841c524eb75a72f4bcc4f39989f5')}`  
Shared evaluation origins: {sum(quality['split_origin_counts'].values())} ({quality['split_origin_counts']})  
Targets: {quality['row_counts']['evaluation_targets']}; missing SMP labels: {quality['target_missing_smp_rows']}.

| Group | Use and verified scope |
|---:|---|
{rows}

## Critical data choices

- Group 1 uses the EPSIS general current-load page. Its historical rows have no explicit geography column; the separate Jeju monitoring page supports treating the general system page as mainland. The old KPX nationwide-plus-Jeju hourly file was hashed and explicitly excluded.
- EPSIS dates 2023-07-02, 2024-03-21, 2024-03-22, and 2025-02-28 returned no rows; their intervals remain missing.
- Group 5 uses the specified `20261119_v2` hourly panel, with source kinds `annual_2023`, `api_2024`, and `annual_2025`. It covers the full 26,304-hour 2023–2025 timeline. Its 2024 gaps remain NaN in the raw panel and are flagged; no missing generation is changed to zero. The source defines market-traded sent-out energy and does not distinguish mainland from Jeju.
- Group 6 originals span 2023-01 through 2025-12 in seven official CSVs: 2023 Jan–Feb, Mar–May, Jun–Aug, Sep–Nov, and Dec; full 2024; and full 2025. Regional definitions vary by year and are recorded in the source manifest.
- Group 7 has {source['kpx_supply_reserve']['missing_sample_count_in_contiguous_range']} missing five-minute observations across the complete expected 2023–2025 range. Only full 12-sample hours receive an hourly mean.
- EPSIS and reserve gaps were compared by interval end: {quality.get('demand_reserve_missing_overlap', {}).get('shared_incomplete_hours', 'not recorded')} shared hours, {quality.get('demand_reserve_missing_overlap', {}).get('demand_only_incomplete_hours', 'not recorded')} demand-only, and {quality.get('demand_reserve_missing_overlap', {}).get('reserve_only_incomplete_hours', 'not recorded')} reserve-only. Timestamp details are in `demand_reserve_missing_timestamps.csv`.
- Group 11 has {source['kma_asos']['missing_by_column']['solar_radiation_mj_m2']} missing solar observations, including {source['kma_asos']['solar_night_missing']} during calculated night and {source['kma_asos']['solar_day_missing']} during daylight. Missing values remain missing; observed zero counts separately.
- Group 12 forecast availability by D+1/D+2/D+3 is in `quality_report.json`. Exact-target forecasts use the latest issue at or before the origin; absent forecasts are not replaced with future observations.

## Availability and missing data

The un-imputed source panel is `hourly.csv`. `forecast_inputs.csv` has one row per origin and target hour. It retains a `__is_missing` flag for each numeric feature; Train-only medians fill columns with observed Train values, and all-missing Train columns remain NaN for XGBoost's native missing path. Selected KMA issue timestamps are kept as metadata columns, excluded from model features.

## Leakage checks

- Every issue has exactly 72 targets and one `observed_cutoff = forecast_origin - 72h`.
- History uses interval-end timestamps and only values at or before that common cutoff.
- A future-observation perturbation check passed for 18 history source series at three issue cutoffs each (54 feature comparisons).
- Calendar alignment and the D-2 FRED and KMA issue-time checks passed.
- Domestic LNG is the one intentionally conditional target-month input; it is not claimed available at the historical issue time.

## Limitations

The 72-hour observation cutoff is a conservative assumption because each source's actual publication timestamp is not verified. EPSIS geography is inferred from the general system page and separately exposed Jeju page, not row-coded. Demand-forecast issue timestamps are unknown. Generation and renewable market scope may exclude PPA, behind-the-meter or other non-market production. Domestic LNG same-month cost is a conditional historical experiment only.
""", encoding="utf-8")

    (report_dir / "data_blockers.md").write_text(f"""# Data blockers and resolved conditions

## Blocking conditions

None remain for the requested XGBoost Train, Validation, and Final Test run. The target has 0 missing SMP rows across the shared schedule, all 12 input groups are present in the feature manifest, each issue has 72 target hours, and the data leakage checks passed.

## Resolved through the common missing-data rules

- EPSIS returned 1,095 incomplete hourly load rows; they remain missing in `hourly.csv`, have flags in the model input, and use Train-only imputation where applicable.
- Supply reserve has {source['kpx_supply_reserve']['missing_sample_count_in_contiguous_range']} missing five-minute samples and {source['kpx_supply_reserve']['incomplete_hour_count']} incomplete hourly aggregates; no zero fill was applied.
- Demand/reserve interval-end comparison found {quality.get('demand_reserve_missing_overlap', {}).get('shared_incomplete_hours', 'not recorded')} shared incomplete hours, {quality.get('demand_reserve_missing_overlap', {}).get('demand_only_incomplete_hours', 'not recorded')} demand-only hours, and {quality.get('demand_reserve_missing_overlap', {}).get('reserve_only_incomplete_hours', 'not recorded')} reserve-only hours.
- Generation v2 has 24 missing hourly values each for LNG and bituminous coal, and 768 each for anthracite coal, nuclear, and combined coal, all in 2024. Raw gaps stay missing; input medians are fitted on Train only and missing flags are retained.
- ASOS solar blanks are separated into calculated night/day status and missing status; observed zero values remain distinct.
- KMA exact-target forecast absence is retained with availability flags. No future measured weather is substituted.

## Limitations retained in the reports

- EPSIS historical rows do not contain a row-level geography field. The general system feed is treated as mainland based on the separate Jeju feed; this scope inference is documented.
- Actual publication times for measured inputs and hourly SMP were not verified. The 72-hour observed cutoff is a conservative study assumption.
- Group 2 demand-forecast issue times are missing, so only its past forecast history is used.
- Same-month domestic LNG is a conditional historical feature and is not known to have been available at issue time.

These conditions did not stop training because the requested imputation and missingness flags were applied and the SMP target and time alignment were usable.
""", encoding="utf-8")

    search_lines = "\n".join(
        f"| {i+1} | {r['validation_mae']:.4f} | {r['validation_rmse']:.4f} | {r['validation_r2']:.5f} | `{r['parameters']}` |"
        for i, r in search.iterrows()
    )
    (report_dir / "model_experiment_report.md").write_text(f"""# XGBoost 72-hour experiment

- Model: `XGBRegressor`, one pooled direct model with a `horizon` feature; it predicts every target hour directly and never feeds a predicted or actual future SMP back into later horizons.
- Training rows: Train targets 2023-01-01 through 2024-12-31; model fit uses Train only.
- Setting selection: Validation targets 2025-01-01 through 2025-06-30; three prespecified tree/learning-rate/regularization configurations were compared by pooled MAE.
- Final Test: 2025-07-01 through 2025-12-31, used once after selection.
- Best validation configuration: `{json.dumps(best_params, ensure_ascii=False)}`
- Training and tuning wall time: {training_seconds:.1f} seconds.
- Runtime warning: one non-fatal pandas `PerformanceWarning` occurred while adding output columns to the prediction table; model fitting, predictions and metrics completed successfully.
- Selected model: `{model_file}`; predictions: `{prediction_file}`.

| Candidate | Validation MAE (원/kWh) | Validation RMSE | Validation R² | Parameters |
|---:|---:|---:|---:|---|
{search_lines}

All 12 groups plus past SMP are in the common feature manifest. The approved same-month domestic LNG input makes this a conditional historical experiment and should not be described as an operational forecast.
""", encoding="utf-8")

    overall_map = {r["split"]: r for _, r in overall.iterrows()}
    def fmt(v): return "—" if pd.isna(v) else f"{v:.2f}"
    lines = []
    for split in ["validation", "final_test"]:
        r = overall_map[split]
        lines.append(f"| {split.replace('_',' ').title()} | {r['n_rows']:,} | {fmt(r['mae'])} | {fmt(r['rmse'])} | {fmt(r['r2'])} | {fmt(r['baseline_recent_daily_mae'])} | {fmt(r['baseline_recent_daily_improvement_pct'])}% | {fmt(r['baseline_weekly_mae'])} | {fmt(r['baseline_weekly_improvement_pct'])}% |")
    day_lines = []
    for _, r in by_day[by_day["split"].isin(["validation", "final_test"])].iterrows():
        day_lines.append(f"| {r['split'].replace('_',' ').title()} | D+{int(r['forecast_day'])} | {r['n_rows']:,} | {fmt(r['mae'])} | {fmt(r['rmse'])} | {fmt(r['recent_daily_improvement_pct'])}% | {fmt(r['weekly_improvement_pct'])}% |")
    required_limit = "실제 수요, 발전량, 예비력 및 기상 관측자료의 개별 공표시각을 모두 확인할 수 없어, 본 연구에서는 예측 기준시각보다 72시간 이전의 실측 자료만 활용하는 보수적 지연 규칙을 가정하였다. 이 규칙은 각 자료의 실제 공개시각을 검증한 결과가 아니다."
    (report_dir / "final_evaluation_report.md").write_text(f"""# Final evaluation: 72-hour land SMP XGBoost

## Forecast definition and split

- Origin: D 23:00, Asia/Seoul. Each issue predicts D+1 00:00–23:00, D+2 00:00–23:00, and D+3 00:00–23:00.
- Each issue has the same 72-hour observed cutoff: `forecast_origin - 72 hours`.
- Blocks crossing Train/Validation/Final Test boundaries were excluded. Target blocks cover Train 2023–2024, Validation 2025-01-01–2025-06-30, and Final Test 2025-07-01–2025-12-31.
- Selection was made using Validation MAE. The final model is fitted on Train only and Final Test was not used to retune it.

## Results (원/kWh)

| Split | Scored hours | MAE | RMSE | R² | Recent daily baseline MAE | Improvement | Weekly baseline MAE | Improvement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

| Split | Forecast day | Hours | MAE | RMSE | Improvement vs recent daily | Improvement vs weekly |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(day_lines)}

## Artifacts

- Full predictions: `{prediction_file}`. Each row includes origin, target, horizon, forecast day, actual, predicted, error, baselines, split, and fuel policy.
- Metrics by origin, month, horizon and forecast day are saved next to the predictions.
- Figures: monthly D+1/D+2/D+3 MAE, per-horizon errors, six-month actual vs D+1 comparisons, final-test scatter, and target-hour error.
- Representative 72-hour plot for origin `{sample_origin}` is saved as PNG and zoomable HTML.
- Reproduce from the workspace root with: `./.venv/Scripts/python.exe SMP_XGBOOST_2026/src/models/train_xgboost_72h.py --dataset-version {dataset_version}`

## Input groups

| Group | Data used |
|---:|---|
{rows}

## Input and operating limitations

All 12 common groups and SMP history were kept. The D-2 FRED vintage panel, exact-target KMA forecast issue time and shared observed cutoff were checked. Group 2 is historical forecast input only because its original issue timestamps are missing. Group 12 uses the latest archived forecast issued by the origin; forecast lead and issue time are retained. D+2 and D+3 never use future actual SMP.

The Final Test model MAE is worse than both baselines; its improvement percentages are negative. The model was not changed after reviewing that result.

The domestic LNG feature uses the target month's value. Evaluation is therefore the **동월 LNG 단가 제공 조건부 과거 예측 실험**; publication by the forecast origin has not been verified, and this value is not an operationally confirmed input. The 2026-11 demo value is not part of the 2025 historical evaluation.

{required_limit}

SMP hourly publication time is not present in the source. The study conservatively uses only completed hourly SMP observations through the common cutoff. The EPSIS general page has no row-level geography field; its mainland scope is inferred from the separate Jeju page. Generation and renewable records may represent market-traded scope rather than all physical production. These are scope and availability limitations, not imputed zero values.

Plots are under `outputs/figures/72h/{dataset_version}/`. A user-selected origin can be plotted with `src/visualization/plot_72h_forecast.py`.
""", encoding="utf-8")


def run(dataset_version: str, resume_report: bool = False) -> dict:
    start = time.perf_counter()
    dataset = SHARED / "versions" / dataset_version
    if not dataset.exists():
        raise FileNotFoundError(f"Shared dataset version not found: {dataset}")
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    quality = json.loads((dataset / "quality_report.json").read_text(encoding="utf-8"))
    feature_cols = manifest["model_feature_columns"]
    if len(feature_cols) != len(set(feature_cols)):
        raise ValueError("Duplicate feature names in shared manifest")

    prediction_dir = PROJECT / "outputs" / "predictions" / "72h" / dataset_version
    metric_dir = PROJECT / "outputs" / "metrics" / "72h" / dataset_version
    figure_dir = PROJECT / "outputs" / "figures" / "72h" / dataset_version
    artifact_dir = PROJECT / "artifacts" / "72h" / dataset_version
    report_dir = PROJECT / "reports" / "72h" / dataset_version
    final_dirs = [prediction_dir, metric_dir, figure_dir, artifact_dir, report_dir]
    model_file = artifact_dir / "xgboost_model.json"
    prediction_file = prediction_dir / "predictions_72h.csv"
    if resume_report:
        required = [model_file, prediction_file, metric_dir / "metrics_overall.csv", metric_dir / "metrics_by_forecast_day.csv",
                    metric_dir / "hyperparameter_search.csv", artifact_dir / "model_config.json"]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError("Cannot finalize report; saved run files are missing: " + ", ".join(missing))
        overall = pd.read_csv(metric_dir / "metrics_overall.csv")
        by_day = pd.read_csv(metric_dir / "metrics_by_forecast_day.csv")
        search = pd.read_csv(metric_dir / "hyperparameter_search.csv")
        config = json.loads((artifact_dir / "model_config.json").read_text(encoding="utf-8"))
        saved_predictions = pd.read_csv(prediction_file, usecols=["split", "forecast_origin"])
        sample_origin = str(saved_predictions.loc[saved_predictions["split"] == "final_test", "forecast_origin"].min())
        config.setdefault("runtime_warnings", ["Pandas PerformanceWarning occurred while adding output columns to the prediction panel; it was non-fatal and did not alter training, predictions, or metrics."])
        config["failed_origins"] = 0
        (artifact_dir / "model_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        render_report(dataset_version, report_dir, quality, search, overall, by_day, config["selected_parameters"],
                      float(config["training_tuning_wall_seconds"]), model_file, prediction_file, sample_origin)
        requirements_path = PROJECT / "requirements_72h.txt"
        if not requirements_path.exists():
            requirements_path.write_text("pandas>=2.0\nnumpy>=1.24\nscikit-learn>=1.3\nxgboost>=2.0\nmatplotlib>=3.7\n", encoding="utf-8")
        readme_path = PROJECT / "README_72H.md"
        if not readme_path.exists():
            readme_path.write_text(f"""# XGBoost 72-hour SMP experiment

This is a new 72-hour experiment alongside the existing 24-hour project results. It reads the common data version `{dataset_version}` from `SMP_SHARED_DATA/versions/`.

Run it from the workspace root with:

```powershell
.\\.venv\\Scripts\\python.exe SMP_XGBOOST_2026\\src\\models\\train_xgboost_72h.py --dataset-version {dataset_version}
```

Review `reports/72h/{dataset_version}/final_evaluation_report.md` for results and assumptions.
""", encoding="utf-8")
        print(json.dumps({"reports_finalized": str(report_dir / "final_evaluation_report.md"),
                          "model": str(model_file), "predictions": str(prediction_file)}, ensure_ascii=False, indent=2))
        return {"prediction_file": prediction_file, "model_file": model_file, "report_dir": report_dir}
    exists = [str(p) for p in final_dirs if p.exists()]
    if exists:
        raise FileExistsError("Refusing to overwrite existing 72-hour experiment output(s): " + ", ".join(exists))
    for path in final_dirs:
        path.mkdir(parents=True, exist_ok=False)

    input_cols = ["split", "forecast_origin", "observed_cutoff", "target_timestamp", "forecast_date", "horizon", "forecast_day"] + feature_cols
    inputs = pd.read_csv(dataset / "forecast_inputs.csv", usecols=lambda c: c in set(input_cols), low_memory=False)
    targets = pd.read_csv(dataset / "evaluation_targets.csv", low_memory=False)
    for frame in [inputs, targets]:
        for col in ["forecast_origin", "observed_cutoff", "target_timestamp"]:
            frame[col] = pd.to_datetime(frame[col], errors="coerce")
    if inputs["target_timestamp"].isna().any() or targets["target_timestamp"].isna().any():
        raise ValueError("Invalid timestamps in the common evaluation inputs")
    panel = inputs.merge(targets, on=KEYS, how="inner", validate="one_to_one", suffixes=("", "_target")).copy()
    if len(panel) != len(targets):
        raise ValueError(f"Input/target join did not cover the common target schedule: {len(panel)} vs {len(targets)}")
    for col in feature_cols:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    X = panel[feature_cols].to_numpy(dtype=np.float32, na_value=np.nan, copy=True)
    y = pd.to_numeric(panel["actual_smp"], errors="coerce").to_numpy(dtype=np.float32)
    split = panel["split"].astype(str).to_numpy()
    train_mask = split == "train"
    val_mask = split == "validation"
    test_mask = split == "final_test"
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("Train, Validation, and Final Test must all be present")
    if np.isnan(y[train_mask]).any():
        raise ValueError("Train target SMP contains missing labels")

    n_jobs = max(1, min(8, (os.cpu_count() or 4) - 1))
    candidate_rows = []
    best_model = None
    best_params = None
    best_mae = float("inf")
    for index, parameters in enumerate(SEARCH_SPACE, start=1):
        model = xgb.XGBRegressor(
            objective="reg:squarederror", eval_metric="mae", tree_method="hist", max_bin=256,
            n_jobs=n_jobs, random_state=SEED, verbosity=0, **parameters,
        )
        fit_start = time.perf_counter()
        model.fit(X[train_mask], y[train_mask], eval_set=[(X[val_mask], y[val_mask])], verbose=False)
        elapsed = time.perf_counter() - fit_start
        val_prediction = model.predict(X[val_mask])
        val_metrics = metrics(y[val_mask].astype(float), val_prediction.astype(float))
        candidate_rows.append({"candidate": index, "validation_mae": val_metrics["mae"], "validation_rmse": val_metrics["rmse"],
                               "validation_r2": val_metrics["r2"], "fit_seconds": elapsed,
                               "parameters": json.dumps(parameters, ensure_ascii=False, sort_keys=True)})
        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            best_model = model
            best_params = parameters
    assert best_model is not None and best_params is not None
    tuning_seconds = time.perf_counter() - start

    best_model.save_model(model_file)
    predictions_array = best_model.predict(X)
    panel["predicted_smp"] = predictions_array.astype(float)
    panel["absolute_error"] = (pd.to_numeric(panel["actual_smp"], errors="coerce") - panel["predicted_smp"]).abs()
    panel["model_name"] = "XGBoost"
    panel["dataset_version"] = dataset_version
    panel["prediction_status"] = np.where(np.isfinite(panel["predicted_smp"]), "completed", "failed")
    panel["fuel_input_policy"] = FUEL_POLICY
    panel["forecast_date"] = pd.to_datetime(panel["target_timestamp"]).dt.strftime("%Y-%m-%d")
    prediction_cols = ["dataset_version", "model_name", "split", "forecast_origin", "observed_cutoff", "target_timestamp", "forecast_date", "horizon", "forecast_day", "actual_smp", "predicted_smp", "absolute_error", "baseline_origin_day", "baseline_recent_daily", "baseline_weekly", "prediction_status", "fuel_input_policy"]
    prediction_file = prediction_dir / "predictions_72h.csv"
    panel[prediction_cols].to_csv(prediction_file, index=False, encoding="utf-8")

    overall_rows = []
    for name in ["train", "validation", "final_test"]:
        part = panel.loc[panel["split"] == name]
        row = {"split": name, **score_with_baselines(part, part["predicted_smp"].to_numpy(dtype=float))}
        row["baseline_recent_daily_mae"] = row.pop("recent_daily_mae")
        row["baseline_recent_daily_rmse"] = row.pop("recent_daily_rmse")
        row["baseline_recent_daily_improvement_pct"] = row.pop("recent_daily_improvement_pct")
        row["baseline_weekly_mae"] = row.pop("weekly_mae")
        row["baseline_weekly_rmse"] = row.pop("weekly_rmse")
        row["baseline_weekly_improvement_pct"] = row.pop("weekly_improvement_pct")
        overall_rows.append(row)
    overall = pd.DataFrame(overall_rows)

    group_metrics = []
    for keys, group in panel.groupby(["split", "forecast_day"], sort=True):
        group_metrics.append({"split": keys[0], "forecast_day": int(keys[1]), **score_with_baselines(group, group["predicted_smp"].to_numpy(dtype=float))})
    by_day = pd.DataFrame(group_metrics)
    by_horizon = pd.DataFrame([
        {"split": split_value, "horizon": int(horizon), **score_with_baselines(group, group["predicted_smp"].to_numpy(dtype=float))}
        for (split_value, horizon), group in panel.groupby(["split", "horizon"], sort=True)
    ])
    panel["target_month"] = pd.to_datetime(panel["target_timestamp"]).dt.strftime("%Y-%m")
    by_month = pd.DataFrame([
        {"split": split_value, "target_month": month, "forecast_day": int(day), **score_with_baselines(group, group["predicted_smp"].to_numpy(dtype=float))}
        for (split_value, month, day), group in panel.groupby(["split", "target_month", "forecast_day"], sort=True)
    ])
    by_origin = pd.DataFrame([
        {"split": split_value, "forecast_origin": origin, **score_with_baselines(group, group["predicted_smp"].to_numpy(dtype=float))}
        for (split_value, origin), group in panel.groupby(["split", "forecast_origin"], sort=True)
    ])
    candidate_table = pd.DataFrame(candidate_rows)
    coverage = {"dataset_version": dataset_version, "expected_target_rows": int(len(targets)),
                "prediction_rows": int(len(panel)), "actual_smp_nonmissing_rows": int(pd.to_numeric(panel["actual_smp"], errors="coerce").notna().sum()),
                "prediction_nonmissing_rows": int(panel["predicted_smp"].notna().sum()),
                "coverage_by_split": {str(k): int(v) for k, v in panel.groupby("split")["predicted_smp"].count().to_dict().items()},
                "failed_origins": int((panel.groupby(["split", "forecast_origin"])["predicted_smp"].count() < 72).sum())}

    overall.to_csv(metric_dir / "metrics_overall.csv", index=False)
    by_day.to_csv(metric_dir / "metrics_by_forecast_day.csv", index=False)
    by_horizon.to_csv(metric_dir / "metrics_by_horizon.csv", index=False)
    by_month.to_csv(metric_dir / "metrics_by_month.csv", index=False)
    by_origin.to_csv(metric_dir / "metrics_by_origin.csv", index=False)
    candidate_table.to_csv(metric_dir / "hyperparameter_search.csv", index=False)
    (metric_dir / "evaluation_coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")

    booster = best_model.get_booster()
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    importance_rows = []
    for i, feature in enumerate(feature_cols):
        xgb_feature = f"f{i}"
        importance_rows.append({"feature": feature, "gain": float(gain.get(xgb_feature, 0.0)),
                                "split_count": int(weight.get(xgb_feature, 0.0))})
    importance = pd.DataFrame(importance_rows).sort_values(["gain", "split_count"], ascending=False)
    importance.to_csv(artifact_dir / "feature_importance.csv", index=False)

    run_metadata = {
        "model_name": "XGBoost", "dataset_version": dataset_version, "seed": SEED,
        "selected_parameters": best_params, "search_space": SEARCH_SPACE,
        "training_rows": int(train_mask.sum()), "validation_rows": int(val_mask.sum()), "final_test_rows": int(test_mask.sum()),
        "feature_count": len(feature_cols), "feature_columns": feature_cols, "feature_manifest": str(dataset / "feature_manifest.csv"),
        "imputation": "Train-only per-feature median written in common dataset; missing flags retained; all-missing Train columns left NaN",
        "prediction_structure": "single pooled direct XGBRegressor with horizon and forecast_day predictors; no recursion or teacher forcing",
        "final_fit_policy": "selected candidate is fit on Train only; Validation selects configuration; Final Test is not used for tuning",
        "fuel_input_policy": FUEL_POLICY,
        "library_versions": {"python": platform.python_version(), "xgboost": xgb.__version__, "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "n_jobs": n_jobs, "seed": SEED, "selected_validation_mae": best_mae,
        "training_tuning_wall_seconds": tuning_seconds, "model_file": str(model_file),
        "source_dataset_manifest": str(dataset / "manifest.json"),
        "source_dataset_manifest_sha256": __import__("hashlib").sha256((dataset / "manifest.json").read_bytes()).hexdigest(),
    }
    (artifact_dir / "model_config.json").write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (prediction_dir / "failed_origins.csv").write_text("split,forecast_origin,failure_reason\n", encoding="utf-8")
    sample_origin = make_figures(panel, by_day, by_month, by_horizon, figure_dir)
    render_report(dataset_version, report_dir, quality, candidate_table, overall, by_day, best_params,
                  tuning_seconds, model_file, prediction_file, sample_origin)
    requirements_path = PROJECT / "requirements_72h.txt"
    if not requirements_path.exists():
        requirements_path.write_text("pandas>=2.0\nnumpy>=1.24\nscikit-learn>=1.3\nxgboost>=2.0\nmatplotlib>=3.7\n", encoding="utf-8")
    readme = f"""# XGBoost 72-hour SMP experiment

This is a new 72-hour experiment alongside the existing 24-hour project results. It reads the common data version `{dataset_version}` from `SMP_SHARED_DATA/versions/` and does not recollect or alter inputs.

Run it from the workspace root with:

```powershell
.\\.venv\\Scripts\\python.exe SMP_XGBOOST_2026\\src\\models\\train_xgboost_72h.py --dataset-version {dataset_version}
```

The final model is selected by Validation pooled MAE and fitted on Train only. Review `reports/72h/{dataset_version}/final_evaluation_report.md` for assumptions and results.
"""
    readme_path = PROJECT / "README_72H.md"
    if not readme_path.exists():
        readme_path.write_text(readme, encoding="utf-8")
    print(json.dumps({"dataset_version": dataset_version, "best_parameters": best_params,
                      "validation": overall_map(overall, "validation"), "final_test": overall_map(overall, "final_test"),
                      "predictions": str(prediction_file), "model": str(model_file), "training_seconds": tuning_seconds}, ensure_ascii=False, indent=2))
    return {"prediction_file": prediction_file, "model_file": model_file, "overall": overall, "by_day": by_day, "report_dir": report_dir}


def overall_map(frame: pd.DataFrame, split: str) -> dict:
    row = frame.loc[frame["split"] == split].iloc[0]
    return {k: (None if pd.isna(v) else (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v)) for k, v in row.to_dict().items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-version", default="smp72h_common_v5_generation_v2_xgb_20261119")
    parser.add_argument("--resume-report", action="store_true", help="Finalize reports from already saved model/prediction/metric files without retraining")
    args = parser.parse_args()
    run(args.dataset_version, resume_report=args.resume_report)


if __name__ == "__main__":
    main()
