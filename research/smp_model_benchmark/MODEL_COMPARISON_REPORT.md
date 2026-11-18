# SMP-only 8-model D+3 comparison

## Summary

This experiment retrains eight algorithms on the same SMP-only chronological D+3 folds: LSTM, GRU, HGB, LightGBM, ExtraTrees, Linear Regression, Random Forest, and XGBoost. It is an additional retrospective benchmark and does not replace V5–V20 research findings, the existing 316-day HGB artifact, or any operating model.

## Data, folds, and leakage controls

- Market: mainland/land SMP from the existing V11 SQLite price store. Target is D 00:00–23:00 KST; forecast origin is D−3 23:00 KST. All input feature timestamps are bounded by each origin. The historical store's original first-seen/publication vintage is not verified, so all scores are timestamp-causal historical replay, not validated operational performance.
- SMP-only inputs: V17 shared price/calendar feature set (including target calendar cycles, D−3 same-hour reference, lagged SMP, and rolling statistics). Weather, demand/supply, renewable generation, fuel, and FX features are excluded.
- Target: residual `actual SMP − D−3 same-hour reference`; final prediction is `0.95 × (reference + residual prediction) + 0.05 × median(SMP at D−7/D−14/D−21/D−28, same hour)`. This 5% anchor weight is fixed for all models. Direct-price predictions are not mixed into this main ranking.
- Each fold uses the published time split: train through June 30 2024 / Dec 31 2024 / June 30 2025; calibration the following quarter; evaluate Oct–Dec 2024 / Apr–Jun 2025 / Oct–Dec 2025. No random split. Tree model parameters are predeclared and not tuned on evaluation days. RNN early stopping and common sequence length selection use calibration only.
- EV: 60 kWh, initial SOC 30%, target SOC 80%, min/max SOC 20/90%, charge/discharge power 7 kW, efficiencies 90%, degradation 5 KRW/kWh, 24-hour connection. V2G is a virtual SMP-linked settlement scenario. Regret is realized cost of the prediction-based fixed schedule on actual SMP minus the actual-SMP oracle schedule cost.

## 8-model result (275 evaluation dates)

MAE and RMSE are pooled over paired hourly predictions; EV regrets are mean daily values.

| Model | Days | MAE | RMSE | Peak MAE | Low-price top-5 overlap | G2V regret/day | G2V actual cost/day | V2G regret/day | V2G actual net cost/day | Parameters / nodes* | Fit s (mean/fold) | Predict s/day |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ExtraTrees | 275 | 10.220 | 14.128 | 18.800 | 0.832 | 71.53 | 3089.69 | 189.01 | 3091.05 | 367,505 nodes | 1.244 | 0.040618 |
| LinearRegression | 275 | 10.591 | 14.660 | 24.784 | 0.826 | 68.62 | 3086.78 | 196.12 | 3098.16 | 21 coef. | 0.014 | 0.000368 |
| RandomForest | 275 | 11.110 | 15.194 | 19.661 | 0.806 | 97.87 | 3116.03 | 224.56 | 3126.60 | 682,895 nodes | 7.650 | 0.046495 |
| LightGBM | 275 | 11.194 | 15.371 | 18.155 | 0.833 | 70.35 | 3088.51 | 191.60 | 3093.64 | 18,300 nodes | 0.435 | 0.000774 |
| HGB | 275 | 11.276 | 15.414 | 18.085 | 0.830 | 71.66 | 3089.82 | 186.08 | 3088.12 | 15,250 nodes | 0.966 | 0.004152 |
| XGBoost | 275 | 11.287 | 15.456 | 18.932 | 0.827 | 72.48 | 3090.64 | 183.63 | 3085.67 | 8,877 nodes | 0.347 | 0.000957 |
| GRU | 275 | 11.821 | 16.414 | 21.771 | 0.772 | 105.59 | 3123.75 | 283.16 | 3185.20 | 35,320 params | 98.240 | 0.135355 |
| LSTM | 275 | 11.852 | 16.446 | 21.467 | 0.777 | 101.79 | 3119.95 | 279.66 | 3181.70 | 36,312 params | 67.251 | 0.117153 |
| 5pct_seasonal_anchor_reference | 275 | 15.649 | 22.220 | 29.488 | 0.759 | 145.22 | 3163.38 | 360.09 | 3262.13 | n/a | nan | nan |
| D3_lag_reference | 275 | 16.219 | 23.052 | 30.014 | 0.727 | 196.84 | 3215.00 | 423.82 | 3325.86 | n/a | nan | nan |

### Recurrent seed variability

Three fixed seeds ((17, 42, 91)) are reported for LSTM and GRU. Main metric uses the mean prediction across the three seeds; the table below shows the mean and standard deviation of the individual-run fold metrics.

| Model | Seed count | MAE mean ± SD | RMSE mean ± SD | G2V regret mean ± SD | V2G regret mean ± SD |
|---|---:|---:|---:|---:|---:|
| LSTM | 3 | 12.307 ± 0.188 | 16.914 ± 0.198 | 118.73 ± 13.40 | 309.68 ± 16.30 |
| GRU | 3 | 12.263 ± 0.216 | 16.878 ± 0.258 | 120.06 ± 12.80 | 306.89 ± 16.88 |

### 168h vs 672h sequence choice

The common sequence length is selected across both architectures and all folds using calibration MAE only.

| Model | Sequence hours | Folds | Calibration days | Mean calibration MAE | Mean selected epoch | Mean fit seconds |
|---|---:|---:|---:|---:|---:|---:|
| GRU | 168 | 3 | 274 | 11.862 | 12.53 | 32.57 |
| GRU | 672 | 3 | 274 | 11.856 | 13.76 | 98.59 |
| LSTM | 168 | 3 | 274 | 11.847 | 13.76 | 24.86 |
| LSTM | 672 | 3 | 274 | 11.842 | 13.31 | 67.56 |


## Result interpretation

- Lowest price MAE: **ExtraTrees**, 10.220 KRW/kWh; lowest G2V regret: **LinearRegression**, 68.62 KRW/day. These are different winners.
- Lowest virtual V2G regret: **XGBoost**, 183.63 KRW/day. V2G remains hypothetical; a lower V2G regret does not establish lower realized cost than G2V.
- GRU vs LSTM ensemble prediction: GRU MAE difference -0.032 KRW/kWh (-0.27%). GRU has 35,320 vs 36,312 parameters (-2.73%) but mean fold training time 98.2s vs 67.3s. Its price error is marginally lower, while its G2V regret is higher; this does not make GRU the preferred overall model.
- The selected RNN window is 672 hours from calibration only; its calibration MAE advantage over 168 hours is small and its training cost is higher. The main evaluation does not tune this choice.
- On this repeated retrospective cohort, ExtraTrees is strongest on aggregate price accuracy, Linear Regression has the lowest G2V regret, and HGB/LightGBM/XGBoost remain competitive tree alternatives. None of these results justify replacing the separately selected V20 models or claiming operational improvement.

## Model structures

- **LSTM**: one 32-unit Long Short-Term Memory recurrent layer encodes the past SMP series; dense layers map its state and known target calendar to 24 residuals. Its cell carries a memory state and input/forget/output gates.
- **GRU**: one 32-unit Gated Recurrent Unit layer uses update/reset gates and a smaller state transition; input, loss, output, normalization, seed set, optimizer, epoch limit, and early-stop rule match LSTM. The fitted parameter-count and runtime difference is recorded in `results/model_runtime.csv`.
- **HGB**: V20 HistGradientBoosting settings (250 iterations, 31 leaves, 0.05 learning rate, L2 2).
- **LightGBM**: V12-style leaf-wise boosted trees, 300 estimators, 31 leaves, 0.05 rate.
- **ExtraTrees**: 150 randomized trees, minimum leaf 10 and 0.8 feature subsampling (V11 settings).
- **Linear Regression**: standardized V17 inputs then ordinary least squares; scaling is fitted on the training fold only.
- **Random Forest**: 200 bagged regression trees, minimum leaf 5, 0.8 feature subsampling.
- **XGBoost**: 300 histogram gradient-boosted trees, depth 4, 0.05 rate, subsampling and L2 regularization.

Neural models output all 24 residuals jointly; tree and linear models estimate hourly residuals row by row. This output-structure difference is a limitation when attributing performance differences to the algorithm alone.

## Uncertainty, stability, and interpretation

Paired day bootstrap intervals against HGB are in `results/paired_bootstrap_vs_hgb.csv`. Fold-specific metrics, hourly errors, top-5 low-price overlap, high-price MAE, actual-SMP realized costs, model fit/predict time and artifact size are in the CSV outputs. Fold dates have been repeatedly reviewed during prior research and are not a fresh independent test. Bootstrap intervals quantify within-cohort variation only.

The lowest aggregate MAE candidate is ExtraTrees (10.220 KRW/kWh), but selection should also consider fold variation, RMSE and charging regret; this repeated historical cohort does not justify an operational model change.

The existing V20 HGB model remains unchanged. Any difference from its prior reported score may arise from the fixed 5% anchor, fixed fold-specific retraining, output structures, library versions, and scoring aggregation; do not present it as an unconditional improvement.

## Environment and reproducibility

Runtime package versions are recorded in `results/environment.json`; fixed model specs and random seeds are in `results/study_manifest.json`. The benchmark environment is separate from existing V20 and weather environments.

```powershell
$env:SMP_SOURCE = (Resolve-Path ..\gradproject).Path
.\.venv-smp-benchmark\Scripts\python.exe -m research.smp_model_benchmark.benchmark
```

## Limits

No sample is verified against original first-seen receipt timestamps. Existing target windows were previously examined. The V2G sale ratio represents a hypothetical SMP-linked scenario, not a guaranteed Korean settlement payment. Models are research-only; no auto-promotion, website, official issuance gate, or operating model was changed.
