# SMP-only eight-model benchmark

This is an isolated D+3 historical replay experiment. It does not modify the V5–V20 models, the existing 316-day HGB artifact, Weather HGB, the public website, operating issuance, or its gate.

The experiment uses the V20 275-date fold layout: train through 2024-06-30 / 2024-12-31 / 2025-06-30, calibrate on the following quarter, then evaluate 2024Q4 / 2025Q2 / 2025Q4. Every prediction is for target-day hours 00–23 from a nominal D−3 23:00 KST origin. Original first-seen receipt times are not established, so these are timestamp-causal historical replays, not verified historical operating forecasts.

All models use V17's SMP/calendar features and predict the residual from the D−3 same-hour reference. Final predictions share a fixed 5% blend with the same-hour median at D−7, D−14, D−21, and D−28. LSTM and GRU emit a 24-hour residual vector from a historical SMP sequence plus known target calendar values. Their common sequence length is selected from 168 or 672 hours using calibration periods only; each selected architecture/fold is trained with three fixed seeds. No target-day realized SMP or other exogenous source enters predictors.

## Run

From repository root in PowerShell:

```powershell
$env:SMP_SOURCE = (Resolve-Path ..\gradproject).Path
.\.venv-smp-benchmark\Scripts\python.exe -m research.smp_model_benchmark.benchmark
```

The benchmark virtual environment is isolated from the earlier research environment. It uses shared system TensorFlow for recurrent models and installs LightGBM/XGBoost locally. Exact package versions are written to `results/environment.json` and `requirements-lock.txt`.

The script writes model checkpoints, daily and hourly predictions, fold summaries, model runtime/size, RNN calibration and seed variability, paired bootstrap intervals, figures, a manifest, and `MODEL_COMPARISON_REPORT.md` only under this directory.

The submitted PR keeps the complete daily and hourly score exports as deterministic gzip files (`results/daily_scores.csv.gz` and `results/hourly_predictions.csv.gz`) to avoid committing duplicate uncompressed exports. Fold checkpoints are generated locally by the command above and are not included in the PR; their measured file sizes are retained in `results/model_runtime.csv`.

## Metrics and scope

Price scores, hourly MAE, high-price MAE (actual SMP >= 150), actual-lowest-five-hour overlap, G2V regret, virtual V2G regret, realized actual-SMP cost, training and inference time, parameter/node counts, and model file size are recorded. Regret compares the actual-SMP cost of a schedule chosen from the forecast with an oracle schedule chosen from actual SMP. EV assumptions: 60 kWh, SOC 30% to 80%, bounds 20–90%, 7 kW charge/discharge, 90% efficiency, 5 KRW/kWh degradation, 24-hour connection. Virtual V2G settlement is a research scenario, not an asserted Korean market payment.

The three evaluation windows and 2026 data (if separately replayed later) have been reviewed in previous project work. Paired bootstrap intervals do not make them independent unseen tests. No benchmark candidate is automatically promoted. Neural parameter counts are exact; tree node totals and linear coefficient totals are complexity descriptors and should not be read as directly comparable parameter counts.
