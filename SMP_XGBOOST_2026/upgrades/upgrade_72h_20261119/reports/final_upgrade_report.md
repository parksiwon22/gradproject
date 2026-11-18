# Final XGBoost 72-hour upgrade report

Dataset: `smp72h_common_v5_generation_v2_xgb_20261119`. Forecast origin is D 23:00 Asia/Seoul; target is D+1 00:00 through D+3 23:00. All 12 common input groups, the common evaluation rows, and the 72-hour observed cutoff are unchanged.
The final-test period had already been examined in the prior experiment; these results are a re-evaluation and not a blind test. No Final Test metric was used to select hyperparameters, blend weights, or corrections.

Validation-selected blend: `weekly`, alpha `0.7`. Validation rows: 12888; Final Test rows: 13104 per experiment.
Existing E0 Final Test MAE: 20.0664 KRW/kWh. Refit E2 MAE: 15.2004; refit plus selected blend E3 MAE: 12.9631, a 35.40% reduction from E0.
E0 Final Test RMSE/R²: 23.9567/-0.0788; E3 RMSE/R²: 16.8020/0.4694.
Recent-daily baseline MAE: 13.0875; E3 improvement over it: 0.95%. Weekly-baseline MAE on the same Final Test rows: 11.4334; E3 improvement over it: -13.38% (still worse than weekly).

## Data and evaluation limits

- Actual demand, generation, reserve, and weather publication times were not individually verified. The 72-hour cutoff remains a conservative study assumption, not a measured publication delay.
- Domestic LNG is matched by target month under the conditional historical experiment; availability at forecast origin is unverified.
- Original inputs, baseline predictions, fixed origins, and targets are read-only. No new APIs, input groups, or feature-selection experiments were introduced.
- Validation/Test target-price distributions, input missingness, and SMP-history lag-feature distributions are compared in `reports/validation_test_shift_analysis.md`; these diagnostics are not causal proof.
- No separate forecast-day models or bias correction were applied. The same per-day alpha was selected for all three days, while walk-forward per-day gain was not robust enough to justify separate settings.
- Existing group-5 data coverage, source limitations, and imputation remain as documented in the common dataset quality report.
- The existing model and its outputs remain unchanged; all upgrade artifacts are below this folder.

## Files

- Validation experiment predictions: `outputs/predictions/experiments_validation.csv`
- Final Test predictions: `outputs/predictions/final_test_predictions.csv`
- Metrics: `outputs/metrics/`
- Figures: `outputs/figures/`
- New Train+Validation model: `models/final_xgboost_model.json`
- Blend, fit, and source metadata: `config/experiment_config.json`

## Reproduction

Run from repository root after restoring the bundled common dataset if needed:

```powershell
.\.venv\Scripts\python.exe SMP_XGBOOST_2026\upgrades\upgrade_72h_20261119\scripts\run_upgrade.py
.\.venv\Scripts\python.exe SMP_XGBOOST_2026\upgrades\upgrade_72h_20261119\scripts\evaluate_upgrade.py
```

To expand the bundled compressed dataset first, run `python SMP_XGBOOST_2026/scripts/restore_shared_dataset.py`; it refuses to overwrite an existing dataset version.
