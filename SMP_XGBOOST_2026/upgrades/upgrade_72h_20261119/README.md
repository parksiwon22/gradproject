# XGBoost 72-hour upgrade

This isolated upgrade compares the existing E0 run with Validation-selected baseline blending and a Train+Validation refit. It reuses common dataset `smp72h_common_v5_generation_v2_xgb_20261119`, keeps the 12 groups and the existing evaluation rows, and leaves the original model, model file, predictions, and reports untouched.

## Experiments

- **E0:** Existing XGBoost predictions, reread from the original CSV and verified against the fixed target rows and saved metrics.
- **E1:** Blend E0 with the recent-daily or weekly SMP baseline. A single alpha from 0.0 to 1.0 in 0.1 increments is selected on Validation only. Monthly walk-forward and per-day weight diagnostics are recorded; the selected primary blend uses one weight across all 72 hours.
- **E2:** Refit XGBoost on Train+Validation with the parameters selected by the original Validation-only search.
- **E3:** Blend E2 with the E1-selected baseline using the same weight chosen before refitting. No Final Test tuning is performed.

The baselines are checked against `observed_cutoff`: the recent-daily source day ends at or before the cutoff, and the weekly source is the target timestamp minus seven days. The common cutoff remains `forecast_origin - 72 hours` for all 72 targets.

No new data or features are added. A separate bias correction is not applied because this upgrade does not estimate a stable, time-ordered out-of-fold correction for the refit model. Separate D+1/D+2/D+3 models are not trained; the pooled horizon model is retained and their errors are analyzed. The known 2025 Final Test period is reused, so this is not a blind future test.

## Run

From the repository root, install the same dependencies listed in `SMP_XGBOOST_2026/requirements.txt`. If the common dataset version is not already restored, expand the bundled compressed tables with the existing non-overwriting helper:

```powershell
python SMP_XGBOOST_2026/scripts/restore_shared_dataset.py
python SMP_XGBOOST_2026/upgrades/upgrade_72h_20261119/scripts/run_upgrade.py
python SMP_XGBOOST_2026/upgrades/upgrade_72h_20261119/scripts/evaluate_upgrade.py
python -m unittest discover -s SMP_XGBOOST_2026/upgrades/upgrade_72h_20261119/tests
```

The run script refuses to overwrite an existing upgrade model or prediction output. Its configuration records the source hashes, selected alpha, parameter values, row counts, and library versions.

## Outputs

- `models/final_xgboost_model.json`: Train+Validation refit.
- `outputs/predictions/experiments_validation.csv`: E0, baselines, and E1 candidates on the fixed Validation rows.
- `outputs/predictions/final_test_predictions.csv`: E0, baselines, E1, E2, and E3 on the fixed Final Test rows.
- `outputs/metrics/`: candidate selection, E0 verification, experiment comparison, D+1/D+2/D+3 and horizon/month metrics, bias and event-slice analysis.
- `outputs/figures/`: model comparison, selected 72-hour plot, monthly MAE, and horizon MAE.
- `reports/`: baseline comparison, refit notes, bias analysis, and final report.
