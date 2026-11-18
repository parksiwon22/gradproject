# LightGBM 72-hour upgrade: final report

## Dataset and evaluation protocol

- Model: LightGBM, 72-hour land SMP forecast at D 23:00 KST.
- Output dataset version: `smp72h_common_v5_generation_v2_lightgbm_20261119`; exact immutable source manifest version: `smp72h_common_v5_generation_v2_xgb_20261119`. The common v5 payload, 458 feature columns, and frozen labels are reused unchanged.
- The common data bundle was read from its existing repository location and was not copied or edited.
- Features: 458 across input groups 1–12. Group 5 is generation v2; all other groups follow the frozen shared panel.
- Train: 2023-01-01 through 2024-12-31; Validation: 2025-01-01 through 2025-06-30; Final Test: 2025-07-01 through 2025-12-31.
- Each forecast uses one observed-input cutoff `forecast_origin - 72 hours` for all horizons. Future target observations were not used as features.
- Both causal baselines were checked against the frozen hourly SMP history in Validation and Final Test: the recent-daily reference is the same hour on the source day one day before the cutoff date; the weekly reference is exactly target time minus 168 hours. All reference hours precede the common cutoff and all values reproduce exactly.
- Final Test was not used for model or blend selection. This period was evaluated before and is a reused test period, not a new blind test.

## Existing LightGBM E0 reproduction and baselines

The saved E0 predictions were matched to v5 by issue and target hour. Split, cutoff, actual SMP, and both causal baseline columns match row-for-row. Saved E0 MAE reproduced from the CSV before comparing new runs. Reproduction details are in `outputs/metrics/e0_reproduction_check.json`.

| Final Test experiment | MAE (KRW/kWh) | RMSE | R² | Bias |
|---|---:|---:|---:|---:|
| Existing LightGBM E0 | 20.3567 | 24.3100 | -0.1108 | 18.7640 |
| Recent daily baseline | 13.0875 | 19.7381 | 0.2677 | 1.1096 |
| Weekly baseline | 11.4334 | 17.7715 | 0.4064 | 1.2681 |
| E1 existing LightGBM, validation-selected blend | 17.3933 | 21.1926 | 0.1558 | 15.2649 |
| E2 Train+Validation refit LightGBM | 16.5111 | 20.7392 | 0.1916 | 13.7437 |
| E3 refit LightGBM, validation-selected blend | 14.6003 | 18.5431 | 0.3537 | 11.2486 |

Validation summary: E0 MAE 10.0290; selected E1 MAE 9.7618; selected Train-only E2 candidate MAE 10.1451; E3 refit-track blend MAE 9.8491. E1 was the lowest Validation MAE across the compared deployable rules, so it is the final selected rule. The Train+Validation refit remains saved and evaluated as E2/E3.

For the validation-selected E1, Final Test MAE changed by -2.9634 KRW/kWh from E0, and by +5.9599 KRW/kWh from the weekly baseline. E3's observed Final Test MAE is lower, but E3 was not selected because its Validation MAE was +0.0872 higher than E1; no test-driven switch was made. A negative change is an improvement.

## Experiments

- **E0:** Existing LightGBM predictions and saved metrics reproduced on the same frozen rows.
- **E1:** Existing LightGBM blended separately with recent-daily and weekly causal baselines. Validation selected `weekly` with model weight α=0.8; selected Validation MAE=9.7618.
- **E2:** Three previously registered LightGBM candidate parameter sets were re-evaluated on the frozen v5 Validation rows. Candidate 2 was selected, with 155 boosting iterations and Validation MAE 10.1451. It was then refit on Train+Validation and scored once on Final Test.
- **E3:** A baseline blend for the selected Train-only Validation model was selected on Validation, then the same baseline family and weight were applied to the Train+Validation refit. Baseline `weekly`, α=0.8.
- **Final selection:** E1 was frozen as the final prediction rule before Final Test because it had the lowest Validation MAE. E2/E3 refit artifacts and scores are retained for comparison.
- **E4 bias correction:** Not adopted. Validation errors and horizon-specific bias are reported; there was not a separately held-out validation period after selecting a correction. Final Test values were not used to select a correction.
- No horizon-specific models or feature ablations were run.

## Split sizes

| Split | Forecast hours | Forecast origins |
|---|---:|---:|
| final_test | 13,104 | 182 |
| train | 52,488 | 729 |
| validation | 12,888 | 179 |

### Final Test by forecast day

| Experiment | Forecast day | MAE | RMSE |
|---|---:|---:|---:|
| E0_existing_lightgbm | D+1 | 20.2321 | 24.1630 |
| E0_existing_lightgbm | D+2 | 20.3159 | 24.2798 |
| E0_existing_lightgbm | D+3 | 20.5222 | 24.4861 |
| E1_existing_best_validation | D+1 | 17.2909 | 21.0639 |
| E1_existing_best_validation | D+2 | 17.3611 | 21.1745 |
| E1_existing_best_validation | D+3 | 17.5279 | 21.3384 |
| E2_lightgbm_train_validation_refit | D+1 | 16.3969 | 20.6037 |
| E2_lightgbm_train_validation_refit | D+2 | 16.4829 | 20.7217 |
| E2_lightgbm_train_validation_refit | D+3 | 16.6535 | 20.8913 |
| E3_refit_plus_selected_baseline | D+1 | 14.5006 | 18.4238 |
| E3_refit_plus_selected_baseline | D+2 | 14.5848 | 18.5340 |
| E3_refit_plus_selected_baseline | D+3 | 14.7154 | 18.6705 |
| baseline_weekly | D+1 | 11.3917 | 17.7093 |
| baseline_weekly | D+2 | 11.4362 | 17.7814 |
| baseline_weekly | D+3 | 11.4722 | 17.8236 |

Final Test target SMP mean was 104.24 vs Validation 117.54 KRW/kWh; population standard deviation was 23.07 vs 20.77. Model input missing-cell rate was 0.655% on Final Test.

Horizon 1–72, monthly metrics, surge/drop error slices, detailed bias, and coverage are in `outputs/metrics/`. The plot files are in `outputs/figures/`.

## Data limits and reproducibility

The 72-hour observation cutoff is a study assumption; actual release timestamps were not verified. The Group 1 EPSIS feed lacks row-level mainland geography, so `data_gate_passed` remains false and this result is conditional. Group 5 missing periods stay missing and use the frozen panel's flags/native LightGBM missing handling; the 2024 periods absent from the source were not zero-filled. Same-month domestic LNG is conditional and not operationally available at historical issue time.

Re-run from the repository root with `python SMP_LightGBM_2026/upgrades/upgrade_72h_20261119/scripts/run_lightgbm_upgrade.py`. The script discovers the immutable common v5 manifest or accepts `--dataset-root`; it writes only to this upgrade folder. Dependencies are listed in `SMP_LightGBM_2026/requirements.txt`.
