# Baseline and blend comparison

Blend weights were selected using Validation only. Alpha means `alpha * XGBoost + (1-alpha) * baseline`.
Selected baseline: `weekly`; alpha: `0.7`.
The E0 row was recomputed from the existing prediction CSV on the same target rows; it was not retrained.
Full-Validation per-day optima for the selected baseline were D+1=0.7, D+2=0.7, D+3=0.7. Walk-forward validation over five later months improved over E0 in 4/5 global folds (mean 3.05%) and 12/15 per-day folds (mean 3.17%). The mean difference is small and the first fold regresses, so one global weight is retained.
Candidate weights and all monthly folds are in `outputs/metrics/blend_candidates_validation.csv` and `outputs/metrics/blend_weight_walk_forward.csv`.

| Split | Experiment | MAE | RMSE | R² | vs recent daily (%) | vs weekly (%) |
|---|---|---:|---:|---:|---:|---:|
| final_test | E0_xgb | 20.0664 | 23.9567 | -0.0788 | -53.3246 | -75.5068 |
| final_test | E1_recent_daily_blend | 17.1225 | 20.9368 | 0.1761 | -30.8309 | -49.7589 |
| final_test | E1_selected_blend | 15.8733 | 19.6201 | 0.2765 | -21.2860 | -38.8330 |
| final_test | E1_weekly_blend | 15.8733 | 19.6201 | 0.2765 | -21.2860 | -38.8330 |
| final_test | E2_train_plus_validation | 15.2004 | 19.2995 | 0.2999 | -16.1445 | -32.9477 |
| final_test | E3_retrained_blend | 12.9631 | 16.8020 | 0.4694 | 0.9508 | -13.3791 |
| final_test | baseline_recent_daily | 13.0875 | 19.7381 | 0.2677 | 0.0000 | -14.4675 |
| final_test | baseline_weekly | 11.4334 | 17.7715 | 0.4064 | 12.6389 | 0.0000 |
| validation | E0_xgb | 10.4756 | 14.2849 | 0.5269 | 26.9851 | 14.2824 |
| validation | E1_recent_daily_blend | 10.1076 | 14.0632 | 0.5414 | 29.5503 | 17.2938 |
| validation | E1_selected_blend | 10.0955 | 14.1528 | 0.5356 | 29.6348 | 17.3930 |
| validation | E1_weekly_blend | 10.0955 | 14.1528 | 0.5356 | 29.6348 | 17.3930 |
| validation | baseline_recent_daily | 14.3473 | 21.6412 | -0.0859 | 0.0000 | -17.3975 |
| validation | baseline_weekly | 12.2211 | 19.0249 | 0.1608 | 14.8193 | 0.0000 |
