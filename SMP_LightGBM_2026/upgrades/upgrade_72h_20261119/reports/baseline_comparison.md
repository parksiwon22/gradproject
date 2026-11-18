# LightGBM baseline blend selection

Existing LightGBM E0 validation selected `weekly` at α=0.8; Validation MAE 9.7618.

Train-only candidate validation for E3 selected `weekly` at α=0.8; Validation MAE 9.8491.

The candidates used the recent-daily SMP and same-hour-previous-week SMP values available at the issue time. α is the LightGBM share. Final Test did not affect either selection. Full grid: `outputs/metrics/e1_validation_blend_grid.csv` and `e3_validation_blend_grid.csv`.
