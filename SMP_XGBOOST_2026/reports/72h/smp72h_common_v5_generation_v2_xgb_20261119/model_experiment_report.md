# XGBoost 72-hour experiment

- Model: `XGBRegressor`, one pooled direct model with a `horizon` feature; it predicts every target hour directly and never feeds a predicted or actual future SMP back into later horizons.
- Training rows: Train targets 2023-01-01 through 2024-12-31; model fit uses Train only.
- Setting selection: Validation targets 2025-01-01 through 2025-06-30; three prespecified tree/learning-rate/regularization configurations were compared by pooled MAE.
- Final Test: 2025-07-01 through 2025-12-31, used once after selection.
- Best validation configuration: `{"max_depth": 6, "n_estimators": 600, "learning_rate": 0.03, "min_child_weight": 8, "subsample": 0.9, "colsample_bytree": 0.8, "reg_lambda": 10.0, "reg_alpha": 0.2}`
- Training and tuning wall time: 235.5 seconds.
- Runtime warning: one non-fatal pandas `PerformanceWarning` occurred while adding output columns to the prediction table; model fitting, predictions and metrics completed successfully.
- Selected model: `artifacts\72h\smp72h_common_v5_generation_v2_xgb_20261119\xgboost_model.json`; predictions: `outputs\predictions\72h\smp72h_common_v5_generation_v2_xgb_20261119\predictions_72h.csv`.

| Candidate | Validation MAE (원/kWh) | Validation RMSE | Validation R² | Parameters |
|---:|---:|---:|---:|---|
| 1 | 10.7794 | 14.6874 | 0.49984 | `{"colsample_bytree": 0.85, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 5, "n_estimators": 450, "reg_alpha": 0.1, "reg_lambda": 5.0, "subsample": 0.85}` |
| 2 | 10.4756 | 14.2849 | 0.52688 | `{"colsample_bytree": 0.8, "learning_rate": 0.03, "max_depth": 6, "min_child_weight": 8, "n_estimators": 600, "reg_alpha": 0.2, "reg_lambda": 10.0, "subsample": 0.9}` |
| 3 | 10.5742 | 14.7207 | 0.49757 | `{"colsample_bytree": 0.9, "learning_rate": 0.04, "max_depth": 3, "min_child_weight": 3, "n_estimators": 650, "reg_alpha": 0.0, "reg_lambda": 3.0, "subsample": 0.9}` |

All 12 groups plus past SMP are in the common feature manifest. The approved same-month domestic LNG input makes this a conditional historical experiment and should not be described as an operational forecast.
