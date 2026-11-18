# Train+Validation retraining

The selected parameters were taken from the existing Validation-only candidate search. The model was refit once on Train plus Validation (2023-01-01 through 2025-06-30); Final Test labels were not used for fitting or selection.
Parameters: `{"colsample_bytree": 0.8, "learning_rate": 0.03, "max_depth": 6, "min_child_weight": 8, "n_estimators": 600, "reg_alpha": 0.2, "reg_lambda": 10.0, "subsample": 0.9}`.
Training rows: 65376; features: 458; fit duration: 38.3s.
The selected E1 blend weight was fixed before the refit and reused for E3. No post-refit blend weight was optimized on Final Test.

E0 Validation MAE: 10.4756; E0 Final Test MAE: 20.0664.
E2 refit Final Test MAE: 15.2004; E3 refit-plus-blend Final Test MAE: 12.9631.
These scores use the same Final Test targets and the original E0 evaluation rows.

The previously inspected Final Test period is reused, so this is a re-evaluation of a known period, not a blind future test.
