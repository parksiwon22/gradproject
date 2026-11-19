# One dated SARIMAX issuance, 2026-11-19 23:00 KST

This package contains the adopted E3 SARIMAX numerical parameters and a sealed, pre-origin input snapshot. It contains no credentials or API responses. `inputs.npz` has numeric state history/exog, 145 bridge+future exog rows, genuine weekly baseline and timestamp/feature labels. `model.npz` is the unchanged refitted checkpoint. `estimator.py` is the original two-stage constrained SARIMAX implementation; no fitting is invoked.

`python issue.py --verify-only` must reproduce all72 preliminary prices. The dated workflow waits for23:00, refreshes main, refilters the frozen state and forecasts again. It records the actual calculation/issuance time, not a fabricated23:00 timestamp. It refuses issuance after the specified presentation-day window or mutation of sealed artifacts. Already-issued results remain unchanged.

This is a manual, one-off workflow, not daily API automation. It does not run the old v11/v17 pipeline. Source inputs were collected before the planned origin; five generation fields have100%missing recent observations and use the original frozen Train imputation. The public UI displays this limitation. Future actual SMP is neither in the package nor displayed as a known target.
