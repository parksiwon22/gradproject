# Model experiment report

**Status: incomplete exploratory calendar-only experiment; not a full-input SARIMAX evaluation.** The project now blocks any new model fitting until all selected inputs are complete and point-in-time safe. Existing experiment artifacts are retained only as historical context.

SARIMAX seasonal period is fixed at 24 hours. The legacy experiment used calendar-only exogenous features and omitted all selected operational and market inputs.

## Legacy Validation candidates

```json
[
  {
    "candidate":"SARIMAX_100_100_24",
    "order":[
      1,
      0,
      0
    ],
    "seasonal_order":[
      1,
      0,
      0,
      24
    ],
    "validation_mae":16.742690315,
    "fit_seconds":116.1451874,
    "elapsed_seconds":126.8745366,
    "converged":false
  },
  {
    "candidate":"SARIMAX_001_001_24",
    "order":[
      0,
      0,
      1
    ],
    "seasonal_order":[
      0,
      0,
      1,
      24
    ],
    "validation_mae":30.6452339995,
    "fit_seconds":109.8902457,
    "elapsed_seconds":119.3547637,
    "converged":false
  }
]
```

These are legacy calendar-only metrics. They are not Validation/Final Test results for the selected-input experiment and must not be used as its conclusion.
