# Model experiment report

SARIMAX seasonal period is fixed at 24 hours. Candidate order space is intentionally small; each model uses the same calendar-only exogenous features.

## Validation candidates

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

The candidate with the lowest walk-forward validation MAE was selected. During a split, model parameters are fixed; observed SMP is appended only after each 72-hour forecast origin advances, with refitting disabled.
