# Data blockers and resolved conditions

## Blocking conditions

None remain for the requested XGBoost Train, Validation, and Final Test run. The target has 0 missing SMP rows across the shared schedule, all 12 input groups are present in the feature manifest, each issue has 72 target hours, and the data leakage checks passed.

## Resolved through the common missing-data rules

- EPSIS returned 1,095 incomplete hourly load rows; they remain missing in `hourly.csv`, have flags in the model input, and use Train-only imputation where applicable.
- Supply reserve has 1085 missing five-minute samples and 676 incomplete hourly aggregates; no zero fill was applied.
- Demand/reserve interval-end comparison found 675 shared incomplete hours, 420 demand-only hours, and 1 reserve-only hours.
- Generation v2 has 24 missing hourly values each for LNG and bituminous coal, and 768 each for anthracite coal, nuclear, and combined coal, all in 2024. Raw gaps stay missing; input medians are fitted on Train only and missing flags are retained.
- ASOS solar blanks are separated into calculated night/day status and missing status; observed zero values remain distinct.
- KMA exact-target forecast absence is retained with availability flags. No future measured weather is substituted.

## Limitations retained in the reports

- EPSIS historical rows do not contain a row-level geography field. The general system feed is treated as mainland based on the separate Jeju feed; this scope inference is documented.
- Actual publication times for measured inputs and hourly SMP were not verified. The 72-hour observed cutoff is a conservative study assumption.
- Group 2 demand-forecast issue times are missing, so only its past forecast history is used.
- Same-month domestic LNG is a conditional historical feature and is not known to have been available at issue time.

These conditions did not stop training because the requested imputation and missingness flags were applied and the SMP target and time alignment were usable.
