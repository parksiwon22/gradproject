# Evaluation status

## Full-input experiment: BLOCKED

No Validation or Final Test was run for the selected-input SARIMAX because required explanatory data are incomplete and some forecast-vintage histories remain unverified. This is an intentional fail-closed stop. See [data collection report](data_collection_report.md) for the input-by-input gap table and official retrieval actions.

## Legacy calendar-only experiment: incomplete, exploratory only

The metrics currently stored in the output directory came from the prior calendar-only SARIMAX. They are not complete-input Validation or Final Test results and must not be represented as the requested final model. No new results were generated in this run.

Target SMP is complete from 2023-01-01 through 2025-12-31 (26,304 hourly observations); target availability alone is insufficient to authorize training.

Observed target coverage in processed file: 2023-01-01 00:00:00 through 2025-12-31 23:00:00. Total hourly grid rows: 26304; observed SMP: 26304; missing SMP: 0 (0.00%).

- 2023: 8760/8760 observed hourly SMP
- 2024: 8784/8784 observed hourly SMP
- 2025: 8760/8760 observed hourly SMP

## Archived legacy metrics (calendar-only; not valid full-input results)

| split | group | count | mae | rmse | r2 |
|---|---|---|---|---|---|
| validation | overall_sarimax | 12888 | 16.742690315048122 | 22.408133200576682 | -0.16420154515508822 |
| validation | previous_day_smp | 12888 | 14.31401225946617 | 21.718541922269107 | -0.09364947168141335 |
| validation | previous_week_smp | 12888 | 12.2210917132216 | 19.02490480073182 | 0.1608070995184563 |
| final_test | overall_sarimax | 13104 | 29.465641498175593 | 37.4365532463806 | -1.63425633350235 |
| final_test | previous_day_smp | 13104 | 12.846539224664223 | 19.664118460939168 | 0.2731993901592028 |
| final_test | previous_week_smp | 13104 | 11.433384462759463 | 17.77151371915934 | 0.4063709279396204 |

These values are preserved for provenance only. They were produced without the selected exogenous variables and do not satisfy the current input requirements. Do not cite them as the project's Validation or Final Test performance.

## Figures

Existing figures, if any, belong to the archived calendar-only experiment and are not full-input model results.

Largest individual forecast errors across validation and final test (descriptive only; no causal attribution):

| split | forecast origin | target time (KST) | lead | actual SMP | predicted SMP | absolute error |
|---|---|---|---|---:|---:|---:|
| final_test | 2025-12-04 23:00:00 | 2025-12-07 13:00:00 | D+3 hour 62 | 0.00 | 158.81 | 158.81 |
| final_test | 2025-12-03 23:00:00 | 2025-12-06 12:00:00 | D+3 hour 61 | 6.28 | 164.25 | 157.97 |
| final_test | 2025-12-27 23:00:00 | 2025-12-30 12:00:00 | D+3 hour 61 | 43.06 | 199.08 | 156.02 |
| validation | 2025-03-08 23:00:00 | 2025-03-09 12:00:00 | D+1 hour 13 | 0.00 | 155.93 | 155.93 |
| validation | 2025-03-08 23:00:00 | 2025-03-09 14:00:00 | D+1 hour 15 | 0.00 | 153.86 | 153.86 |
| validation | 2025-03-08 23:00:00 | 2025-03-09 11:00:00 | D+1 hour 12 | 0.00 | 153.45 | 153.45 |
| validation | 2025-03-08 23:00:00 | 2025-03-09 13:00:00 | D+1 hour 14 | 0.00 | 152.91 | 152.91 |
| final_test | 2025-12-05 23:00:00 | 2025-12-07 13:00:00 | D+2 hour 38 | 0.00 | 151.34 | 151.34 |
| validation | 2025-03-08 23:00:00 | 2025-03-09 10:00:00 | D+1 hour 11 | 0.00 | 150.54 | 150.54 |
| final_test | 2025-11-23 23:00:00 | 2025-11-26 12:00:00 | D+3 hour 61 | 0.00 | 150.27 | 150.27 |

## Next step

Complete every field and missing period in the collection report, preserve source publication/issue timestamps, build the full panel, and finish full-input SARIMAX wiring before running `python -m src.cli backtest`. Calendar-only SARIMAX remains an incomplete exploratory experiment.
