# Final evaluation report

## 1. Research objective and period

Forecast land SMP in KRW/kWh at 23:00 KST from D+1 00:00 through D+3 23:00 (72 hours). Requested period: 2023-01-01 through 2025-12-31; train through 2024-12-31, validation 2025-01-01 through 2025-06-30, final test 2025-07-01 through 2025-12-31.

## 2–3. Data and preprocessing

See [data collection report](data_collection_report.md) and [feature selection report](feature_selection_report.md). Missing target SMP is never replaced with zero or model output. Post-origin actual demand, generation, weather, and SMP are not supplied as future exogenous variables.

Observed target coverage in processed file: 2023-01-01 00:00:00 through 2025-12-31 23:00:00. Total hourly grid rows: 26304; observed SMP: 26304; missing SMP: 0 (0.00%).

- 2023: 8760/8760 observed hourly SMP
- 2024: 8784/8784 observed hourly SMP
- 2025: 8760/8760 observed hourly SMP

## 4–6. Model and performance

Validation-selected SARIMAX is reported in `artifacts/selected_model.json`. Split metrics are below; D+1/D+2/D+3 details are in the metrics CSV.

| split | group | count | mae | rmse | r2 |
|---|---|---|---|---|---|
| validation | overall_sarimax | 12888 | 16.742690315048122 | 22.408133200576682 | -0.16420154515508822 |
| validation | previous_day_smp | 12888 | 14.31401225946617 | 21.718541922269107 | -0.09364947168141335 |
| validation | previous_week_smp | 12888 | 12.2210917132216 | 19.02490480073182 | 0.1608070995184563 |
| final_test | overall_sarimax | 13104 | 29.465641498175593 | 37.4365532463806 | -1.63425633350235 |
| final_test | previous_day_smp | 13104 | 12.846539224664223 | 19.664118460939168 | 0.2731993901592028 |
| final_test | previous_week_smp | 13104 | 11.433384462759463 | 17.77151371915934 | 0.4063709279396204 |

The selected candidate did not converge within the configured 25 optimizer iterations. SARIMAX also scored worse than the previous-day and previous-week persistence baselines on both validation and final test, so these results are preliminary and should not be presented as a successful forecast model. Detailed per-day and horizon metrics are saved in `outputs/metrics/metrics_by_split.csv`. Forecast rows remain distinct by `forecast_origin` and `target_timestamp` in `outputs/predictions/`.

## 7. Graphs

Monthly actual-versus-predicted graphs (one per month), six-month hourly overview graphs (interactive HTML and PNG), and error-analysis graphs are in `outputs/figures/`. Daily 72-hour comparison graphs were omitted at the user's request.

## 8. Large-error forecasts

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

## 9. Limitations and extensions

Current exogenous variables are restricted to deterministic calendar features because historical point-in-time vintages for operating forecasts/weather were not established. The train, validation, and final-test date windows are fixed. Add complete source data, preserve publication vintages, and rerun the same SARIMAX protocol before interpreting performance. XGBoost, LightGBM, LSTM, and web integration are outside this project's scope.
