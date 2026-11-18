# Final evaluation: 72-hour land SMP XGBoost

## Forecast definition and split

- Origin: D 23:00, Asia/Seoul. Each issue predicts D+1 00:00–23:00, D+2 00:00–23:00, and D+3 00:00–23:00.
- Each issue has the same 72-hour observed cutoff: `forecast_origin - 72 hours`.
- Blocks crossing Train/Validation/Final Test boundaries were excluded. Target blocks cover Train 2023–2024, Validation 2025-01-01–2025-06-30, and Final Test 2025-07-01–2025-12-31.
- Selection was made using Validation MAE. The final model is fitted on Train only and Final Test was not used to retune it.

## Results (원/kWh)

| Split | Scored hours | MAE | RMSE | R² | Recent daily baseline MAE | Improvement | Weekly baseline MAE | Improvement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 12,888 | 10.48 | 14.28 | 0.53 | 14.35 | 26.99% | 12.22 | 14.28% |
| Final Test | 13,104 | 20.07 | 23.96 | -0.08 | 13.09 | -53.32% | 11.43 | -75.51% |

| Split | Forecast day | Hours | MAE | RMSE | Improvement vs recent daily | Improvement vs weekly |
|---|---:|---:|---:|---:|---:|---:|
| Final Test | D+1 | 4,368 | 20.00 | 23.86 | -37.89% | -75.55% |
| Final Test | D+2 | 4,368 | 19.98 | 23.87 | -50.34% | -74.68% |
| Final Test | D+3 | 4,368 | 20.22 | 24.14 | -76.29% | -76.29% |
| Validation | D+1 | 4,296 | 10.44 | 14.36 | 36.14% | 15.18% |
| Validation | D+2 | 4,296 | 10.46 | 14.22 | 27.91% | 14.07% |
| Validation | D+3 | 4,296 | 10.52 | 14.27 | 13.59% | 13.59% |

## Artifacts

- Full predictions: `outputs\predictions\72h\smp72h_common_v5_generation_v2_xgb_20261119\predictions_72h.csv`. Each row includes origin, target, horizon, forecast day, actual, predicted, error, baselines, split, and fuel policy.
- Metrics by origin, month, horizon and forecast day are saved next to the predictions.
- Figures: monthly D+1/D+2/D+3 MAE, per-horizon errors, six-month actual vs D+1 comparisons, final-test scatter, and target-hour error.
- Representative 72-hour plot for origin `2025-06-30 23:00:00` is saved as PNG and zoomable HTML.
- Reproduce from the workspace root with: `./.venv/Scripts/python.exe SMP_XGBOOST_2026/src/models/train_xgboost_72h.py --dataset-version smp72h_common_v5_generation_v2_xgb_20261119`

## Input groups

| Group | Data used |
|---:|---|
| 1 | EPSIS actual hourly load, mean of 12 five-minute snapshots, MW; 1,095 hourly gaps retained and imputed for model input. |
| 2 | Complete mainland day-ahead demand forecast history; publication time unverified, so only historical forecast values up to the shared cutoff are used. |
| 3 | KASI official holidays, separate from weekends. |
| 4 | Asia/Seoul calendar and hour/weekday/month cyclical features; checked against the supplied calendar panel. |
| 5 | KPX generation input `20261119_v2`: 26,304 hourly rows for 2023–2025 from annual 2023, API 2024, and annual 2025 sources. Includes LNG, bituminous coal, anthracite coal, nuclear, and combined coal generation (MWh); 2024 source gaps remain missing and flagged. |
| 6 | Official regional solar/wind sources; Jeju excluded where explicit; missing provincial sums remain missing; PPA/BTM coverage is not asserted. |
| 7 | Supply reserve MW and percent; 5-minute means require all 12 samples, 1,085 absent samples and 676 incomplete hours retained as missing. |
| 8 | Domestic LNG generation fuel cost in KRW/kWh matched by target month under the agreed conditional experiment. |
| 9 | FRED WTI and Brent values selected from the source vintage panel by D-2 as-of date. |
| 10 | FRED international LNG and coal monthly values selected from the source vintage panel by D-2 as-of date. |
| 11 | Seoul ASOS 108 temperature, wind and solar observations; night flag, observed zero and missing SI are separate. |
| 12 | KMA Seoul temperature and metro weather/precipitation, selecting the latest archived issue no later than the origin; issue time, target day and lead are retained. |

## Input and operating limitations

All 12 common groups and SMP history were kept. The D-2 FRED vintage panel, exact-target KMA forecast issue time and shared observed cutoff were checked. Group 2 is historical forecast input only because its original issue timestamps are missing. Group 12 uses the latest archived forecast issued by the origin; forecast lead and issue time are retained. D+2 and D+3 never use future actual SMP.

The Final Test model MAE is worse than both baselines; its improvement percentages are negative. The model was not changed after reviewing that result.

The domestic LNG feature uses the target month's value. Evaluation is therefore the **동월 LNG 단가 제공 조건부 과거 예측 실험**; publication by the forecast origin has not been verified, and this value is not an operationally confirmed input. The 2026-11 demo value is not part of the 2025 historical evaluation.

실제 수요, 발전량, 예비력 및 기상 관측자료의 개별 공표시각을 모두 확인할 수 없어, 본 연구에서는 예측 기준시각보다 72시간 이전의 실측 자료만 활용하는 보수적 지연 규칙을 가정하였다. 이 규칙은 각 자료의 실제 공개시각을 검증한 결과가 아니다.

SMP hourly publication time is not present in the source. The study conservatively uses only completed hourly SMP observations through the common cutoff. The EPSIS general page has no row-level geography field; its mainland scope is inferred from the separate Jeju page. Generation and renewable records may represent market-traded scope rather than all physical production. These are scope and availability limitations, not imputed zero values.

Plots are under `outputs/figures/72h/smp72h_common_v5_generation_v2_xgb_20261119/`. A user-selected origin can be plotted with `src/visualization/plot_72h_forecast.py`.
