# XGBoost 다음 날 SMP 24시간 학습 및 평가

## 목표와 데이터

- V5의 `smp`/`actual_smp` 레이블 정의를 따랐습니다. V5 `aligned_hourly_all_features_2023.csv`의 2023년 8,760개 값은 사용한 전체 SMP 파일과 전부 일치했습니다.
- 전체 label: 2023-01-01 00시–2025-12-31 23시, 26,304행. 2024–2025년 자료는 기존 프로젝트의 land SMP actual 계열이며 raw 파일과 수집 manifest의 검증 가능한 중복 구간이 일치합니다. Final Test 기간에는 관측값 0원이 8건 있으며 누락값으로 바꾸지 않았습니다.
- 다만 2024–2025 레이블에는 세부 가격상품 및 각 시간의 공표 빈티지가 별도 표기되어 있지 않습니다. 원본 열 이름과 V5 사용방식에 따라 land SMP target으로 통일했으며 이 제한은 모델 결과 해석에 남깁니다.
- 기존 SMP lag 및 rolling 값은 각 일별 origin(D-1 23:00 KST)까지로 고정했습니다. target-day 실제 수요·발전량·예비력·기상은 사용하지 않았습니다.

## 사용 입력

- SMP: 같은 시간 전일 SMP baseline, 같은 시간 전주 lag, origin 시점의 SMP 및 1h/2h/24h/168h lag, 24h/168h rolling mean·std, 1h/24h 변화.
- Calendar: target hour, weekday, month, cyclic transforms, weekend. 미래 시각에서 결정적으로 계산했습니다.
- Weather: 서울 ASOS station 108 temperature/wind at target timestamp minus 48 hours. 기상청 자료 공개 지연(전일 자료 다음 날 10시 이후)을 반영했습니다. 과거 원본에는 당시 수정 빈티지가 없어 사후 정정 여부는 확인할 수 없습니다. 학습 panel 26,112행 중 기온 41행·풍속 68행이 비어 있으며 0으로 채우지 않고 XGBoost의 missing 처리에 맡겼습니다.
- 제외: target-day 수요·발전량·예비력 및 공개 지연 미확인 lag, 불완전/빈티지 미확인 KPX mlfd, ASOS SI, 공휴일 지정 빈티지, 발표시각 없는 연료비/FRED. EWP 예보는 2025-07-18 이후에만 있어 train 구간에 관측이 없어 제외했고, KMA point forecast 빈티지는 확보되지 않았습니다.

## 14개 입력 그룹별 선택

| # | 입력 그룹 | 이번 모델 사용 | 판단 및 제외 사유 |
|---:|---|---|---|
| 1 | 실제 시간별 전력수요 | 제외 | 대상일 실측은 전일 23시 이후의 미래값이며, 과거 lag의 공개 지연과 육지/제주 범위 일치도 검증되지 않았습니다. |
| 2 | 하루전 전력수요 예측 `mlfd` | 제외 | 이력이 희소하고 2023년 누락 날짜의 API 요청이 429로 실패했습니다. 전일 23시 시점에 실제 공개된 빈티지인지도 확인되지 않았습니다. |
| 3 | 공휴일 `locdate/isHoliday` | 제외 | 임시·대체공휴일의 당시 지정 및 공개 시각이 보존되지 않았습니다. |
| 4 | 결정론적 달력 | 사용 | 예측일의 시각·요일·월·주말 여부와 주기 변환값을 사용했습니다. |
| 5 | 발전원별 발전량 | 제외 | 대상일 실측은 미래값이고, 이력과 과거 lag의 공개 시점이 불완전합니다. |
| 6 | 지역별 태양광·풍력 실적 | 제외 | 대상일 실측은 미래값이며, 지역·기술별 이력 범위가 불완전합니다. |
| 7 | 하루전 태양광 예측 `solgenKpx` | 제외 | 2025-07-18 이후 일부 날짜·지역·시간대만 제공되어 Train 기간을 채우지 못합니다. |
| 8 | 공급예비력·대응예비율 | 제외 | 대상일 실측을 쓸 수 없고, 과거 lag가 cutoff 전에 공개됐는지 검증되지 않았습니다. |
| 9 | 국내 LNG 연료비단가 | 제외 | 월별 값은 확보했지만 해당 월 값의 공표 시각을 확인할 수 없습니다. |
| 10 | WTI·Brent | 제외 | 관측·빈티지 날짜만 있고 전일 23시 이전 실제 공개 시각이 없습니다. |
| 11 | 국제 LNG·석탄 | 제외 | 월별 관측값은 있으나 실제 공개 시각을 확인할 수 없습니다. |
| 12 | KMA ASOS `TA/SI/WS` | 일부 사용 | target timestamp보다 48시간 이전의 서울 ASOS 기온·풍속을 사용했습니다. 일사량 `SI`는 제외했습니다. 과거 수정 빈티지는 확인할 수 없습니다. |
| 13 | KMA 단기예보 `TMP/WSD` | 제외 | 2023–2025 point/horizon 빈티지 자료가 없고 현재 point API 접근도 승인되지 않았습니다. |
| 14 | 동서발전 `icsrPred/temp/ws` | 제외 | #7과 같은 제한된 예보 이력으로 학습 구간을 채우지 못합니다. |

## 시간순 분할 및 성능

초기 192시간은 origin 기준 168시간 lag/rolling warm-up이 없어 제외했습니다.

| Split | 기간 | 행 수 | 모델 | MAE | RMSE | R² |
|---|---|---:|---|---:|---:|---:|
| validation | 2025-01-01–2025-06-30 | 4,344 | XGBoost | 9.1042 | 14.3134 | 0.5328 |
| validation | 2025-01-01–2025-06-30 | 4,344 | 전일 같은 시간 baseline | 11.9669 | 19.1252 | 0.1659 |
| validation | 2025-01-01–2025-06-30 | 4,344 | 전주 같은 시간 baseline | 12.2720 | 19.1613 | 0.1627 |
| final_test | 2025-07-01–2025-12-31 | 4,416 | XGBoost | 9.2412 | 13.4839 | 0.6580 |
| final_test | 2025-07-01–2025-12-31 | 4,416 | 전일 같은 시간 baseline | 10.7742 | 17.4477 | 0.4274 |
| final_test | 2025-07-01–2025-12-31 | 4,416 | 전주 같은 시간 baseline | 11.4352 | 17.7605 | 0.4066 |

Validation MAE로 세 XGBoost 설정을 선택하고, 선택한 tree 수로 train+validation에 재학습한 뒤 Final Test를 한 번 평가했습니다. 세부 설정은 `outputs/metrics/experiment_summary.json`에 저장했습니다.

## V5 비교와 제한

- V5 보존 결과 MAE 11.0938은 2023-10-01–2023-12-31의 92일 Final Test입니다. 이번 Final Test는 2025-07-01–2025-12-31이므로 직접 우열 비교는 하지 않습니다.
- 비교 가능한 내부 baseline은 각 split의 전일 동일시간 persistence입니다. 2026년 시연 및 실시간 API는 이번 실험에 포함하지 않았습니다.
- Final Test에 원본에서 관측 label로 표시된 0원 값 8건이 그대로 포함됐습니다. XGBoost bias(예측−실제)는 +2.43원/kWh이며, 세부 가격상품 및 0값 정의는 원본 메타데이터에서 확인되지 않았습니다.
- 입력 panel row-level cutoff 검사에서 미래 관측 사용은 0행입니다. 목표 가격의 세부 상품/공표 시점이 완전히 기록되지 않은 점은 후속 source metadata 확인 대상입니다.

## 생성 산출물

- `data/processed/day_ahead_supervised_panel_2023_2025.csv`: 공통 입력에서 구성한 feature/label panel 및 cutoff audit columns.
- `outputs/predictions/xgboost_day_ahead_predictions.csv`: validation 및 final-test 예측.
- `outputs/metrics/model_metrics.csv`, `hourly_metrics.csv`, `validation_candidate_metrics.csv`, `feature_importance.csv`, `experiment_summary.json`.
- `outputs/figures/xgboost_final_test_actual_vs_predicted.png`, `xgboost_final_test_example_week.png`, `xgboost_final_test_hourly_mae.png`.
- `models/xgboost_day_ahead_24h.json`: 재사용 가능한 XGBoost booster.

XGBoost feature importance 상위 10개: horizon (0.021), smp_same_hour_prev_day (0.033), smp_same_hour_prev_week (0.021), smp_daily_weekly_spread (0.226), origin_smp (0.014), origin_smp_lag_1h (0.018), origin_smp_lag_2h (0.020), origin_smp_lag_24h (0.033), origin_smp_lag_168h (0.019), origin_smp_roll24_mean (0.020)
