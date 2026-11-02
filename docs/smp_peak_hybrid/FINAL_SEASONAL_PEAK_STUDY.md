# 계절·시간 상호작용 피크 예측 연구 결과

## 범위와 방법

- 기존 D+1 V11 ExtraTrees 및 D+2/D+3 HGB-anchor 후보, 원본 HGB, 피크 가중 HGB, V12 LightGBM 계열을 기존 저장 fold 예측과 비교했습니다.
- 신규 후보는 기존 21개 V11/V17 피처를 그대로 사용했습니다. 월·시간·요일 변수를 추가하지 않았고, 학습 구간에서 고정 계절(DJF/MAM/JJA/SON) 또는 월별 24시간 피크 빈도 프로파일 KMeans(k=4)로 모델을 라우팅했습니다.
- 피크 분류기와 SMP≥150 가격 회귀기를 별도 학습했습니다. 예측에서는 분류 확률과 회귀값만으로 보정 여부를 판단했습니다. 실제 피크 라벨은 보정 입력으로 쓰지 않았습니다.
- 임계값·보정비율 선택은 각 fold의 calibration 구간에만 제한했습니다. MAE +0.10, RMSE +0.25, 비피크 MAE +0.10, G2V regret +2.0 이내를 지키며 피크 MAE를 줄이는 설정이 없으면 보정을 비활성화하도록 했습니다.
- 평가 fold는 fall2024, spring2025, fall2025의 validation 구간(각 horizon 275일)입니다. 반복 검토된 과거 구간이며 독립적인 미사용 테스트가 아닙니다. 원본 수신/공개 first_seen과 당시 모델 가용성은 검증되지 않아 모두 retrospective fold replay입니다.

| name | train_end | cal_start | cal_end | val_start | val_end |
| --- | --- | --- | --- | --- | --- |
| fall2024 | 2024-06-30 | 2024-07-01 | 2024-09-30 | 2024-10-01 | 2024-12-31 |
| spring2025 | 2024-12-31 | 2025-01-01 | 2025-03-31 | 2025-04-01 | 2025-06-30 |
| fall2025 | 2025-06-30 | 2025-07-01 | 2025-09-30 | 2025-10-01 | 2025-12-31 |

## 기존 달력 피처의 피크 분류 기여

아래 차이는 calibration 분기에서 계산한 평균입니다. 피처 제거는 남은 피처로 재학습했고, permutation은 학습된 전체 분류기에서 한 그룹을 함께 섞었습니다. `delta_average_precision`이 음수면 피처 제거/섞기가 AP를 낮췄다는 뜻으로 해당 그룹의 기여와 일치합니다.

| horizon_days | variant | removed_group | mean_delta_AP | mean_delta_brier | folds |
| --- | --- | --- | --- | --- | --- |
| 1 | drop_group_refit | hour | -0.0048 | 0.0001 | 3 |
| 1 | drop_group_refit | month | 0.0013 | -0.0001 | 3 |
| 1 | drop_group_refit | weekday | -0.0129 | 0.0044 | 3 |
| 1 | full_existing_features | NA | 0.0000 | 0.0000 | 3 |
| 1 | group_permutation | hour | -0.0741 | 0.0044 | 3 |
| 1 | group_permutation | month | -0.0002 | -0.0001 | 3 |
| 1 | group_permutation | weekday | -0.0393 | 0.0057 | 3 |
| 2 | drop_group_refit | hour | -0.0024 | 0.0001 | 3 |
| 2 | drop_group_refit | month | -0.0029 | -0.0011 | 3 |
| 2 | drop_group_refit | weekday | -0.0115 | 0.0039 | 3 |
| 2 | full_existing_features | NA | 0.0000 | 0.0000 | 3 |
| 2 | group_permutation | hour | -0.0718 | 0.0065 | 3 |
| 2 | group_permutation | month | 0.0007 | -0.0002 | 3 |
| 2 | group_permutation | weekday | -0.0362 | 0.0055 | 3 |
| 3 | drop_group_refit | hour | -0.0165 | -0.0027 | 3 |
| 3 | drop_group_refit | month | -0.0067 | -0.0030 | 3 |
| 3 | drop_group_refit | weekday | -0.0081 | 0.0013 | 3 |
| 3 | full_existing_features | NA | 0.0000 | 0.0000 | 3 |
| 3 | group_permutation | hour | -0.0760 | 0.0043 | 3 |
| 3 | group_permutation | month | 0.0009 | 0.0001 | 3 |
| 3 | group_permutation | weekday | -0.0272 | 0.0038 | 3 |

## 동일 fold pooled 비교

MAE/RMSE/피크 MAE는 원/kWh, 일별 최고 가격 오차와 피크시각 오차는 일별 평균(가격: 원/kWh, 시각: 시간), 오경보율은 실제 비피크 시간 중 예측 피크 비율, G2V regret은 EV 60kWh·SOC 30→80%·7kW·효율 0.90·24시간 선택 조건에서 일평균 실제 SMP 비용 대비 oracle 최소 비용 차이(원)입니다.

| horizon_days | model | days | MAE | RMSE | peak_MAE | normal_MAE | daily_max_abs_error | daily_peak_hour_error | false_alarm_rate | G2V_regret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | V11_ExtraTrees_fold_refit | 275 | 9.248 | 13.200 | 16.694 | 9.051 | 12.296 | 3.778 | 0.012 | 69.699 |
| 1 | data_cluster_seasonal_peak_gate | 275 | 9.639 | 13.598 | 14.759 | 9.504 | 12.573 | 3.778 | 0.019 | 69.535 |
| 1 | fixed_season_seasonal_peak_gate | 275 | 9.770 | 13.700 | 14.085 | 9.656 | 12.616 | 3.756 | 0.025 | 67.209 |
| 1 | global_peak_gate | 275 | 9.335 | 13.279 | 15.452 | 9.173 | 12.291 | 3.764 | 0.021 | 68.266 |
| 1 | global_seasonal_peak_gate | 275 | 9.565 | 13.500 | 14.542 | 9.433 | 12.264 | 3.669 | 0.018 | 69.699 |
| 1 | hgb | 275 | 9.780 | 13.732 | 16.288 | 9.608 | 11.626 | 3.811 | 0.011 | 66.402 |
| 1 | ml_anchor_blend | 275 | 9.626 | 13.528 | 15.936 | 9.459 | 11.273 | 3.731 | 0.010 | 59.183 |
| 1 | peak_weighted | 275 | 9.625 | 13.681 | 15.089 | 9.480 | 11.687 | 3.909 | 0.014 | 65.307 |
| 1 | persistence | 275 | 12.149 | 18.934 | 27.035 | 11.755 | 17.160 | 4.313 | 0.020 | 160.918 |
| 1 | weekday_median4 | 275 | 11.088 | 16.042 | 21.830 | 10.804 | 12.625 | 3.869 | 0.007 | 77.007 |
| 2 | data_cluster_seasonal_peak_gate | 275 | 10.962 | 15.174 | 16.433 | 10.818 | 12.546 | 3.971 | 0.015 | 63.081 |
| 2 | fixed_season_seasonal_peak_gate | 275 | 10.774 | 14.972 | 16.512 | 10.622 | 12.208 | 3.855 | 0.016 | 63.209 |
| 2 | global_peak_gate | 275 | 10.695 | 14.825 | 16.814 | 10.533 | 11.991 | 3.822 | 0.012 | 64.887 |
| 2 | global_seasonal_peak_gate | 275 | 10.802 | 14.981 | 16.385 | 10.654 | 12.028 | 3.927 | 0.012 | 64.887 |
| 2 | hgb | 275 | 10.640 | 14.460 | 18.072 | 10.444 | 12.474 | 3.945 | 0.012 | 73.774 |
| 2 | lightgbm_anchor_blend | 275 | 10.465 | 14.635 | 20.071 | 10.211 | 12.066 | 3.858 | 0.003 | 69.064 |
| 2 | lightgbm_v12 | 275 | 10.859 | 14.826 | 17.878 | 10.673 | 12.507 | 3.858 | 0.011 | 74.656 |
| 2 | lightgbm_v12_peak | 275 | 11.124 | 15.131 | 13.209 | 11.069 | 13.195 | 3.665 | 0.038 | 74.479 |
| 2 | ml_anchor_blend | 275 | 10.175 | 14.223 | 19.490 | 9.929 | 12.200 | 3.898 | 0.003 | 65.408 |
| 2 | peak_weighted | 275 | 10.628 | 14.552 | 15.742 | 10.492 | 12.561 | 3.702 | 0.019 | 73.859 |
| 2 | persistence | 275 | 16.017 | 22.986 | 30.414 | 15.637 | 20.520 | 4.524 | 0.021 | 200.621 |
| 2 | weekday_median4 | 275 | 11.088 | 16.042 | 21.830 | 10.804 | 12.625 | 3.869 | 0.007 | 77.007 |
| 3 | data_cluster_seasonal_peak_gate | 275 | 10.884 | 15.142 | 16.682 | 10.731 | 11.926 | 3.742 | 0.014 | 61.899 |
| 3 | fixed_season_seasonal_peak_gate | 275 | 11.002 | 15.317 | 16.040 | 10.869 | 12.052 | 3.698 | 0.020 | 63.175 |
| 3 | global_peak_gate | 275 | 10.936 | 15.208 | 16.021 | 10.802 | 11.969 | 3.705 | 0.018 | 61.368 |
| 3 | global_seasonal_peak_gate | 275 | 10.811 | 15.047 | 16.481 | 10.661 | 11.879 | 3.735 | 0.015 | 61.368 |
| 3 | hgb | 275 | 11.127 | 15.216 | 15.284 | 11.017 | 12.470 | 3.764 | 0.018 | 72.734 |
| 3 | lightgbm_anchor_blend | 275 | 10.396 | 14.565 | 17.184 | 10.217 | 11.992 | 3.731 | 0.011 | 62.842 |
| 3 | lightgbm_v12 | 275 | 11.154 | 15.288 | 15.420 | 11.041 | 12.859 | 3.778 | 0.019 | 68.437 |
| 3 | lightgbm_v12_peak | 275 | 11.612 | 15.747 | 12.019 | 11.601 | 13.129 | 3.858 | 0.045 | 79.729 |
| 3 | ml_anchor_blend | 275 | 10.429 | 14.621 | 18.342 | 10.220 | 11.910 | 3.811 | 0.006 | 63.175 |
| 3 | peak_weighted | 275 | 11.220 | 15.235 | 14.212 | 11.141 | 13.024 | 3.538 | 0.023 | 70.439 |
| 3 | persistence | 275 | 16.219 | 23.052 | 27.089 | 15.932 | 21.360 | 4.564 | 0.020 | 188.075 |
| 3 | weekday_median4 | 275 | 11.088 | 16.042 | 21.830 | 10.804 | 12.625 | 3.869 | 0.007 | 77.007 |

## 기준 모델 대비 fold 차이

`delta`는 후보−기준이며 양수는 악화입니다. 세 fold 가운데 유리한 fold 수를 함께 확인하고, 특정 한 fold의 개선을 일반화로 주장하지 않습니다.

### D+1

평균 fold 차이(후보−기준):

| model | folds | MAE_delta | RMSE_delta | peak_MAE_delta | normal_MAE_delta | daily_max_delta | peak_hour_delta | regret_delta | MAE_better_folds | peak_MAE_better_folds | regret_better_folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data_cluster_seasonal_peak_gate | 3 | 0.390 | 0.410 | -1.250 | 0.458 | 0.275 | 0.000 | -0.164 | 0 | 2 | 1 |
| fixed_season_seasonal_peak_gate | 3 | 0.520 | 0.511 | -1.695 | 0.612 | 0.319 | -0.022 | -2.481 | 0 | 2 | 1 |
| global_peak_gate | 3 | 0.087 | 0.081 | -1.170 | 0.123 | -0.006 | -0.014 | -1.428 | 0 | 3 | 1 |
| global_seasonal_peak_gate | 3 | 0.316 | 0.310 | -1.724 | 0.386 | -0.033 | -0.109 | 0.000 | 0 | 3 | 0 |
| hgb | 3 | 0.529 | 0.398 | -1.448 | 0.537 | -0.668 | 0.035 | -3.314 | 2 | 3 | 2 |
| ml_anchor_blend | 3 | 0.376 | 0.240 | -0.751 | 0.395 | -1.020 | -0.046 | -10.513 | 2 | 3 | 3 |
| peak_weighted | 3 | 0.375 | 0.401 | -2.101 | 0.420 | -0.608 | 0.131 | -4.378 | 1 | 3 | 2 |
| persistence | 3 | 2.900 | 5.764 | 7.942 | 2.710 | 4.856 | 0.536 | 91.022 | 0 | 0 | 0 |
| weekday_median4 | 3 | 1.839 | 2.780 | 6.982 | 1.752 | 0.333 | 0.093 | 7.310 | 0 | 0 | 0 |

### D+2

평균 fold 차이(후보−기준):

| model | folds | MAE_delta | RMSE_delta | peak_MAE_delta | normal_MAE_delta | daily_max_delta | peak_hour_delta | regret_delta | MAE_better_folds | peak_MAE_better_folds | regret_better_folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data_cluster_seasonal_peak_gate | 3 | 0.784 | 0.915 | -2.119 | 0.897 | 0.345 | 0.072 | -2.319 | 0 | 2 | 1 |
| fixed_season_seasonal_peak_gate | 3 | 0.596 | 0.729 | -1.914 | 0.701 | 0.008 | -0.043 | -2.192 | 0 | 3 | 1 |
| global_peak_gate | 3 | 0.518 | 0.595 | -1.863 | 0.611 | -0.209 | -0.077 | -0.520 | 0 | 3 | 1 |
| global_seasonal_peak_gate | 3 | 0.625 | 0.739 | -2.097 | 0.734 | -0.172 | 0.029 | -0.520 | 0 | 3 | 1 |
| hgb | 3 | 0.464 | 0.247 | -0.924 | 0.512 | 0.273 | 0.046 | 8.359 | 0 | 2 | 0 |
| lightgbm_anchor_blend | 3 | 0.289 | 0.394 | 0.565 | 0.278 | -0.134 | -0.040 | 3.657 | 0 | 0 | 0 |
| lightgbm_v12 | 3 | 0.681 | 0.486 | -2.230 | 0.728 | 0.304 | -0.040 | 9.254 | 0 | 3 | 0 |
| lightgbm_v12_peak | 3 | 0.948 | 0.898 | -7.068 | 1.139 | 0.988 | -0.234 | 9.079 | 0 | 3 | 0 |
| peak_weighted | 3 | 0.451 | 0.307 | -3.619 | 0.560 | 0.358 | -0.196 | 8.456 | 0 | 3 | 0 |
| persistence | 3 | 5.838 | 8.778 | 12.552 | 5.717 | 8.309 | 0.625 | 134.958 | 0 | 0 | 0 |
| weekday_median4 | 3 | 0.914 | 1.817 | 4.283 | 0.881 | 0.425 | -0.030 | 11.642 | 0 | 0 | 1 |

### D+3

평균 fold 차이(후보−기준):

| model | folds | MAE_delta | RMSE_delta | peak_MAE_delta | normal_MAE_delta | daily_max_delta | peak_hour_delta | regret_delta | MAE_better_folds | peak_MAE_better_folds | regret_better_folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data_cluster_seasonal_peak_gate | 3 | 0.453 | 0.530 | -1.209 | 0.516 | 0.016 | -0.069 | -1.272 | 0 | 2 | 1 |
| fixed_season_seasonal_peak_gate | 3 | 0.571 | 0.698 | -1.496 | 0.657 | 0.141 | -0.112 | 0.000 | 0 | 2 | 0 |
| global_peak_gate | 3 | 0.505 | 0.592 | -2.229 | 0.588 | 0.057 | -0.105 | -1.801 | 0 | 3 | 1 |
| global_seasonal_peak_gate | 3 | 0.381 | 0.433 | -1.785 | 0.446 | -0.033 | -0.076 | -1.801 | 0 | 3 | 1 |
| hgb | 3 | 0.695 | 0.408 | -4.543 | 0.777 | 0.557 | -0.048 | 9.506 | 1 | 3 | 1 |
| lightgbm_anchor_blend | 3 | -0.033 | -0.060 | -0.615 | -0.003 | 0.083 | -0.080 | -0.372 | 1 | 1 | 1 |
| lightgbm_v12 | 3 | 0.722 | 0.464 | -4.298 | 0.800 | 0.945 | -0.033 | 5.196 | 1 | 3 | 1 |
| lightgbm_v12_peak | 3 | 1.181 | 1.006 | -7.744 | 1.369 | 1.211 | 0.048 | 16.497 | 0 | 3 | 0 |
| peak_weighted | 3 | 0.789 | 0.469 | -5.973 | 0.904 | 1.108 | -0.273 | 7.261 | 0 | 3 | 0 |
| persistence | 3 | 5.790 | 8.568 | 9.295 | 5.728 | 9.433 | 0.753 | 124.764 | 0 | 0 | 0 |
| weekday_median4 | 3 | 0.661 | 1.493 | 5.438 | 0.597 | 0.716 | 0.058 | 13.876 | 1 | 0 | 0 |

## 학습 구간에서 본 계절별 시간대

시간대 순위와 표본 수는 각 fold 학습 구간에서만 계산했습니다. 날짜가 겹치는 fold/horizon 행은 독립 표본처럼 합산하지 않았습니다. 월별 표는 D+3 fall2025 fold의 2025-06-30 cutoff 학습 구간만 집계했습니다.

### 월별 발생률 및 피크 가격

| month | samples | peak_hours | peak_rate | mean_peak_smp | mean_peak_uplift_over_daily | highest_peak_rate_hour | highest_peak_rate | highest_peak_hour_sample_count | training_cutoff | horizon_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1992 | 813 | 0.408 | 208.432 | 16.387 | 9 | 0.687 | 83 | 2025-06-30 | 3 |
| 2 | 2040 | 831 | 0.407 | 234.491 | 8.489 | 10 | 0.682 | 85 | 2025-06-30 | 3 |
| 3 | 2232 | 898 | 0.402 | 207.999 | 18.874 | 23 | 0.570 | 93 | 2025-06-30 | 3 |
| 4 | 2160 | 673 | 0.312 | 170.151 | 4.211 | 10 | 0.467 | 90 | 2025-06-30 | 3 |
| 5 | 2232 | 345 | 0.155 | 165.393 | 20.182 | 19 | 0.301 | 93 | 2025-06-30 | 3 |
| 6 | 2160 | 466 | 0.216 | 159.786 | 6.668 | 14 | 0.433 | 90 | 2025-06-30 | 3 |
| 7 | 1488 | 550 | 0.370 | 165.758 | 4.607 | 14 | 0.677 | 62 | 2025-06-30 | 3 |
| 8 | 1488 | 722 | 0.485 | 158.847 | 2.377 | 14 | 0.887 | 62 | 2025-06-30 | 3 |
| 9 | 1440 | 415 | 0.288 | 157.786 | 5.204 | 13 | 0.667 | 60 | 2025-06-30 | 3 |
| 10 | 1488 | 356 | 0.239 | 160.678 | 17.992 | 16 | 0.435 | 62 | 2025-06-30 | 3 |
| 11 | 1440 | 62 | 0.043 | 158.317 | 25.912 | 9 | 0.333 | 60 | 2025-06-30 | 3 |
| 12 | 1488 | 248 | 0.167 | 158.150 | 19.628 | 9 | 0.532 | 62 | 2025-06-30 | 3 |

### 고정 계절별 시간대 상위 3개

| horizon_days | fold | season | top3_hours | top3_peak_rate | top3_peak_mean_smp | top3_peak_uplift | min_hour_samples | max_hour_samples | data_partition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | fall2025 | autumn | 9,10,15 | 0.390,0.379,0.374 | 161.0,159.7,163.3 | 19.3,14.4,11.3 | 182 | 182 | training only |
| 3 | fall2025 | spring | 19,20,10 | 0.402,0.399,0.391 | 188.6,185.9,186.5 | 3.4,2.7,24.2 | 276 | 276 | training only |
| 3 | fall2025 | summer | 14,15,13 | 0.636,0.598,0.570 | 165.9,166.1,164.4 | 3.9,3.0,3.7 | 214 | 214 | training only |
| 3 | fall2025 | winter | 9,10,11 | 0.626,0.622,0.496 | 203.6,195.5,201.1 | 20.7,18.2,20.2 | 230 | 230 | training only |

### 데이터 기반 월 군집

| month | data_cluster | train_end | selection_source |
| --- | --- | --- | --- |
| 1 | 1 | 2025-06-30 | training monthly-hour peak frequency only |
| 2 | 1 | 2025-06-30 | training monthly-hour peak frequency only |
| 3 | 1 | 2025-06-30 | training monthly-hour peak frequency only |
| 4 | 1 | 2025-06-30 | training monthly-hour peak frequency only |
| 5 | 0 | 2025-06-30 | training monthly-hour peak frequency only |
| 6 | 0 | 2025-06-30 | training monthly-hour peak frequency only |
| 7 | 3 | 2025-06-30 | training monthly-hour peak frequency only |
| 8 | 3 | 2025-06-30 | training monthly-hour peak frequency only |
| 9 | 0 | 2025-06-30 | training monthly-hour peak frequency only |
| 10 | 0 | 2025-06-30 | training monthly-hour peak frequency only |
| 11 | 2 | 2025-06-30 | training monthly-hour peak frequency only |
| 12 | 0 | 2025-06-30 | training monthly-hour peak frequency only |

## 결론

- 계절 분리/데이터 기반 월 군집 게이트는 일부 fold에서 피크 MAE와 regret을 줄였지만, pooled 기준으로 전체 MAE가 높아져 현 단계의 최종 모델 교체 근거가 되지 않습니다.
- 피크 가중 LightGBM은 피크 MAE를 크게 낮추는 대신 전체 MAE, 급등 오경보 및/또는 regret을 악화시켰습니다. 이는 피크 전용 최적화의 trade-off로 유지해야 합니다.
- D+3 LightGBM-anchor는 pooled MAE가 10.396으로 현재 HGB-anchor 10.429보다 0.033 낮고 regret도 0.33 낮지만, MAE 개선은 3개 fold 중 1개에서만 나타나 승격 근거가 약합니다.
- 따라서 D+1 V11 ExtraTrees, D+2/D+3 기존 HGB-anchor를 현 최종 연구 후보로 유지합니다. 이번 결과는 별도 연구 산출물이며 V20·공식 발급 gate·운영 모델에 연결하지 않았습니다.
- 표본의 SMP≥150 발생률이 높고 계절·시각 패턴의 변화가 있어, 임계값 분류와 가격 회귀의 성능은 레짐 변화 및 가격 기준 공개시각 검증에 민감합니다. 실운영 시점 vintage 검증, 더 많은 비중복 기간 검증이 남아 있습니다.

## 재현 명령

저장된 기존 비교 결과를 바탕으로 같은 순서로 다시 계산할 때 repository root에서 실행합니다.

```powershell
.venv-smp-research\Scripts\python.exe -m experiments.smp_peak_hybrid.run_peak_hybrid_study
.venv-smp-research\Scripts\python.exe -m experiments.smp_peak_hybrid.run_seasonal_peak_interactions
.venv-smp-research\Scripts\python.exe -m experiments.smp_peak_hybrid.finalize_seasonal_peak_study
```
