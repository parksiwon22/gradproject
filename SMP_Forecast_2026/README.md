# 2026 하루전 SMP 예측 모델 결과

2023–2025 시간별 자료로 다음 날 24시간 SMP를 예측한 XGBoost와 LightGBM 실험입니다. 두 모델은 같은 27개 입력 변수, 같은 목표값, 시간 순서 분할, 평가 타임스탬프를 사용했습니다.

## 빠른 결과

| 구간 | 모델 | MAE (원/kWh) | RMSE (원/kWh) | R² |
|---|---|---:|---:|---:|
| Validation | XGBoost | 9.1042 | 14.3134 | 0.5328 |
| Validation | LightGBM | 9.0023 | 14.1814 | 0.5414 |
| Final Test | XGBoost | 9.2412 | 13.4839 | 0.6580 |
| Final Test | LightGBM | 9.2568 | 13.5457 | 0.6548 |

Final Test에서는 XGBoost가 세 지표에서 근소하게 앞섰습니다. MAE 차이는 0.0156원/kWh로 작습니다. LightGBM은 전일 같은 시간 SMP 기준선보다 MAE를 14.08% 낮췄습니다.

## 현재 모델 폴더

- [XGBoost 결과·코드·그래프](XGBoost/README.md)
- [LightGBM 결과·코드·단독 그래프](LightGBM/README.md)
- [공통 데이터 출처와 해시 manifest](Shared_Data/README.md)

## 그래프 바로 보기

**XGBoost**

- [최종 테스트 Actual vs Predicted](XGBoost/outputs/figures/xgboost_final_test_actual_vs_predicted.png)
- [최종 테스트 예시 주간](XGBoost/outputs/figures/xgboost_final_test_example_week.png)
- [시간대별 MAE](XGBoost/outputs/figures/xgboost_final_test_hourly_mae.png)

**LightGBM 단독 결과**

- [최종 테스트 Actual vs Predicted](LightGBM/outputs/figures/lightgbm_final_test_actual_vs_predicted.png)
- [예시 주간](LightGBM/outputs/figures/lightgbm_final_test_example_week.png)
- [시간대별 MAE](LightGBM/outputs/figures/lightgbm_final_test_hourly_mae.png)
- [입력 변수 중요도](LightGBM/outputs/figures/lightgbm_feature_importance.png)
- [실제값 대비 예측값 산점도](LightGBM/outputs/figures/lightgbm_actual_vs_predicted_scatter.png)
- [예측 오차 분포](LightGBM/outputs/figures/lightgbm_error_distribution.png)

## 데이터와 분할

- Train: 2023-01-01–2024-12-31 (17,352행)
- Validation: 2025-01-01–2025-06-30 (4,344행)
- Final Test: 2025-07-01–2025-12-31 (4,416행)
- 입력 특성, 행, 타임스탬프, 목표값 및 해시의 동일성은 [검증 보고서](LightGBM/outputs/metrics/data_equivalence_report.json)에 기록했습니다.
- 원본·가공 데이터는 포함하지 않았습니다. 다운로드 출처, 로컬 경로, 파일 크기와 SHA-256은 Shared_Data에서 확인할 수 있습니다.

## 공개 파일 범위

이 공개 묶음에는 학습 코드, 가중치를 제외한 결과 파일, 행 단위 예측 CSV와 그래프가 포함됩니다. 원본 데이터 파일과 직렬화된 모델 가중치는 포함하지 않았습니다.

## 이전 산출물

앞선 모델 비교 리포트, 집계 CSV, 결합 예측 CSV와 비교 전용 그래프는 [old/previous_comparison](old/previous_comparison/)에 보관했습니다. 최신 모델별 결과는 위 모델 폴더에서 확인하세요.
