# XGBoost 기준 모델

기존 `SMP_XGBOOST_2026` 실험의 학습 코드와 결과를 분리해 보관합니다.

## 학습 조건

- 예측 기준: 전일 23:00 KST, 다음 날 24시간 SMP
- Train: 2023-01-01–2024-12-31
- Validation: 2025-01-01–2025-06-30
- Final Test: 2025-07-01–2025-12-31
- 기존 SMP lag·rolling, 달력 특성, 48시간 지연 서울 ASOS 기온·풍속 등 27개 특성
- 전일 동일 시간 SMP 대비 잔차를 예측하고 기준값에 더하는 방식

## 결과

| 구간 | MAE (원/kWh) | RMSE (원/kWh) | R² |
|---|---:|---:|---:|
| Validation | 9.1042 | 14.3134 | 0.5328 |
| Final Test | 9.2412 | 13.4839 | 0.6580 |

- [학습 리포트](reports/xgboost_day_ahead_training_report.md)
- [평가 지표](outputs/metrics/model_metrics.csv)
- [행 단위 예측 CSV](outputs/predictions/xgboost_day_ahead_predictions.csv)
- [공통 데이터 출처·해시](../Shared_Data/README.md)

## 그래프

- [실제값과 예측값](outputs/figures/xgboost_final_test_actual_vs_predicted.png)
- [최종 테스트 예시 주간](outputs/figures/xgboost_final_test_example_week.png)
- [시간대별 MAE](outputs/figures/xgboost_final_test_hourly_mae.png)

직렬화된 XGBoost 가중치는 이 저장소에 포함하지 않았습니다.
