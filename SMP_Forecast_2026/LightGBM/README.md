# LightGBM 하루전 SMP 모델

전일 23:00 KST 시점의 기존 입력 패널로 다음 날 24시간 SMP를 예측했습니다. 기존 XGBoost와 동일한 27개 특성, 목표값, 시간 순서 분할, 학습·평가 행을 사용했습니다.

## 학습 조건

- Train: 2023-01-01–2024-12-31 (17,352행)
- Validation: 2025-01-01–2025-06-30 (4,344행); 후보 설정 선택에 사용
- Final Test: 2025-07-01–2025-12-31 (4,416행)
- 입력: 기존 SMP lag·rolling, 달력, 48시간 지연 서울 ASOS 기온·풍속 27개 변수
- 목표: 실제 SMP − 전일 같은 시간 SMP
- Random seed: 42
- 하이퍼파라미터와 후보별 Validation 결과: [experiment summary](outputs/metrics/experiment_summary.json), [후보 지표](outputs/metrics/validation_candidate_metrics.csv)
- 동일 입력 패널 및 타임스탬프 검증: [data equivalence report](outputs/metrics/data_equivalence_report.json)

## LightGBM 성능

| 구간 | MAE (원/kWh) | RMSE (원/kWh) | R² |
|---|---:|---:|---:|
| Validation | 9.0023 | 14.1814 | 0.5414 |
| Final Test | 9.2568 | 13.5457 | 0.6548 |

Final Test MAE는 전일 같은 시간 SMP 기준선보다 14.08% 낮았습니다.

- [평가 지표](outputs/metrics/model_metrics.csv)
- [시간대별 지표](outputs/metrics/hourly_metrics.csv)
- [월별 지표](outputs/metrics/monthly_metrics.csv)
- [특성 중요도 CSV](outputs/metrics/feature_importance.csv)
- [행 단위 예측 CSV](outputs/predictions/lightgbm_day_ahead_predictions.csv)
- [공통 데이터 출처와 해시](../Shared_Data/README.md)

## LightGBM 단독 그래프

- [실제 SMP와 LightGBM 예측 SMP](outputs/figures/lightgbm_final_test_actual_vs_predicted.png)
- [Final Test 예시 주간 (XGBoost 기준 그래프와 같은 기간·축 구성)](outputs/figures/lightgbm_final_test_example_week.png)
- [시간대별 MAE (XGBoost 기준 그래프와 같은 시간대·단위)](outputs/figures/lightgbm_final_test_hourly_mae.png)
- [입력 변수 중요도](outputs/figures/lightgbm_feature_importance.png)
- [실제값 대비 예측값 산점도](outputs/figures/lightgbm_actual_vs_predicted_scatter.png)
- [예측 오차 분포](outputs/figures/lightgbm_error_distribution.png)

직렬화된 LightGBM 가중치는 이 저장소에 포함하지 않았습니다. 이전 비교 자료는 [old/previous_comparison](../old/previous_comparison/)에 있습니다.
