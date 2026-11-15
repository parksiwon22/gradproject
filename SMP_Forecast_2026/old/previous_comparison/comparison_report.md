# LightGBM 하루전 SMP 예측 및 XGBoost 비교

## 데이터 동일성 및 예측 기준

- 저장된 XGBoost panel을 읽어 학습했고, 원본 공통 로더와 XGBoost panel builder로 다시 생성한 결과가 모든 열에서 정확히 일치했습니다. panel SHA-256: b48208aeaad0627d922ea885195eea37a14de7a505e7a5e97dba53b1b4a9f442.
- 27개 특성의 이름과 순서, split별 행 수, 평가 시각, 실제 SMP, 전일 baseline이 모두 일치했습니다. Final Test 공통 시각은 4,416개입니다.
- Train/Validation/Final Test 행: 17,352 / 4,344 / 4,416. 각각 2023-01-01~2024-12-31, 2025-01-01~2025-06-30, 2025-07-01~2025-12-31.
- 예측 시점은 전일 23시(KST)입니다. 모델 목표는 actual_smp - 전일 같은 시간 SMP 잔차이고, 예측 잔차를 전일 baseline에 더해 SMP를 복원합니다.
- 원래 XGBoost 특성 결측은 대치하지 않았습니다. 동일 float32 matrix를 사용하고 NaN은 두 트리 모델의 기본 결측 분기로 처리했습니다.
- 예측 cutoff 검사 실패 0행; cutoff 뒤 source timestamp 0행.

## LightGBM 선택 기준과 설정

- 사전 정의한 3개 트리 용량/규제 설정을 Train으로 학습하고 Validation의 복원 SMP MAE로 선택했습니다. Early stopping은 Validation 잔차 MAE 기준 100회 patience를 사용했습니다.
- Final Test 값은 파라미터 또는 반복 횟수 선택에 사용하지 않았으며, 최종 선택 반복 횟수로 Train+Validation을 재학습해 Final Test를 평가했습니다.
- random seed: 42; LightGBM 4.7.0; 최종 설정: {"max_depth": 8, "num_leaves": 31, "learning_rate": 0.02, "min_child_samples": 100, "feature_fraction": 0.85, "bagging_fraction": 0.8, "bagging_freq": 1, "reg_lambda": 16.0, "reg_alpha": 0.2, "n_estimators": 130}.

## 성능 비교

| 항목 | XGBoost | LightGBM | 비교 결과 |
|---|---:|---:|---|
| Validation MAE | 9.1042 | 9.0023 | LightGBM |
| Final Test MAE | 9.2412 | 9.2568 | XGBoost |
| Final Test RMSE | 13.4839 | 13.5457 | XGBoost |
| Final Test R² | 0.6580 | 0.6548 | XGBoost |
| 실험 총 시간 | 11.252초 | 7.192초 | 각 실행 스크립트 end-to-end 경과 시간 |
| Final Test 전일 baseline 대비 MAE 개선율 | 14.23% | 14.08% | 동일 baseline 기준 |

Final Test에서 MAE는 XGBoost가 낮고, RMSE는 XGBoost가 낮으며, R²는 XGBoost가 높습니다. MAE 차이는 0.0156 KRW/kWh입니다.
이 비교는 2025년 하반기 한 기간의 결과이며, 다른 시장 국면이나 가격 상품에 일반화한다고 해석하지 않습니다.

## 시간대별·월별 MAE

Final Test 시간대별 MAE (hour-ending 1–24):

| 시간대 | XGBoost MAE | LightGBM MAE |
|---:|---:|---:|
| 1 | 9.3026 | 9.0790 |
| 2 | 8.2174 | 7.9137 |
| 3 | 7.1244 | 6.8334 |
| 4 | 6.8449 | 6.5379 |
| 5 | 7.1544 | 6.8191 |
| 6 | 7.6776 | 7.3324 |
| 7 | 8.0717 | 7.7954 |
| 8 | 9.1911 | 9.1774 |
| 9 | 9.7095 | 10.0543 |
| 10 | 10.4371 | 10.7783 |
| 11 | 9.9713 | 10.1545 |
| 12 | 10.8600 | 11.1748 |
| 13 | 14.3088 | 14.6801 |
| 14 | 10.4867 | 10.8148 |
| 15 | 10.9866 | 11.3291 |
| 16 | 11.4523 | 11.6256 |
| 17 | 8.9230 | 9.1376 |
| 18 | 10.0951 | 10.2341 |
| 19 | 8.7092 | 8.6830 |
| 20 | 8.9875 | 8.8928 |
| 21 | 8.9718 | 8.9506 |
| 22 | 8.3001 | 8.2876 |
| 23 | 8.0759 | 8.0533 |
| 24 | 7.9297 | 7.8237 |

Final Test 월별 MAE:

| 월 | XGBoost MAE | LightGBM MAE |
|---|---:|---:|
| 2025-07 | 6.5816 | 6.6217 |
| 2025-08 | 7.1876 | 7.2673 |
| 2025-09 | 9.5763 | 9.5365 |
| 2025-10 | 11.7575 | 11.7371 |
| 2025-11 | 10.0890 | 10.2074 |
| 2025-12 | 10.2933 | 10.2104 |

## 데이터 취급 및 재현 조건

- 원본 및 supervised panel은 SMP_XGBOOST_2026 / SMP_SARIMAX_2026 경로에서 읽었습니다. 본 저장소에는 원본 데이터나 full panel을 복사하지 않았습니다.
- 파일 크기와 SHA-256, 로컬 경로는 Shared_Data/data_manifest.csv 및 LightGBM/outputs/metrics/data_equivalence_report.json에 기록했습니다. 공개 조건이 명확하지 않은 SMP·ASOS 파일은 업로드하지 않았습니다.
- 재실행하려면 SMP_XGBOOST_PROJECT와 SMP_SARIMAX_PROJECT 환경 변수를 해당 로컬 프로젝트 경로로 지정하세요. 키가 필요한 API 호출은 사용하지 않습니다.

## 결과 파일

- LightGBM 예측/지표/설정/모델: LightGBM/outputs/, LightGBM/models/.
- 동일 시각 XGBoost/LightGBM 비교 및 비교 그래프: Comparison/.
- 모든 8개 PNG 목록은 LightGBM README와 최상위 README에 있습니다.
