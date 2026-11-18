# SMP LightGBM 72시간 예측 결과

이 폴더에는 Group 5를 팀 공유 `generation_20261119_v2`로 교체한 LightGBM 학습·평가 산출물이 있습니다. 다른 입력 그룹, 예측 일정, split, cutoff 규칙은 공통 v3와 동일하게 유지했습니다.

## 상태와 범위

- 공통 패널: `smp72h_common_v4_generation_v2_20261119`
- Train / Validation / Final Test: 52,488 / 12,888 / 13,104 target rows (각각 729 / 179 / 182 origins)
- 공통 평가 일정은 v3와 동일하며, Group 5 외 공통 입력 특성값은 전부 행별 동일함을 확인했습니다. v3와 달라진 특성 열은 Group 5에 한정됩니다.
- Group 1은 사용자 지시에 따라 기존 자료와 해석을 그대로 유지했습니다. EPSIS 자료에 행별 지역 키가 없어 이 결과는 조건부 연구 실행이며, 독립적으로 검증된 육지 전용 평가라고 표현하지 않습니다.
- 사용자가 결과를 팀에 공개하라고 명시해 조건부 상태를 밝힌 채 업로드했습니다. 데이터 게이트 자체는 `false`로 보존했습니다.

## Group 5 출처와 처리

입력은 [generation_20261119_v2](https://github.com/gemmanzu-a11y/smp-graduation-2026/tree/main/shared_datasets/kpx_generation/20261119_v2)의 `generation_2023_2025_hourly.csv`입니다. 해당 CSV의 SHA-256은 `911281bfa5832147f083efa1f5bbcc504757563bd43aca68404ce9b9c1a53821`입니다. 단위는 MWh, 시각은 각 시간 구간의 시작이며, cutoff에는 구간 종료가 포함되는 시간까지만 반영했습니다.

모델에는 LNG, 유연탄, 무연탄, 원자력, 유연탄+무연탄 합계의 5개 원시 계열에서 만든 과거시차/요약 특성을 넣었습니다. 원자료 누락은 0으로 채우지 않고 Train 중앙값과 결측 표시로 처리했습니다. 2023년 8,760시간과 2025년 8,760시간은 다섯 계열 모두 결측이 없습니다. 2024년에는 2월 29일 누락과 8월 무연탄·원자력 누락을 원자료 그대로 보존했습니다.

## 결과

| 구간 | LightGBM MAE | 최근 일별 반복 MAE | 전주 반복 MAE |
|---|---:|---:|---:|
| Validation | 10.029 | 14.347 | 12.221 |
| Final Test | 20.357 | 13.088 | 11.433 |

Final Test에서 LightGBM의 오차가 두 기준선보다 큽니다. Final Test MAE 변화율은 일별 기준선 대비 -55.54%, 주간 기준선 대비 -78.05%입니다. 최종 평가값을 좋게 보이도록 튜닝하지 않았습니다. Validation에서 후보 3개를 비교해 설정과 143 iteration을 선택한 뒤 Final Test는 한 번만 평가했습니다.

## 주요 파일

- `outputs/runs/smp72h_common_v4_generation_v2_20261119_lightgbm/`: 예측 CSV, 직렬화 모델, 후보 점수, 전체/상세 지표, feature importance, 그림, 실행 요약
- `reports/evaluation_report.md`: 결과 해석 및 한계
- `reports/input_gate.json`: 입력 게이트와 Group 5 provenance
- `reports/group5_source/`: 새 Group 5 README, manifest, 결측 구간
- `reports/dataset_metadata/`: 공통 데이터 manifest와 feature manifest
- `src/models/train_lightgbm_72h_common_v3.py`: 학습/평가 코드
- `inputs/group5/generation_2023_2025_hourly.csv`: 사용한 Group 5 통합 원자료 사본

원본 12그룹 패널(`forecast_inputs.csv`, 약 274 MB)은 이 폴더에 중복 저장하지 않았습니다. 공통 데이터 버전과 입력 해시는 `reports/dataset_metadata/dataset_manifest.json`에서 확인할 수 있습니다.
