# LightGBM 조건부 학습 및 평가 보고서

## 데이터 버전

- Dataset: `smp72h_common_v4_generation_v2_20261119`
- Group 5: `generation_20261119_v2`, SHA-256 `911281bfa5832147f083efa1f5bbcc504757563bd43aca68404ce9b9c1a53821`
- 공통 12그룹, Train 729 origins / 52,488시간, Validation 179 / 12,888시간, Final Test 182 / 13,104시간
- v3 대비 `evaluation_origins.csv`와 `evaluation_targets.csv`는 동일하고, Group 5를 제외한 348개 공통 feature 값은 행별로 동일합니다. Group 5가 8개 과거 연료 계열에서 새 원자료의 5개 계열로 바뀌어 총 입력 feature 수는 524개에서 458개로 달라졌습니다.

## Group 5 원자료 처리

원자료는 시간 구간 시작 시각을 사용합니다. 2023/2025 전체 연도는 제공된 5개 계열에 결측이 없고, 2024년에는 LNG·유연탄 2월 29일 24시간 결측, 무연탄·원자력·석탄합계 2월 29일 및 8월 각 768시간 결측이 있습니다. 모든 빈칸을 그대로 유지했습니다. 공통 cutoff는 origin−72h이며, 실제 구간 종료가 cutoff 이하인 값만 history feature에 들어갑니다. 학습 시 Train 데이터로 중앙값을 계산하고 missing indicator를 유지했습니다.

Group 5 원자료는 시장거래 송전단 발전량이며 지역 구분이 없습니다. 원자료 설명 및 세부 비교는 `group5_source/`에 있습니다.

## 모델 선택

Validation MAE 기준 세 후보 중 candidate 2를 골랐습니다. 선택 파라미터는 `num_leaves=31`, `max_depth=7`, `learning_rate=0.025`, `min_child_samples=80`, `reg_lambda=12`, `reg_alpha=0.1`, `feature_fraction=0.9`, `bagging_fraction=0.85`, `bagging_freq=1`입니다. 선택 iteration은 143입니다. Final Test 결과는 튜닝이나 모델 선택에 사용하지 않았습니다.

## 평가 결과 (원/kWh)

| Split | Model | MAE | RMSE | R² | Bias (pred−actual) |
|---|---|---:|---:|---:|---:|
| Train | LightGBM | 8.3366 | 12.1178 | 0.9239 | -0.0004 |
| Validation | LightGBM | 10.0290 | 13.9651 | 0.5478 | +0.6709 |
| Final Test | LightGBM | 20.3567 | 24.3100 | -0.1108 | +18.7640 |
| Final Test | 최근 일별 반복 | 13.0875 | 19.7381 | 0.2677 | +1.1096 |
| Final Test | 전주 같은 시간 반복 | 11.4334 | 17.7715 | 0.4064 | +1.2681 |

Final Test D+1/D+2/D+3 MAE는 20.2321 / 20.3159 / 20.5222입니다. LightGBM은 Final Test에서 기준선보다 나빴으며 개선율은 각각 -55.54%, -78.05%입니다. 결과를 성능 개선으로 해석하지 않습니다.

## 조건부 상태

Group 1은 사용자 지시에 따라 v3의 입력, 해석, 처리 방식을 그대로 유지했습니다. 그 입력의 행별 육지/제주 구분은 검증되지 않았으므로, 이 보고서는 육지 전용 입력이 독립 확인된 결과라고 주장하지 않습니다. 사용자는 결과를 팀이 볼 수 있도록 조건부 상태를 밝히고 업로드하라고 명시했습니다. 따라서 업로드 허용과 데이터 게이트 통과를 별도로 기록했습니다: `github_upload_ready=true`, `data_gate_passed=false`.

국내 LNG 대상월 입력은 합의된 조건부 과거 실험이고, 실측 공표시각에 대한 origin−72h 규칙은 보수적 가정입니다.
