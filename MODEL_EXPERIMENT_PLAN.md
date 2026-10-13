# V6 모델 비교 및 검증 계획

## 1. 고정 기준

V5는 63개 특성과 horizon을 사용한 pooled direct residual HGB이다. 24개 별도 모델이나 재귀 예측이 아니다. 전일 동일시간 가격에 잔차를 더하고 일/주 프로파일과 혼합한다. 분류기는 직전 실제 시간보다 가격이 15원/kWh 이상 오른 사건을 학습한다. 미래 실제값은 라벨과 평가에만 사용한다.

첨부 기준은 train 235일, validation 30일(2023-09), test 92일(2023-10~12), 각각 5640/720/2208행이다. 저장 테스트 MAE 11.093786, daily baseline 13.575802, R² 0.642919다. 급등 F1은 validation 0.435897, test 0.277512이며 test recall 0.189542다. 가격 >=170은 49시간이고 모두 과소예측되었다. 상세 재현 이슈는 PROJECT_AUDIT.md를 먼저 읽는다.

원본 2023-10~12 결과가 이미 공개적으로 검토되었으므로 V6의 새로운 미열람 독립 테스트로 간주하지 않는다. 재현/회귀검증 데이터로만 쓴다.

## 2. 데이터와 분할

데이터 기간은 권한과 품질이 확보된 연속 3년 이상을 목표로 하며 2021~2025 및 2026 완료분 확보 가능성을 조사한다. 실제 확보 전 수집 완료라고 쓰지 않는다. 육지를 기본 후보로 하되 ZIP SMP의 시장구분은 원출처에서 확인한다. 제주를 혼합하지 않는다. 개정 제도와 가격 상한, 데이터 정의 변경은 별도 구간으로 기록한다.

확보 시 가장 최근의 완료된 3개월을 untouched test로 동결하고 날짜 목록과 해시를 기록한다. 이전 구간에서 최소 4개 월별 expanding-window 검증 fold를 만든다. 각 fold는 fit / chronological early-stop / selection 구간을 시간순으로 분리한다. 학습 라벨의 interval_end가 해당 fold의 모델 fit cutoff 이후인 행을 제거한다. 날짜별 24행은 같이 움직이며 셔플 분할하지 않는다. 공통 warm-up 및 결측 규칙으로 후보 간 같은 날을 평가한다. 탈락한 날도 사유와 개수를 보고한다.

스케일링, 결측 대체 통계, 변수선택, 가격 분위수, peak 가중치, 확률 보정, 혼합계수와 경고 임계값은 각 학습/검증 구간 안에서만 정한다. HGB early_stopping='auto'는 큰 데이터에서 내부 무작위 분할을 켤 수 있으므로 V6에서는 명시적 시간순 early-stop 데이터 또는 early_stopping=False와 검증 iteration 선택을 사용한다. V5 설정은 바꾸지 않는다.

## 3. 공정한 모델 비교

| 후보 | 최초 비교 설정 | 추가 실험 | 제한 |
| --- | --- | --- | --- |
| V5 동결 | 원본 HGB 및 프로파일 정책 | 원환경 재현 | 저장 기준과 재실행 기준 구분 |
| HGB | squared_error, absolute_error, max_iter 200/400, leaf 15/31 | quantile 0.1/0.5/0.9 | 시간순 early-stop |
| LightGBM | regression 및 regression_l1, leaves 15/31, lr .03/.05 | quantile, peak 가중치 | 동일 feature와 split |
| CatBoost | RMSE 및 MAE, depth 4/6, lr .03/.05 | Quantile, 캘린더 범주형 별도 ablation | 범주 처리 이득은 추가 실험으로 표시 |
| XGBoost | reg:squarederror 및 reg:absoluteerror, hist, depth 3/6 | reg:quantileerror | 지원 버전 확인 후 환경 고정 |
| 단순 기준선 | 전일, 전주, 검증 혼합 | 공식 가격 이용 가능 기준선 | 공식 값은 예측 MAE 경쟁과 구분 |

단계 A는 모델 family당 12개 이하 사전 정의 조합, 같은 63개 특성과 잔차 타깃, CPU와 seed=42를 사용한다. family별 탐색시간 상한을 초기 30분으로 두고 실행 횟수와 중단 사유를 기록한다. 단계 B는 상위 2 family에서 seed 7/42/2026으로 안정성을 검증한다. 단계 C에서 직접가격 타깃 대 잔차, pooled horizon 대 시간군 모델을 비교한다. 한 번에 하나의 변경만 도입한다. optional 모델 패키지는 별도 lock으로 두고 미설치 family는 실패/미실행으로 명시한다.

## 4. 피크 과소예측 실험

희소 피크, 평균으로의 수축, 프로파일 혼합, 누락된 계통/기상 정보, 계절과 제도 변화가 원인 후보다. 현재 증거는 >=170 49시간 모두 과소예측과 test 최고 예측 176.52 대 실제 218.25다. 원인이 확정되었다고 쓰지 않는다.

E0: 피크 발생일별 bias, 오차, 입력 결측, 시간군 및 모델 간 오차를 진단한다. E1: train 가격 상위 10%/5% 샘플 가중치 1/2/4를 비교한다. E2: absolute_error와 squared_error, q=.5/.8/.9를 비교한다. 상위 분위수는 위험용 출력이며 점예측을 무조건 대체하지 않는다. E3: train-only threshold의 peak classifier와 conditional regressor를 결합하고 검증자료에서 확률을 보정한다. E4: lag 48/72/336, 24/72시간 통계, hour별 프로파일 변동, 예보 net load 후보를 추가한다. E5: 당시 예보가 확보된 공통기간에서 예보 미사용/사용을 비교한다. 가중치로 평균 오차와 허위 경보가 늘어나는지도 보고한다.

고가(>=150,>=170), 급등(delta>=15,>=20), 일 최고가격 오차 및 최고시각 오차는 별도 지표다. V5 분류기의 class balancing 점수는 보정된 사건 발생확률이라고 가정하지 않는다. PR-AUC, Brier score, reliability curve와 검증 calibration을 추가하고 임계값은 오경보 비용을 반영해 정한다. UI 라벨의 고정 임계값과 모델의 검증 임계값을 정책 파일 한 곳에서 관리한다.

## 5. 지표와 선정 규칙

모든 후보의 MAE, RMSE, R², MAPE(0 근처 취약성 표시), signed bias, horizon별/계절별 지표를 남긴다. peak conditional MAE와 bias, 과소예측 비율, ramp precision/recall/F1/PR-AUC, 예측구간 coverage/width/pinball loss를 함께 보고한다.

충전은 2/3/4/5/6/8시간에 대해 실제 가격으로 계산한 regret, exact match, overlap을 보고한다. V5 regret는 고정 1kW, 1시간 슬롯 가정의 가격 합 차이며 실제 사용자의 원화 절감액이 아니다. V6에서는 가용시간, 효율, 부분시간과 실제 요금으로 총 구매비를 비교한다. 추천에 실제 미래값을 쓰지 않고 실제값은 oracle 평가에만 쓴다.

사전 기본 선정 규칙: 검증 fold 평균 MAE가 현실적 V5 대비 2% 이상 낮고, peak MAE와 충전 regret가 각 5% 초과 악화하지 않으면 채택 후보가 된다. 이 수치는 프로젝트 기본 제안이며 승인 가능한 기술 선택이다. 탈락 시 V5 유지, peak 특화 모델은 위험 레이어로만 검토한다. 일 단위 paired block bootstrap(7일 블록, 2000회)로 MAE/regret 차이 95% CI를 보고하고 계절별 안정성도 검토한다. 이 통계만으로 보장된 우월성을 주장하지 않는다.

최종 모델과 모든 정책을 validation에서 동결한 뒤 test를 한 번 평가한다. 테스트를 본 뒤 재튜닝하면 그 결과는 탐색으로 재분류하고 새로운 테스트기간을 마련한다. 공식 가격이 이미 확보된 23시 서비스의 효용은 별도의 서비스 KPI로 평가한다.

## 6. 재현 산출물

각 run은 source SHA, 데이터 hash 및 발표시각 정책, 환경 lock, split 날짜, seed, 실제 parameters, feature 순서, 예측 CSV, 모델, calibration, 정책, 실패로그를 보존한다. 원본 baseline 폴더와 outputs는 자동 overwrite 금지다. joblib은 신뢰할 수 있는 내부 생성물만 로드한다.

## 공식 근거

- HGB와 손실/early stopping: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
- LightGBM objectives: https://lightgbm.readthedocs.io/en/stable/Parameters.html
- CatBoost regression losses: https://catboost.ai/docs/en/concepts/loss-functions-regression
- XGBoost objectives: https://xgboost.readthedocs.io/en/stable/parameter.html

각 라이브러리의 지원 손실을 확인한 비교 제안이다. 이 프로젝트에서 우월성이 확인된 모델 목록이 아니다.
