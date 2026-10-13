# 전력예보 V5 실파일 감사와 V6 준비

작성: 2026-10-13. GitHub 비교 기준 main commit `b2d4f642bf9132d79590402142aa7b6d389e88c7`. 원격 저장소: https://github.com/parksiwon22/gradproject . 작업 브랜치: `work/v5-audit-v6-prep-20261013`.

## 1. 증거 범위와 기준

현재 대화에 전달된 V5 ZIP, 발표 PPTX, 대본 PDF, 1차 결과보고서 HWPX, 진행현황 DOCX와 공개링크 TXT를 실제 로컬 경로에서 읽었다. 문서들은 XML/텍스트 추출로 내용과 버전 맥락을 확인했다. 원본 문서의 시각 레이아웃 검토/수정은 수행하지 않았다. ZIP 안의 추가 PDF 3개는 파일 목록/해시만 조사했고 개발 기준 자료로 사용하지 않았다. 공개링크 TXT는 연결 주소 확인만 하며 실서비스 변경/배포는 하지 않았다. 과거 ChatGPT 프로젝트 대화 전체나 별도 V6 실험본에는 접근했다고 가정하지 않았다.

GitHub plugin으로 저장소 권한과 재귀 트리를 확인하고 main의 16개 blob 전부를 base64로 가져와 비교했다. 저장소는 public이며 선택 계정은 소유자 parksiwon22다. CLI clone 네트워크에 의존하지 않고 connector로 현재 bytes를 확보했다. 원본과 정규화 text, JSON 의미를 구분했다. sha256 전수 목록은 `benchmarks/v5/source_manifest.json`, 비교 목록은 `comparison_manifest.json`에 있다.

최신 기준은 ZIP의 `01_model_v5` 코드/설정/저장 연구 결과와 동일 내용의 V5 웹이다. LightGBM 단어만으로 구버전이라고 판단하지 않았다. 보고서, 발표, 진행현황도 HGB와 profile blend를 설명한다. 단, 이름의 V5 및 README 주장만으로 원결과의 완전 재현을 인증하지 않는다.

## 2. 웹과 GitHub 전수 비교

| main 파일 | 실제 역할/비교 결과 | 통합 처리 |
| --- | --- | --- |
| app.py | ZIP 02_web/app.py와 text 동일. 혼합 CRLF/LF 때문에 bytes는 다름. V5 metrics와 outputs CSV를 읽음 | main bytes 그대로 보존 |
| requirements.txt | ZIP과 bytes 동일. Streamlit 1.58.0, pandas 범위만 있음 | 기존 웹 요건 보존, 모델 요건 별도 추가 |
| services/v5_web_adapter.py | V5 CSV 변환 로직이나 Windows 절대경로가 남음. ZIP은 상대경로 수정본 | portable 경로/환경변수 및 정책/측정값 로딩 개선 |
| outputs/next_day_smp_forecast_tuned.csv | 2023-10-01 V5 24행. ZIP과 BOM/줄바꿈 정규화 text 동일 | 변경 안 함 |
| outputs/recommended_3h_charging_window_tuned.csv | 같은 날 03~06시 3행. ZIP과 정규화 text 동일 | 변경 안 함 |
| outputs/v5_metrics.json | ZIP과 JSON 내용 동일. 마지막 newline만 다름 | 변경 안 함 |
| README.md | 실행 설명은 유효하나 metrics 경로가 LightGBM JSON이고 현 모델 입력 설명이 낡음 | V5/개발/제약 기준으로 수정 |
| lightgbm_tuned_metrics.json | MAE 6.555, train/test 17352/8784. ZIP legacy JSON과 의미 동일, V5 분할 아님 | 보존, V5 성능으로 사용 금지 |
| next_day_smp_forecast_tuned.csv | root의 2025-01-01 초기 예측, V5 outputs와 다름 | 보존, 현재 app은 안 읽음 |
| recommended_3h_charging_window_tuned.csv | root의 2025-01-01 초기 추천 | 보존, app은 안 읽음 |
| outputs/lightgbm_tuned_metrics.json | 위 legacy metrics 복제 | 보존, app은 v5_metrics를 읽음 |
| outputs/archive_lightgbm_demo/lightgbm_tuned_metrics.json | legacy 복제 | 보존, 실험 기준 제외 |
| outputs/archive_lightgbm_demo/next_day_smp_forecast_tuned.csv | root 초기 예측과 bytes 동일 | 보존 |
| outputs/archive_lightgbm_demo/recommended_3h_charging_window_tuned.csv | root 초기 추천과 bytes 동일 | 보존 |
| outputs/.gitkeep | 빈 디렉터리 유지 파일 | 보존 |
| outputs/archive_lightgbm_demo/.gitkeep | 동일 목적 | 보존 |

main은 웹 표시용 V5 통합본이다. 'GitHub app.py가 옛 LightGBM 앱이다'라는 판단은 실제 코드와 맞지 않는다. 학습 전체가 GitHub에 있는 상태도 아니다.

## 3. ZIP 파일별 역할과 누락

| ZIP 경로(pkg/ 이후) | 원 main 여부 | 실제 역할 및 처리 |
| --- | --- | --- |
| 01_model_v5/smp_predictor_v5.py | 없음 | HGB 회귀/급등분류, 특성/분할/평가/추론/저장. exact 원본과 portable 실행본 추가 |
| 01_model_v5/models_config/day_ahead_feature_metadata.json | 없음 | 63개 순서 및 전처리 메타데이터. 복사 |
| 01_model_v5/models_config/day_ahead_policy.json | 없음 | 모델 선택, 혼합계수 .69/.31, ramp threshold .45, 검증 탐색. 복사 |
| 01_model_v5/data/aligned_hourly_all_features_2023.csv | 없음 | 8760행 69열 연구 입력. 비공개 로컬 연결, Git 제외 |
| 01_model_v5/results/README_V5.md | 없음 | 저장 결과 설명. benchmarks/v5에 보존 |
| 01_model_v5/results/v5_model_comparison.csv | 없음 | train/validation/test 모델 성능. aggregate benchmark 보존 |
| 01_model_v5/results/v5_per_horizon_metrics.csv | 없음 | 시간별 오차 집계. benchmark 보존 |
| 01_model_v5/results/v5_risk_metrics.csv | 없음 | 고가/급등 오차와 분류 성능 집계. benchmark 보존 |
| 01_model_v5/results/v5_charging_window_backtest.csv | 없음 | 기간별 regret/match/overlap 집계. benchmark 보존 |
| 01_model_v5/results/v5_training_summary.json | 없음 | 실행 정보/63개 순서/결과 및 로컬 Windows 명령. 원본은 첨부로 보존, 공개 감사는 필요한 수치만 재구성 |
| 01_model_v5/results/v5_day_ahead_predictions.csv | 없음 | 92일 2208행 실제값/후보예측/위험. private_reference 연결, Git 제외 |
| 02_web/app.py | 있음 | 동일 내용 V5 화면. bytes 원격 보존 |
| 02_web/requirements.txt | 있음 | 동일 bytes |
| 02_web/services/v5_web_adapter.py | 경로 차이 | ZIP의 상대경로도 현재 평탄화 repo에서는 그대로 맞지 않아 root/runs 기준으로 정리 |
| 02_web/outputs/next_day_smp_forecast_tuned.csv | 있음 | V5 표시 데모, 의미 동일 |
| 02_web/outputs/recommended_3h_charging_window_tuned.csv | 있음 | V5 표시 데모, 의미 동일 |
| 02_web/outputs/v5_metrics.json | 있음 | 의미 동일 |
| 99_legacy_ignore/smp_simple_LSTM_old.py | 없음 | 초기 LSTM. 비교 자료일 뿐 V5 실행에 필요하지 않아 통합 제외 |
| 99_legacy_ignore/lightgbm_tuned_metrics.json | 의미 동일 JSON 있음 | 초기 모델 다른 검증 조건. V5 평가 제외 |
| 00_HANDOFF_README.md | 없음 | 인수인계 배경. 새 문서로 사실 정정 |
| 03_docs_pdf/ PDF 3개 | 없음 | 원보고서/발표/기획자료. 내용/공개권한 확인 없이 업로드하지 않음 |

ZIP에도 원래 학습된 `day_ahead_hgb_residual_model.joblib`, `day_ahead_ramp_classifier.joblib`가 없다. 생성 PNG 3개와 원 학습 환경 lock도 없다. summary의 model_artifacts_written=true는 당시 실행 주장이지 첨부에 joblib이 있다는 증거가 아니다. 원 데이터 수집/정렬 코드와 당시 예보 vintage도 없다. 이 누락은 새 작업으로 해결해야 한다.

## 4. 저장된 V5 성능 검산

| 항목 | validation | final test |
| --- | --- | --- |
| 기간/표본 | 2023-09, 30일 720행 | 2023-10~12, 92일 2208행 |
| selected MAE | 7.393895 | 11.093786 |
| daily baseline MAE | 8.664014 | 13.575802 |
| weekly baseline MAE | 9.837042 | 12.163909 |
| selected R² | 0.716057 | 0.642919 |
| ramp precision | 0.500000 | 0.517857 |
| ramp recall | 0.386364 | 0.189542 |
| ramp F1 | 0.435897 | 0.277512 |

63개 pooled direct 특성, train 235일 5640행이다. 최종 예측은 `0.31*(0.69*daily+0.31*weekly)+0.69*(daily+HGB_residual)`이다. 일/주 alpha .69와 최종 profile alpha .31를 혼동하지 않는다. V5 uses HGB, not LightGBM, and no recursive predicted-SMP feedback.

고가 >=170의 테스트 49행은 모두 selected 예측이 실제보다 낮고 conditional bias/MAE는 -25.861123/25.861123이다. 전체 test 최고 실제 218.25, 최고 selected 예측 176.520710이다. 전체 bias -0.0251이 작아도 피크 과소예측이 없는 것은 아니다. 저장 CSV에서 직접 계산했다.

2/3/4/5/6/8시간에 대한 평균 regret 집계의 평균이 2.4643이다. 단위는 시간별 SMP 합 차(고정 1kW와 1시간 가정일 때 비용으로 환산 가능)이며 사용자 요금의 실제 절감액이 아니다. ZIP handoff의 'F1 약 .44'는 validation이고 final test .278과 구분해야 한다.

## 5. 현재 환경 전체 실행 결과

Python 3.12.14, numpy 2.3.5, pandas 2.2.3, sklearn 1.8.0, scipy 1.17.0, joblib 1.5.3, matplotlib 3.10.8, CPU single-thread에서 full을 실행했다. 동일 train/validation/test 표본과 같은 selected model이 나왔고 원 모델이 없던 자리에는 새 joblib 2개가 `runs/audit-full/models`에 생성됐다. 이것은 원 학습 모델 파일의 복원이 아니라 현재 환경 재학습이다.

| 항목 | ZIP 저장 기준 | 현재 full 재실행 |
| --- | --- | --- |
| validation selected MAE | 7.393895 | 7.192549 |
| final test selected MAE | 11.093786 | 11.417825 |
| daily/week alpha | .69 | .69 |
| profile/HGB alpha(profile weight) | .31 | .23 |
| ramp threshold | .45 | .30 |

원본 source를 같은 환경에서 경로만 주입해 실행한 예측과 portable 실행본의 2208행은 exact equality를 확인했다. torch가 설치돼 있지 않아 원본 실행 감사에서 torch seed/CUDA 보고에만 audit shim을 사용했다. V5에는 torch 모델 계산이 없으며 shim은 학습/추론에 관여하지 않는다. 따라서 portability 변경이 이번 오차 차이를 만든 증거는 없다. 원 학습 버전/로그/직렬화 모델이 없으므로 버전 차이 또는 저장 산출물 생성 이력의 차이라는 가능성만 남으며 원인을 확정하지 않는다. 정확한 원 성능 재현은 미완료다. detailed current results: benchmarks/v5/reproduction_audit.json.

7개 unittest 통과: 원본 hash, 미래 실제값 변경 시 특성 불변, cutoff 누락 거부, 시간 gap 충전 제외, 동가 가장 이른 구간, actual cost oracle, 어댑터 날짜/중복 horizon 검사. 어댑터는 임시 디렉터리에 24행/3행을 생성하고 현재 run의 MAE 11.417825를 읽는 것도 확인했다. 원 outputs를 overwrite하지 않았다. Streamlit/torch는 이 환경에 미설치이고 UI를 띄운 렌더/실서비스 테스트는 미실행이다.

## 6. 기술적 한계와 확정/추정 구분

확정: 2023 1년만 사용, 극단 고가 표본 부족, 실제 운영 수집/알림 없음, 원 joblib/lock 없음, 모델 스크립트 원경로 부적합, metadata의 required_input_columns는 69열 raw 전체여서 실제 사용 63개 특성과 혼동 가능, 학습 holiday는 actual calendar에서 가져오나 미래 추론은 제공 안 하면 0, 위험 라벨 함수는 학습 threshold와 별도 고정 기준, ffill age 제한 없음.

확정: timestamp는 연속이지만 원코드는 임의 새 입력의 전체 시계열 연속성을 엄격히 확인하지 않으며 shift는 행 기반이다. 심야 웹 필터는 날짜 없이 22~08시를 같은 하루의 두 조각으로 처리하고 자정을 가로지르는 연속구간을 구성하지 못한다. 충전 효율/부분시간/실제 요금/배터리 열화/실차 V2G는 구현되지 않았다.

확정: 원 CSV 풍력 3600행, 강수 7673행, 일사/일조 3979행 결측이다. reserve/LNG는 예시에서 하루 24행 같은 값이 반복된다. 'net_load'는 전국 수요에서 선택 지역 발전량들의 합을 빼는 proxy로 국가 순부하와 동등하지 않으며 서부/서울 집계의 중복과 단위도 확인해야 한다.

미확정: 일별 수급/연료값이 23시에 이미 공개되었는지, timestamp가 시작/끝 구간인지, 결측 강수가 무강수인지, SMP의 시장구분/개정 영향과 원 데이터 재배포 권한. 과거 timestamp만 사용하는 테스트는 publication leakage가 없음을 증명하지 않는다. HGB 원 fit은 5640행에서 auto early-stop 임계보다 작지만 데이터 확장시 내부 분할이 켜질 수 있어 V6는 시간순 early-stop을 명시한다.

피크 원인으로 희소 표본과 평균 수축/혼합, 빠진 미래 계통 정보가 의심되지만 원인 분석 실험 전에는 추정이다. 모델 이름 교체만으로 개선을 보장하지 않는다.

## 7. 통합 변경 및 보존

새 파일: model_v5/reference exact source, portable model_v5 source와 __init__, models_config 2개, requirements-model.txt, tools/run_v5.py, tools/import_v5_package.py, tests/test_v5_contracts.py, benchmarks/v5 원집계 4개 및 README/해시/현재 재현 기록, 요청한 인수인계 문서 6개, gitignore. 수정: README.md, services/v5_web_adapter.py. main app.py/outputs/legacy 파일은 기존 bytes 그대로 유지한다.

어댑터는 SMP_V5_RUN_DIR 또는 repo/runs/v5의 결과를 읽고 선택 정책과 측정 지표에서 출력한다. 날짜/24시간/horizon 및 duration 검사를 보강했다. 원본 V5 학습 알고리즘/손실/하이퍼파라미터는 변경하지 않았다. portable 실행본의 수정은 root 경로, torch optional 처리, 실행명령 안내에 한정한다.

GitHub로 새 공개한 자료는 소스, 설정, 집계 성능, 테스트 및 개발 문서다. 실제값을 포함한 행단위 연구 CSV, 원 보고서, 개인정보, 인증키, joblib은 제외한다. 이미 main에 공개된 demo/legacy files는 보존한다. 미확인 과거 실험본, 원문 보고서와 데이터는 형식적으로 통합하지 않는다.

## 8. 중요한 API/서비스 발견

KPX 신규 API는 23시경 갱신과 시간 끝점 정의를 명시한다. 구 API는 폐지 예정이고 시간 시작점 정의가 다르다. 공개 가격이 서비스 시점에 이미 있는지 확인하지 않으면 예측 연구의 정보집합과 제품 설명이 맞지 않을 수 있다. 신규 API 인증 응답/최신 시장규칙/실제 공개시각 검증은 남아 있다. 연구 예측과 공식 가격 기반 운영을 분리하는 계획은 PROJECT_SPEC.md와 DATA_API_PLAN.md에 기록했다.

공식 근거:

- https://www.data.go.kr/data/15131225/openapi.do
- https://www.data.go.kr/data/15076302/openapi.do
- https://www.data.go.kr/data/15084084/openapi.do
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html

## 9. 인수인계 가능 범위

Codex는 브랜치에서 코드/테스트/스키마/UI 개발을 바로 시작할 수 있다. 공개 Git만으로 비공개 연구 데이터를 이용한 학습까지 무조건 시작할 수 있다고 말할 수는 없다. 현재 작업 환경에는 자료가 직접 연결되어 있으며 새 환경에는 안전한 attachment 접근/마운트가 필요할 수 있다. API 키와 원환경 정보는 없어 운영 인증/원 성능 완전 재현이 대기 상태다. 사용자에게 파일 수동 복사를 요청하지 않고 접근 가능한 전달 경로를 먼저 사용해야 한다.
