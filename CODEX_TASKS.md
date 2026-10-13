# Codex 작업 순서와 완료 조건

## 시작 지시

이 작업 브랜치에서 PROJECT_AUDIT.md -> PROJECT_SPEC.md -> MODEL_EXPERIMENT_PLAN.md -> DATA_API_PLAN.md -> WEB_REQUIREMENTS.md 순서로 읽는다. V5를 기준으로 사용하고 별도 V6 실험본을 기준에 섞지 않는다. main 직접 수정/병합/배포 금지. 원본 baseline 및 benchmarks를 overwrite하지 않는다. 작은 구현 판단은 아래 기본값으로 진행한다.

## 현재 실행 가능한 상태

소스/설정/인수인계 문서와 7개 unittest가 있다. 모델 실행을 현재 첨부 데이터로 직접 검증했다. 공개 Git만으로 V5 학습 데이터와 원 joblib 2개를 얻을 수는 없다. 현재 작업환경에는 `private_data/aligned_hourly_all_features_2023.csv`, `private_reference/v5_day_ahead_predictions.csv`, `runs/audit-full/models`가 있다. 별도 Codex 환경에서는 해당 파일이 실제로 보이는지 먼저 확인한다. 이전 ChatGPT 프로젝트 첨부/대화 전체 접근을 가정하지 않는다.

```bash
python -m pip install -r requirements-model.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
python tools/import_v5_package.py --package /accessible/path/smp_project_v5_package.zip
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/run_v5.py --full --run-dir runs/v5
SMP_V5_RUN_DIR=runs/v5 python services/v5_web_adapter.py
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

도구가 이미 첨부 ZIP에 접근한다면 import를 직접 수행한다. 사용자의 다운로드/수동 복사를 요구하지 않는다. 접근하지 못하면 그 환경에서 가능한 비공개 전달/마운트를 먼저 사용한다. 실제 자료를 얻지 못한 상태에서는 synthetic 테스트와 기능 개발은 계속하되 V5 재현 성공이라고 주장하지 않는다. 설치/네트워크가 막히면 정확한 패키지와 단계만 보고한다.

## 순서와 산출물

| 순서 | 작업 | 산출물 | 테스트 및 완료 조건 |
| --- | --- | --- | --- |
| 0 | 브랜치/기준 확인 | base SHA 및 audit manifest | main 보존, 원본 hash 일치 |
| 1 | 원환경 재현 조사 | environment lock, reproduction report | 저장 MAE 11.093786 대비 차이 설명; 저장 원본 보존 |
| 2 | publication/시간 정의 검증 | 데이터 출처대장, API 계약, 시장/interval 맵 | API start/end 시간 정렬, 23시 공개 여부 확인 |
| 3 | as-of 파이프라인 | 관측/예보 vintage 저장, feature builder | 미래 publication/revision 제거, gap/결측 정책 |
| 4 | 공정 모델 실험 | experiment config, folds, run registry | validation 선정, test 잠금, 동일 표본 비교 |
| 5 | 피크/확률 평가 | diagnostics, calibrated policy | peak MAE/bias, F1/PR-AUC, coverage 및 regret |
| 6 | EV 최적화 | 독립 charging service | SOC 보존, 효율/부분시간, 심야 날짜, infeasible |
| 7 | V6 UI | 별도 entrypoint와 adapter | AppTest 및 모바일 렌더, 상태/날짜/단위 |
| 8 | 자동 실행/dry-run | 수집/추론 CLI, outbox, atomic store | 실패시 이전본 보존, idempotency, 키 마스킹 |
| 9 | V2G 시뮬레이션 | 최적화 모듈 및 계약 가정 | 동시 충방전 금지, SOC/출력, 열화/정산 민감도 |
| 10 | 사용자 검토 | 변경요약/성능/리스크/배포안 | 검토 후에만 main 병합과 서비스 배포 |

## Task 1 재현성

ZIP 저장 결과와 재학습 결과 차이는 현재 미해결이다. 2026-10-13 환경에서 test MAE 11.417825, alpha_profile .23, ramp threshold .30이 나왔다. 원래 alpha .31와 threshold .45를 맞추기 위해 test를 보면서 임의 튜닝하지 않는다. 원 학습 lock/log/joblib의 접근 가능한 경로를 조사하고, source/data hash와 sklearn/numpy 버전에 따른 차이를 재현한다. 근거가 없으면 '원환경 미상'으로 남긴다. 새 환경 결과는 별도 기준 후보이며 기존 11.09를 새 실험 결과처럼 사용하지 않는다.

## Task 2 서비스 문제 정의

현재 제공 API의 23시경 갱신과 공식 가격 발표를 실제 관측한다. 가격이 이미 알려져 있다면 공식 가격 기반 추천이 기본이다. 예측 연구는 더 이른 cutoff 또는 미공개 기간으로 별도 정의한다. 발표 전 cutoff로 바꾸면 V5의 23시 origin 특성은 그대로 사용할 수 없으므로 새 history horizon과 as-of 입력을 설계한다. 이 선택을 코드에 조용히 하드코딩하지 말고 검토자료에 명시한다.

## Task 4 실험 실행

초기 실행은 HGB 후보와 전일/전주 baseline부터 수행한다. optional 모델 설치는 별도 requirements로 관리하고 feature 인터페이스를 공유한다. feature-selection/calibration/threshold는 validation 안에서 정한다. 실험 실패, 미설치, 시간초과도 registry에 기록한다. 모델 개선이 없으면 그 결과로 종료할 수 있다.

## 필수 회귀 테스트

24개 연속 시간 및 horizon 1~24, origin 단일 고정, 실제 미래가격/수요 변조에도 입력 특성 불변, 공개시각 뒤 예보와 정정값 차단, time-ending API의 자정 처리, 시장 분리, 잘못된 단위, 결측 cutoff, hour gap, 날짜 미존재, SOC/에너지/효율, 출발 제약, stale/failure fallback, 중복 알림, secret 로그 마스킹을 포함한다. 코드 구현을 그대로 복제하는 테스트 대신 입력이 깨졌을 때의 실제 계약을 검증한다.

## 완료 보고 방식

실제로 수정한 파일, 실행 명령과 결과, 기준/후보 성능, 원본 보존 여부, 실패/미실행 항목, 남은 사용자 검토 지점만 보고한다. raw 데이터/보고서/연락처/키를 공개하지 않는다. V6 완료, 실시간 운영, 알림 전송, V2G 실차 제어는 실제 수행 근거 없이는 주장하지 않는다.
