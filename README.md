# 전력예보 V5 기준 보존 및 V6 개발 준비

현재 app.py는 첨부 V5와 동일 내용의 Streamlit 앱이고, 2023-10-01 저장된 24시간 SMP와 충전 추천을 보여주는 과거 데모다. 실시간 데이터 수집이나 매일 알림은 아직 구현되지 않았다. main의 LightGBM 파일은 현재 앱이 읽는 V5 기준 파일이 아니다.

2026-10-14 연구 재설계 추천은 발표 전 AI 예측, 공식 가격 수신 후 재최적화와 가상 V2G를 결합한 C 최소형이다. 현재 문서만 수정했으며 구현/배포 완료를 뜻하지 않는다. RESEARCH_DIRECTION.md와 CODEX_PROMPT.md를 먼저 읽고 CODEX_TASKS.md부터 개발한다. 실제 감사는 PROJECT_AUDIT.md, 전체 방향은 PROJECT_SPEC.md, 모델/데이터/화면 요구사항은 MODEL_EXPERIMENT_PLAN.md, DATA_API_PLAN.md, WEB_REQUIREMENTS.md에 있다.

## 웹 실행

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

표시 파일은 outputs/next_day_smp_forecast_tuned.csv, outputs/recommended_3h_charging_window_tuned.csv, outputs/v5_metrics.json이다. 기존 demo bytes는 보존한다.

## 모델 개발과 재현

Python 3.12를 권장한다. requirements-model.txt는 이번 실행 환경이며 원 학습 lock으로 주장하지 않는다. HGB는 CPU를 사용하고 torch는 portable 실행본에서 선택 사항이다.

```bash
python -m pip install -r requirements-model.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
python tools/import_v5_package.py --package /accessible/path/smp_project_v5_package.zip
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/run_v5.py --full --run-dir runs/v5
```

private_data와 private_reference에는 첨부 ZIP에서 hash를 검사한 자료를 직접 로컬 연결한다. 원 데이터/행단위 실제값 CSV는 공개 Git에 포함하지 않았다. 별도 Codex 환경이 첨부에 접근하지 못하면 비공개 전달/마운트가 필요하다. 공개 Git만으로 학습 입력을 얻을 수 없다.

model_v5/reference/smp_predictor_v5.py는 ZIP bytes 그대로다. model_v5/smp_predictor_v5.py는 알고리즘을 보존한 portable 실행본이며 runs 아래에 새 결과를 저장한다. 원 설정은 model_v5/models_config, 원집계는 benchmarks/v5에 있다. 원 joblib 2개는 ZIP에도 없고 full 실행 시 새로 학습한다.

웹 출력 갱신은 명시적 개발 작업이다. 아래 실행은 outputs를 바꾸므로 검토 후 별도 결과 경로나 작업 브랜치에서 수행한다.

```bash
SMP_V5_RUN_DIR=runs/v5 python services/v5_web_adapter.py
```

ZIP 저장 MAE는 11.093786, 이번 환경 full 재실행은 11.417825다. 원환경/모델 부재로 차이는 미해결이며 저장 기준을 덮어쓰지 않았다. 급등 F1 .435897은 검증값이며 테스트값은 .277512다.

## 제약

main 직접 변경/병합과 실서비스 배포는 사용자 승인 후 진행한다. 연구 데이터와 운영 가격/요금을 구분한다. 23시 공식 SMP가 이미 제공되는지 먼저 확인한다. 소매 충전요금과 SMP를 같은 값으로 가정하지 않는다. V2G는 초기 시뮬레이션이며 실제 제어/수익을 보장하지 않는다. 완료조건은 CODEX_TASKS.md에 있다.
