# 원본 조사 — 2026-10-13

기준 저장소: parksiwon22/gradproject, 시작 커밋 b2d4f64.
별도 개발 브랜치: codex/v5-reproduction. main은 변경하지 않았습니다.

전체 원본 Git 파일 및 ZIP 디렉터리 목록은 reproduction.json에 기록했습니다.
원본 Git에는 app.py, services/v5_web_adapter.py, requirements.txt, README와
outputs 및 초기 데모 CSV/JSON만 있었습니다. Work가 작성한 PROJECT_AUDIT.md,
PROJECT_SPEC.md, MODEL_EXPERIMENT_PLAN.md, DATA_API_PLAN.md, WEB_REQUIREMENTS.md,
CODEX_TASKS.md는 원본 checkout에 없었습니다. 이 문서는 Codex가 새로 작성한 조사입니다.

현재 작업 공간에서 smp_project_v5_package.zip을 직접 확보했습니다.
ChatGPT Work의 파일 접근 가능성을 전제로 하지 않았고 추가 다운로드는 필요하지 않았습니다.
ZIP의 인계 README, V5 코드 전체, 데이터 스키마, 모델 설정과 결과 CSV/JSON을
조사했습니다. PDF 원문은 이번 실행 복구의 근거로 사용하지 않았습니다.
ZIP의 구형 LSTM과 LightGBM 지표는 인계 문서가 지정한 legacy 자료입니다.
기존 Git 파일들은 보존했습니다.

Git app.py와 ZIP app.py는 줄바꿈을 정규화한 텍스트가 동일합니다.
바이트 비교는 줄바꿈 형식 때문에 달랐으며 소스 내용의 버전 차이는 없습니다.

V5 학습 소스와 데이터가 Git에는 없었고 ZIP에는 있었습니다.
model/v5.py로 원본 알고리즘을 통합하고 경로 설정 및 실행 안내만 이식했습니다.
ZIP에 없던 day_ahead_hgb_residual_model.joblib,
day_ahead_ramp_classifier.joblib은 전체 학습으로 로컬 재생성했습니다.
필수 학습 파일은 모두 확보됐습니다. ZIP 원본은 local_original/에도 보존했습니다.

원본 CSV·ZIP·PDF·모델 바이너리·전체 학습 예측은 공개 권한이 확인되지 않아
Git에서 제외했습니다. 기존 공개 웹 데모만 기존 outputs 경로에서 유지합니다.
API 키와 개인정보를 추가하지 않았습니다.

변경 사항: 모델 소스 통합, 상대 경로 및 환경 변수 설정, 저장 모델 예측 CLI,
웹 지표의 실제 결과 연동, 의존성 고정, 단위·통합·서버 테스트, 개발 안내 및 재현 보고.
웹 app.py는 변경하지 않았습니다. 데이터 수집·V6·실서비스·배포는 이번 범위 밖입니다.
