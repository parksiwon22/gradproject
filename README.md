# 전력예보 V5 개발 환경

전일 23시까지의 이력으로 다음 날 24시간 SMP를 직접 예측하고 EV 충전 시간을
추천합니다. 기존 Streamlit 화면과 정적 데모를 유지합니다. V5는 HGB 잔차 회귀와
일별·주별 프로파일의 혼합 모델입니다.

## 환경 구성

검증 환경: Windows, Python 3.12.14, CPU.

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-model.txt
python -m pip check
```

웹만 실행할 때는 requirements.txt만 설치하면 됩니다. 현재 작업 공간에서는
상위 폴더의 기존 가상환경 `..\.venv\Scripts\python.exe`를 재사용해 검증했습니다.
독립된 새 가상환경 설치와 Linux/macOS 실행은 아직 검증하지 않았습니다.

## 데이터와 실행

개인적으로 확보한 smp_project_v5_package.zip의
`pkg/01_model_v5/data/aligned_hourly_all_features_2023.csv`를 data/에 넣으세요.
데이터와 모델은 Git에서 제외되므로 새 checkout에는 별도로 준비해야 합니다.
데이터가 없으면 학습할 수 없습니다.

저장소 루트에서 실행합니다.

```bash
python -m model.v5 --full
python -m scripts.predict --date 2023-10-01
python -m services.v5_web_adapter
python -m unittest discover -s tests -v
python -m scripts.smoke_server
python -m streamlit run app.py
```

웹 주소는 http://localhost:8501 입니다. 웹은 저장된 2023-10-01 데모를 표시합니다.
실시간 데이터 수집·API 실서비스 등록·V6 모델 교체·배포는 수행하지 않았습니다.

환경 변수 SMP_DATA_PATH, SMP_RESULTS_DIR, SMP_MODELS_DIR로 입력과 산출물 위치를
지정할 수 있습니다. 기본 경로는 소스 파일 위치를 기준으로 결정됩니다.
`--quick`은 성능 재현용이 아닙니다. 별도 결과 폴더를 지정해 실행하세요.
`--full`은 기존 로컬 결과와 모델을 다시 생성합니다.

## 구조

- model/: 원본 V5 학습·예측 알고리즘, 경로 설정만 이식
- data/: 로컬 학습 CSV, 공개 제외
- artifacts/models/: 재생성한 joblib 2개와 설정 JSON
- artifacts/results/: 재현 CSV·JSON·그래프, 공개 제외
- services/: V5 결과를 기존 웹 스키마로 변환
- app.py, outputs/: 기존 웹과 데모
- scripts/, tests/: 예측 CLI, 서버 확인, 단위·통합 테스트
- docs/: 원본 조사와 재현 기록

## V5 재현

2023년 1~8월 학습, 9월 검증, 10~12월 최종 테스트를 사용했습니다.
168시간 입력 및 lag 특징 생성으로 학습 235일, 검증 30일, 테스트 92일입니다.
최종 MAE **11.0937861480원/kWh**, R² **0.6429194849**로 ZIP 결과와 일치했습니다.
근거와 경고는 [재현 기록](docs/V5_REPRODUCTION.md)과
[실행 환경 및 입력 해시](docs/reproduction.json)를 보세요.

7개 단위·통합 테스트와 Streamlit 서버 HTTP 확인이 통과했습니다.
학습 데이터는 2023년뿐이며 실시간 미래 예측 성능을 검증한 결과는 아닙니다.
후속 V6 실험은 V5를 유지하고 별도 소스·산출물 폴더에서 진행하세요.

## V5 동결 및 후속 준비

현재 V6 모델 개선과 웹 개편은 사용자 시작 지시까지 대기합니다.
[로컬 자산 경로·재생성·Cloud 확보 안내](docs/PRIVATE_ASSETS_AND_CLOUD.md)와
[Work 명세서 수신 절차](docs/WORK_SPEC_HANDOFF.md)를 참고하세요.
Cloud 비공개 자료 전송은 아직 설정하지 않았습니다.
