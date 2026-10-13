# V5 로컬 자산 및 Codex Cloud 준비

확인일: 2026-10-13. 현재 Cloud 환경 생성, 자산 업로드, 저장소 공개 범위 변경은
수행하지 않았습니다. 아래 Cloud 절차는 준비 안내이며 실제 Cloud 실행은 미검증입니다.

## 로컬 경로와 확보 방법

경로는 저장소 루트 기준입니다. 현재 컴퓨터의 절대 경로는 Git에서 제외된 artifacts/LOCAL_ASSET_PATHS.md에 보관합니다.

| 자산 | 저장소 기준 경로 | 확보 / 생성 방법 |
| --- | --- | --- |
| 원본 ZIP | `../smp_project_v5_package.zip` | 기존 ChatGPT 프로젝트 `smp 졸작`의 V5 첨부파일과 같은 이름으로 로컬에 확보된 패키지 |
| 학습 CSV | `data\aligned_hourly_all_features_2023.csv` | ZIP의 `pkg/01_model_v5/data/aligned_hourly_all_features_2023.csv` 추출 |
| HGB 회귀 모델 | `artifacts\models\day_ahead_hgb_residual_model.joblib` | `python -m model.v5 --full` |
| 급등 분류 모델 | `artifacts\models\day_ahead_ramp_classifier.joblib` | 같은 전체 학습 명령 |
| 특징 메타데이터 | `artifacts\models\day_ahead_feature_metadata.json` | 같은 전체 학습 명령 |
| 선택 정책 | `artifacts\models\day_ahead_policy.json` | 같은 전체 학습 명령 |
| 학습 결과 | `artifacts\results\` | 같은 전체 학습 명령, CSV·JSON·그래프 생성 |
| 원본 보존본 | `local_original\pkg\01_model_v5\` | ZIP에서 추출한 소스·데이터·설정·기존 결과 |

CSV는 4,049,633 bytes, 회귀 모델은 1,494,662 bytes, 분류 모델은 1,494,912 bytes입니다.
ZIP에는 joblib 2개가 없었으므로 해당 바이너리는 로컬 전체 학습으로 재생성했습니다.
추론 파일을 이전할 때는 joblib 2개와 같은 실행에서 생성한 JSON 2개를 함께 가져옵니다.
joblib은 신뢰하는 원본 또는 직접 학습한 파일만 불러옵니다.

CSV SHA-256:
`aa9101486434d9a812edf9fc9d56b3a6b93e1140a288c7de55f68bbbac6419f2`

ZIP SHA-256:
`304d64170ddab2d8fb52a0319498cca3febbe342b1264ebd3e2f6f2a0a3d4850`

이 해시는 현재 로컬 ZIP과 추출 CSV의 동일성 기준입니다. ChatGPT 첨부파일 서버와
직접 대조한 결과는 아닙니다. 상세 환경은 `docs/reproduction.json`을 사용합니다.

## 새 환경에서 재생성

권한을 가진 사용자가 동일 CSV를 `data/`에 준비한 후 저장소 루트에서 실행합니다.

```bash
python -m pip install -r requirements-model.txt
python -m pip check
python -m model.v5 --full
python -m scripts.predict --date 2023-10-01
python -m unittest discover -s tests -v
```

Python 3.12.14와 고정 의존성을 기준으로 합니다. `--full`은 로컬 모델과 결과를
덮어씁니다. 별도 재검증에서는 SMP_RESULTS_DIR과 SMP_MODELS_DIR을 새 폴더로
지정하세요. Linux에서도 경로 설정은 이식됐지만 동일 성능과 설치는 실제 실행으로
확인해야 합니다. `--quick`은 성능 재현 및 배포 모델 생성용이 아닙니다.

환경 변수 SMP_DATA_PATH, SMP_RESULTS_DIR, SMP_MODELS_DIR을 쓰지 않으면
저장소의 data/, artifacts/results/, artifacts/models/가 기본 경로입니다.
CSV만 있으면 모델은 다시 학습할 수 있습니다. 저장 모델만 이전한 경우에도
2023-10-01 추론에는 2023-09-30 23시까지의 이력 데이터가 필요합니다.
웹 어댑터 재생성에는 모델뿐 아니라 학습 결과 CSV와 summary JSON도 필요합니다.
`python -m services.v5_web_adapter`는 공개 웹 데모 outputs/를 덮어쓰므로
현재 동결 기간에는 실행하지 않습니다. 기존 웹 실행은 저장된 outputs/로 가능합니다.

## Cloud에서 확보할 수 있는 경로

현재 공개 Git 브랜치는 코드·문서·기존 웹 데모만 제공합니다. 로컬의 data/,
artifacts/, local_original/와 원본 ZIP은 checkout으로 가져올 수 없습니다.
현재 세션에서 사용 가능한 비공개 자산 URL이나 Cloud 전송 연결은 확보되지 않았습니다.
ChatGPT 프로젝트 첨부파일과 로컬 디스크가 Cloud에 자동 연결된다고 가정하지 않습니다.

사용자가 클라우드 처리 권한을 확인한 뒤, 접근을 제한한 HTTPS 저장소에 필요한
CSV만 보관하고 Cloud에서 인증해 받는 방식을 준비안으로 제안합니다.
원본 ZIP 전체보다 CSV 하나를 가져와 모델을 재생성하면 PDF와 다른 원본 자료의
불필요한 이전을 줄일 수 있습니다. 이미 만든 모델을 옮길 경우 JSON 2개도 포함하고
송신 측 SHA-256과 실행 환경을 함께 기록합니다. 인증정보·서명 URL은 Git에 넣지 않습니다.

현재 Codex Cloud는 저장소·의존성·접근 설정을 가진 환경에서 작업별 workspace를
제공합니다. 설치 과정에서 비공개 자산을 준비하고, 환경의 공유 범위를 제한합니다.
HTTPS 서비스 인증에는 대상 도메인을 제한한 network secret을 사용할 수 있습니다.
이는 확보 절차 설계이며 실제 저장소 접근과 계정 설정은 아직 확인하지 않았습니다.
[공식 Cloud 환경 문서](https://learn.chatgpt.com/docs/environments/cloud-environments).

설정 화면이 Codex Cloud (Legacy)인 경우에는 별도 절차가 적용됩니다.
공식 문서상 legacy secrets는 setup 단계에서만 제공되므로 해당 단계에서
자료를 받아야 합니다. 현재 Cloud의 network secret과 혼동하지 않습니다.
[공식 Legacy 환경 문서](https://learn.chatgpt.com/docs/environments/cloud-environment).

Cloud 초기 준비에서 확인할 순서:

1. `codex/v5-reproduction` 브랜치와 원하는 커밋이 실제 checkout됐는지 확인합니다.
2. 승인된 비공개 전송 경로와 Cloud 처리·환경 공유 권한을 확인합니다.
3. 동일 CSV를 내려받아 위 SHA-256과 대조합니다. 다운로드 실패·해시 불일치 시 학습을 중단합니다.
4. 고정 의존성으로 전체 학습하고 모델 2개 및 설정 JSON 2개를 생성합니다.
5. 재현 테스트에서 MAE·R²·행 수를 확인합니다. 실패 결과는 기록하고 성능을 추정하지 않습니다.
6. 원본 데이터·모델·결과가 Git 추적 목록에 없는지 확인합니다.

Linux 파일 확인 예시(확보된 CSV만 읽음):

```bash
sha256sum data/aligned_hourly_all_features_2023.csv
git ls-files data artifacts local_original
```

두 번째 명령은 안내 README만 출력해야 합니다. `.gitignore`는 Git 업로드를
막을 뿐 Cloud 환경 사용자 간 접근을 제한하지 않으므로 환경 권한도 별도로 확인합니다.
실제 비공개 저장소와 인증이 준비되기 전에는 Cloud 학습이 가능한 상태라고 보고하지 않습니다.
