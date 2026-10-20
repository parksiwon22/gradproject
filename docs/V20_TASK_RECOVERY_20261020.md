# V20 Windows 예약작업 복구 — 2026-10-20

실제 로컬 Windows `DESKTOP-VTB6AP7`, 사용자 `desktop-vtb6ap7\박시원`에서 확인했다. 저장소 절대경로는 `C:\smpsmp\smpsmpcodexuse\gradproject`이며 브랜치는 `research/v20-issuance-auto-ev-20261017`이다. 신규 V21은 만들지 않았다.

## 확인한 실제 구성과 원인

- `C:\smpsmp\smpsmpcodexuse\.venv\Scripts\python.exe`와 `pythonw.exe`가 모두 존재한다. Python 3.12.14의 프로젝트 환경에서 Streamlit 1.58.0, pandas, NumPy, scikit-learn, LightGBM, CatBoost, XGBoost, joblib import가 성공했다. 앞선 시스템 Python에서 Streamlit이 빠졌던 것은 `.venv`가 아닌 인터프리터를 사용했기 때문이다.
- 실제 작업은 박시원 사용자, Interactive 로그인, 제한 권한으로 등록돼 있고 시작 폴더는 저장소다. `GradprojectV12ForwardEV`는 `-m v20.service`, 15분 반복 및 매일 22:50/22:58, 10분 제한이다. 단일 `GradprojectV19Publish`는 `-m v19.publisher --warm`, 매일 22:59:30/23:00/23:02, 2분 제한 및 중복 실행 무시다. 기존 V11/V13/V14/V15/V16 작업도 별도 이름으로 남아 있었다.
- 작업은 `pythonw.exe`를 사용해 stdout/stderr가 보이지 않았다. 두 핵심 작업 모두 `WakeToRun=False`였다. 현재 전원 계획은 AC 절전 타임아웃 0(절전 안 함), 배터리 절전 180초, AC wake timer 허용, 배터리 wake timer 비허용이었다. 과거 누락일에 기기가 절전했는지 기록은 없어 단일 원인으로 확정할 수 없지만, 배터리에서 실행되지 않을 수 있는 재현 가능한 조건을 확인했다.
- Task Scheduler Operational 로그는 `enabled:false`였고, 현재 제한 계정에서 활성화는 Access denied로 거부됐다. 과거 이벤트가 없으므로 10월 17~19일 누락의 정확한 OS 단계 원인은 확정할 수 없다. `GradprojectV19Publish`의 직전 결과 `0x41306`은 [Microsoft Task Scheduler 결과 코드](https://learn.microsoft.com/en-us/windows/win32/taskschd/task-scheduler-error-and-success-constants)에 정의된 `SCHED_S_TASK_TERMINATED`다. 종료 주체/이유는 이벤트 기록 부재로 알 수 없다.

## 변경 및 백업

실제 XML을 수정 전에 백업했다. 처음 적용 전 백업은 로컬 ignored 경로 `artifacts/v20/scheduler/GradprojectV12ForwardEV_before_20261020T223027142.xml`, `artifacts/v20/scheduler/GradprojectV19Publish_before_20261020T223032753.xml`이고, pythonw 모드로 전환 직전 백업은 `GradprojectV12ForwardEV_before_20261020T225253763.xml`, `GradprojectV19Publish_before_20261020T225255459.xml`이다. 작업 이름/개수와 트리거는 유지했다. 변경 후 두 작업 모두 Python 경로, 인수, 저장소 시작 폴더, 3개 trigger, `WakeToRun=True`, `StartWhenAvailable=True`, idle 종료 시 중단 안 함을 확인했다. 기존 전역 AC/DC 절전 설정은 바꾸지 않았다.

새 `v20.task_runner`는 모듈 실행 시작/완료 시각, Python 경로, 작업 폴더, 종료코드, stderr/stdout, 예외를 `artifacts/v20/task_runs`에 별도 불변 JSON 두 개로 기록한다. 환경변수 이름이 KEY/TOKEN/SECRET/PASSWORD/CREDENTIAL인 값과 URL 인증 파라미터는 로그에서 가린다. 모듈 import 실패와 모듈 실행 실패를 구분한다. publisher 내부의 준비본 없음, 공개상태 보류, 공식가격 기확보 등의 상세 사유는 publisher 자체 `publish_runs`와 함께 확인한다.

22:39:39의 첫 스케줄 실행은 시작 JSON을 남겼지만 완료 JSON 없이 종료했고 LastTaskResult는 `0xC000013A`였다. 같은 실행을 포그라운드에서 재현했을 때 V20 service 완료 로그를 쓰는 V19 공용 writer가 Windows `os.link`에서 `PermissionError: [WinError 5] Access is denied`를 냈다. `v20.task_runner`가 실행 중인 프로세스에서만 V12/V18/V19 공용 `write_once` 함수를 create-only rename fallback으로 바꾸도록 했다. 기존 버전 소스와 불변 기록 writer는 수정하지 않았다.

이후 콘솔 모드 `python.exe`를 사용하는 Task Scheduler 실행도 22:50:02에 시작됐지만 완료 기록 없이 `0xC000013A`로 끝났다. 파일 로그가 stdout에 의존하지 않도록 되어 있어 기존에 성공했던 비콘솔 Python 모드 `pythonw.exe`로 두 작업을 다시 설정했다. 22:53:02에 기존 collector를 Task Scheduler에서 수동 시작한 비교 실행은 22:53:44에 종료됐고 Task Scheduler 결과 0, wrapper 시작/완료 파일, observer 및 service 로그를 모두 남겼다. 현재 확인으로는 `python.exe` 콘솔 모드와 종료가 연관되며, Operational 이벤트가 비활성화되어 OS가 해당 프로세스를 중단한 세부 이유는 여전히 미확정이다. `pythonw.exe`+파일 로그 조합은 Task Scheduler 경로에서 한 번 성공했다. 이 수동 시험은 22:50 정시 예약 실행이 아니다.

Task Scheduler Operational 로그 활성화는 Windows가 Access denied를 반환해 완료되지 않았다. 따라서 Python이 시작하기 전의 실패는 아직 worker 로그에 남지 않으며, 그 구간은 관리자 세션에서 이벤트 로그를 켜야 확인할 수 있다. 배터리 wake timer는 배터리 소모 때문에 그대로 두었다. 절전 중 실행이 필요하면 AC 전원 연결을 권장한다.

## 2026-10-20 실제 트리거 및 발급 감사

22:42:09 수동 smoke run은 22:42:51에 종료했다. KPX API 정상 무자료 2건, 일반 공개 0원 미검증 2건, EPSIS 미관측 4건을 기록했고 V19 observer, V20 감사, D-2(22일)/D-3(23일) 준비 및 기존 예측 사후평가 단계가 성공했다. 이는 실제 정시 발급은 아니다.

22:50 실제 트리거는 콘솔 `python.exe` 액션을 사용했고 작업 결과 `0xC000013A`, 완료 로그가 없는 실패로 남았다. 콘솔 액션 종료 원인은 Task Scheduler Operational 로그가 꺼져 있어 더 세분할 수 없다. 액션을 `pythonw.exe`로 되돌린 뒤 22:53 수동 Task Scheduler 비교 실행이 정상 종료했고, 22:58 실제 준비 트리거도 22:58:03~22:58:28에 정상 종료했다. 준비 단계는 관측 8건(KPX API `normal_no_data` 2건, 일반 공개 `unverified_all_zero` 2건, EPSIS `not_observed` 4건), 모델 추론/EV 준비 및 사후평가 단계를 모두 완료했다.

준비 포인터 `artifacts/v19/ready/20261020T2300.json`은 22:58:21.630752 KST에 생성됐고 가리키는 패킷의 SHA-256은 `b1e79abda15192b7eebf193f9cb1ba5b8ad2471a9d1b5d90b7fb28cf14e6e34a`다. 입력 스냅샷은 22:58:18 KST, 패킷 내 최신 입력 최초 수신은 20:32:06 KST였다. 모델 registry는 10월 17일 12:20:13 KST에 잠겼으며 registry 및 각 모델 파일 해시는 준비 및 발급 검사에서 일치했다. 이 자료는 `UNISSUED_PREPARATION_NOT_LIVE_FORECAST` 준비본이다.

22:59:30 warm 작업은 실제 시작해 23:00 경계까지 정상 종료했고, 23:00 발급 트리거와 23:02 재시도도 각각 실제 시작해 Task Scheduler 결과 0 및 시작/완료 로그를 남겼다. 발급 JSON은 23:00:00.137611, 23:00:02.342255, 23:02:02.422008 KST에 생성됐다. 처음 두 검사에서 9개 모델/대상 조합은 모두 `unverified_publication_general_public_unverified_all_zero`로 건너뛰었다. 일반 홈페이지의 숫자 0 응답은 미공개 증거로 판정하지 않았고, KPX의 `normal_no_data`만으로도 발급 조건을 통과시키지 않았다. 23:02 재시도 때는 최신 KPX API 표본이 180초 freshness 한도를 넘어 9개 조합 모두 `publication_check_not_fresh_kpx_api`로 건너뛰었다. publisher hot path의 `network_calls`는 모두 0이다. 이 시도의 공식 공개 상태는 “가격 확보 확인 없음, 미공개도 확정할 수 없음”이다.

이날 publisher 작업들은 모두 정상 실행했지만 AI 예측 발급은 0건이다. `artifacts/v19/forecasts`와 `artifacts/v19/commits`에 이 발급 시도의 신규 파일은 없다. 기존 예상 기록/모델을 비교하거나 수정하지 않았고 과거 시각 발급도 만들지 않았다. 22:58 준비본, 예약작업 성공, 적격한 예측 발급은 각각 별개로 기록된다.

23:02 직후 V20 읽기 전용 감사를 다시 실행해 `artifacts/v20/status.json` 및 `docs/V20_STATUS.json`, 공개 집계표/그래프를 갱신했다. 22:58 예약 collector가 갱신한 대상일별 홈페이지 0 응답 관측은 기존 0 응답 감사 그래프에 반영됐다. 독립 사전예측은 0, 공식가격 평가 추가 완료일도 0이다. 보존 감사는 848개 기존 파일에서 변경 0건이었다. 회귀검증은 기존 full suite와 V20 집중 테스트를 통과했다.

다음 실행을 개선할 때는 23:02 fallback 직전에 publisher가 새 KPX 공개상태 표본을 받아 ledger freshness를 갱신할 수 있는 별도 짧은 준비 트리거가 필요하다. 단, 공식 공개 여부가 불확실한 홈페이지 0 응답을 정상 미관측으로 바꾸거나 발급 gate를 완화해서는 안 된다. Task Scheduler Operational 이벤트 로깅은 여전히 관리자 권한이 필요하므로 Python이 시작되기 전 실패 원인은 worker 파일 로그로 검출할 수 없다.
