# V20 Windows 예약작업 복구 — 2026-10-20

실제 로컬 Windows `DESKTOP-VTB6AP7`, 사용자 `desktop-vtb6ap7\박시원`에서 확인했다. 저장소 절대경로는 `C:\smpsmp\smpsmpcodexuse\gradproject`이며 브랜치는 `research/v20-issuance-auto-ev-20261017`이다. 신규 V21은 만들지 않았다.

## 확인한 실제 구성과 원인

- `C:\smpsmp\smpsmpcodexuse\.venv\Scripts\python.exe`와 `pythonw.exe`가 모두 존재한다. Python 3.12.14의 프로젝트 환경에서 Streamlit 1.58.0, pandas, NumPy, scikit-learn, LightGBM, CatBoost, XGBoost, joblib import가 성공했다. 앞선 시스템 Python에서 Streamlit이 빠졌던 것은 `.venv`가 아닌 인터프리터를 사용했기 때문이다.
- 실제 작업은 박시원 사용자, Interactive 로그인, 제한 권한으로 등록돼 있고 시작 폴더는 저장소다. `GradprojectV12ForwardEV`는 `-m v20.service`, 15분 반복 및 매일 22:50/22:58, 10분 제한이다. 단일 `GradprojectV19Publish`는 `-m v19.publisher --warm`, 매일 22:59:30/23:00/23:02, 2분 제한 및 중복 실행 무시다. 기존 V11/V13/V14/V15/V16 작업도 별도 이름으로 남아 있었다.
- 작업은 `pythonw.exe`를 사용해 stdout/stderr가 보이지 않았다. 두 핵심 작업 모두 `WakeToRun=False`였다. 현재 전원 계획은 AC 절전 타임아웃 0(절전 안 함), 배터리 절전 180초, AC wake timer 허용, 배터리 wake timer 비허용이었다. 과거 누락일에 기기가 절전했는지 기록은 없어 단일 원인으로 확정할 수 없지만, 배터리에서 실행되지 않을 수 있는 재현 가능한 조건을 확인했다.
- Task Scheduler Operational 로그는 `enabled:false`였고, 현재 제한 계정에서 활성화는 Access denied로 거부됐다. 과거 이벤트가 없으므로 10월 17~19일 누락의 정확한 OS 단계 원인은 확정할 수 없다. `GradprojectV19Publish`의 직전 결과 `0x41306`은 [Microsoft Task Scheduler 결과 코드](https://learn.microsoft.com/en-us/windows/win32/taskschd/task-scheduler-error-and-success-constants)에 정의된 `SCHED_S_TASK_TERMINATED`다. 종료 주체/이유는 이벤트 기록 부재로 알 수 없다.

## 변경 및 백업

실제 XML을 먼저 백업한 뒤 기존 두 작업의 액션만 `python.exe -u -m v20.task_runner` 경유로 바꿨다. 준비된 실제 XML 백업은 로컬 ignored 경로 `artifacts/v20/scheduler/GradprojectV12ForwardEV_before_20261020T223027142.xml`, `artifacts/v20/scheduler/GradprojectV19Publish_before_20261020T223032753.xml`에 있다. 작업 이름/개수와 트리거는 유지했다. 변경 후 다시 조회해 두 작업 모두 Python 경로, 인수, 저장소 시작 폴더, 트리거 3개, `WakeToRun=True`, `StartWhenAvailable=True`, idle 종료 시 중단 안 함을 확인했다. 기존 전역 AC/DC 절전 설정은 바꾸지 않았다.

새 `v20.task_runner`는 모듈 실행 시작/완료 시각, Python 경로, 작업 폴더, 종료코드, stderr/stdout, 예외를 `artifacts/v20/task_runs`에 별도 불변 JSON 두 개로 기록한다. 환경변수 이름이 KEY/TOKEN/SECRET/PASSWORD/CREDENTIAL인 값과 URL 인증 파라미터는 로그에서 가린다. 모듈 import 실패와 모듈 실행 실패를 구분한다. publisher 내부의 준비본 없음, 공개상태 보류, 공식가격 기확보 등의 상세 사유는 publisher 자체 `publish_runs`와 함께 확인한다.

Task Scheduler Operational 로그 활성화는 Windows가 Access denied를 반환해 완료되지 않았다. 따라서 Python이 시작하기 전의 실패는 아직 worker 로그에 남지 않으며, 그 구간은 관리자 세션에서 이벤트 로그를 켜야 확인할 수 있다. 배터리 wake timer는 배터리 소모 때문에 그대로 두었다. 절전 중 실행이 필요하면 AC 전원 연결을 권장한다.

## 오늘의 발급 관측

복구 후 실제 다음 트리거는 10월 20일 22:50, 22:58, 22:59:30, 23:00, 23:02다. 감사 시각 기준 아직 해당 트리거가 도래하지 않아 이번 발급을 성공/실패로 판정하지 않았다. 기준은 10월 22일 D-2, 10월 23일 D-3이며, 기존 공식가격 공개 gate와 모델/입력 시각 gate를 유지한다. 준비 워커 시작은 예측 발급 성공과 별도로 평가한다.
