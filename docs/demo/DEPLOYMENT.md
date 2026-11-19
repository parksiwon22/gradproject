# 정적 화면 배포: 소유자에게 필요한 한 번의 설정

웹 화면·실제 저장 예측·오프라인 HTML은 main에 반영되었습니다(PR #6). GitHub Pages 공개 주소는 아직 이전 화면입니다.

현재 연결 OAuth 토큰에 `workflow` scope가 없어 새 workflow push가 GitHub에서 거절되었습니다. 아래 파일은 **실행되지 않는 참고용 텍스트**입니다. 소유자 또는 workflow 권한이 있는 계정이 한 번 등록하면 됩니다.

1. [준비된 배포 파일](./static-pages-workflow.yml.txt)의 전체 내용을 복사합니다.
2. GitHub에서 [새 workflow 파일 만들기](https://github.com/parksiwon22/gradproject/new/main?filename=.github/workflows/smp72-static-pages.yml)를 엽니다.
3. 내용을 붙여넣고 main에 커밋하거나 PR을 병합합니다.
4. Actions → **SMP72 static presentation Pages**가 자동 실행됩니다. 완료되면 https://parksiwon22.github.io/gradproject/ 를 새로고침합니다.

이 작업은 API 키를 사용하지 않습니다. 모델 학습·추론·자동 수집·상태 저장도 하지 않습니다. 저장된 예측자료를 검사한 뒤 docs 파일만 Pages에 게시합니다. 기존 수집 workflow와 같은 `smp-pages` 동시성 그룹을 사용합니다.

기존 **SMP public forecast and Pages refresh**를 수동 실행하는 방식은 화면 게시 외에 과거 v11/v17 수집·예측·상태 저장도 실행하므로 이번 작업에서 실행하지 않았습니다. 해당 작업이 기존 일정대로 돌아갈 경우에도 main/docs의 새 화면이 포함됩니다.

발표용 즉시 실행은 `docs/downloads/smp72-offline.html`을 다운로드한 뒤 Chrome으로 열면 됩니다. GitHub 파일 미리보기는 HTML 앱을 실행하지 않으므로 Raw 다운로드 후 열어주세요.
