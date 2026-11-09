# 전력예보 V20 공개 연구 웹 (GitHub Pages)

기존 `main/app.py`(V5 Streamlit 공개 웹), `app_v20.py`(V20 로컬 연구 웹), 운영 모델, 공식가격 발급 및 Windows 예약작업을 변경하지 않는 **새 웹 프런트엔드**입니다.

## 웹 기능

- D+1 V11 ExtraTrees, D+2/D+3 HGB + seasonal anchor의 **과거 검증 fold 예측값**을 실제 SMP와 전일 가격 기준선에 맞춰 그림
- 선택한 과거 날짜의 24시간 MAE, RMSE, 가장 저렴한 시간, 기준선 대비 MAE 표시
- 차량 배터리 용량, SOC, 충전 전력, 충전 효율, 연결 시간 설정
- 24시간 구간 내 예측가격 저가순 G2V 충전 시뮬레이션 및 실제 SMP 기준 사후 비용, oracle 대비 regret, 즉시 충전 대비 비용 계산
- 총 275개 검증일, 모델 성능표와 연구용 시연 한계 명시

HTML/CSS/JavaScript로 브라우저에서 직접 처리합니다. 과거 시연은 기존 저장소의
`docs/smp_improvement_ml_lgbm/final_fold_demo.csv.gz`를 읽고, 이 브랜치에서는 Actions가 생성하는 `docs/live/latest.json`을 별도 최신 예측 탭에서 읽습니다. 신규 예측은 공식 발급과 분리된 연구용 경로이며 gate를 호출하거나 변경하지 않습니다. 과거 평가 데이터의 최초 수신시각과 당시 모델 존재시각은 미검증입니다.

## 현재 공개와 자동화 배포

기존 공개 URL은 `https://parksiwon22.github.io/gradproject/`입니다. 새 자동화 브랜치는 아직 기본 `main`에 반영되지 않았으므로, 현재 URL에서 신규 최신예측 탭이나 Actions 발급을 production 검증한 상태가 아닙니다.

자동화를 활성화할 때는 이 브랜치를 `main`에 반영한 뒤 **Settings → Pages → Build and deployment → Source → GitHub Actions**로 전환해야 합니다. 아래 관리자 절차를 먼저 완료하고 Actions 수동 실행의 Pages 배포가 성공한 뒤 공개 URL의 `live/latest.json` 및 화면을 확인하세요. V5 Streamlit 웹은 별도로 보존됩니다.

## 로컬 정적 웹 미리보기

저장소 루트에서:
```powershell
python -m http.server 8765 --directory docs
```

접속: `http://localhost:8765/`.

주의: `file:///...`로 HTML을 직접 열면 브라우저가 gzip 파일 fetch를 차단할 수 있으므로 HTTP 서버로 여세요.

## 검증과 한계

현재 UI는 실제 모델 실행물이 아닌, **이미 저장된 275일 fold 결과의 공개 재생**입니다. 실제 발급 예측과 다른 평가집단이므로 운영 성능으로 표시하지 않습니다.

G2V 알고리즘은 1시간 슬롯을 가격 오름차순으로 고르고 마지막 슬롯의 충전량은 부분 시간으로 조절합니다. 충전량 계산은 `배터리 용량×SOC 증가율/충전효율`, 비용은 `sum(SMP×그리드 충전 kWh)+배터리 통과 1kWh당 5원` 가정을 사용합니다. 24시간 밖 밤샘 충전, 가상 V2G 방전, 실차 소매요금 정산은 이 공개 앱에서 지원하지 않습니다.

원본 참고:
- [최종 연구 보고서](../smp_improvement_ml_lgbm/FINAL_RESEARCH_REPORT.md)
- [선정 모델 검증 수치](../smp_improvement_ml_lgbm/final_recommended_models.csv)
- [원본 V20 Streamlit 앱](../../app_v20.py)
# Automated public research forecast (this branch)

The new `docs/live/` view is an optional **research-only** mode. The previous fold-based V20 demo and its MAE/RMSE figures remain intact as a separate tab. This automation does not invoke or relax the V20 official publication gate.

## Server-side run

`.github/workflows/smp-public-live.yml` precollects at 22:45 KST, checks the scheduled origin's target-day official price page, then retains the fixed 23:00 KST issue origin. The public EPSIS input and KPX target publication probes are keyless. PR validation has its own unprivileged job; only the separate deploy job uses the `github-pages` environment and Pages OIDC permissions. The V20 research source is pinned to `fb6fc173d682b5cce2657b65c0b97c11e26ebd32`. No KPX or KMA secrets are required or read. Weather is not a feature of these saved price-only candidates.

The EPSIS SQLite store is cached between Actions runs. It preserves revisions, retrieval (`first_seen`) timestamps and idempotent interval keys. A cache miss triggers the existing collector's historical fill. The workflow records its current run status separately from the last successful forecast. The web never presents a previous run as a current forecast.

## Model and timing limits

`automation/models/` contains new refit weights for V11-style D+1 ExtraTrees and D+2/D+3 HGB residual plus 5% weekly anchor. They were trained through 2026-10-26 from saved land SMP. They are not the historical fold checkpoint files, and the public 275-day CV metrics do not apply to them. The historical source's original first-seen vintages are not verified. Checkpoints are deployed with the workflow code and are not served to the browser.

An issue is created only when the full 23:00 KST input hour and all preceding 168 hourly inputs have `first_seen` no later than the scheduled origin, the KPX target-day availability probe was received before origin, and all 24 features for each horizon are complete. If 23:00 SMP arrives later, the run is withheld; the data is not backdated. If target-day prices were already shown on the official page, that horizon is marked ineligible as a prospective forecast. A negative page probe means only “not observed on that page at probe time,” not proof that no other publication channel existed. GitHub schedule delay is recorded separately from origin and issue time.

D+2/D+3 seasonal anchors use the same-hour median of the prior 7, 14, 21 and 28 days. This matches the final fold study; the `weekly` feature remains a separate 7-day HGB input feature. The committed refit weights are separate model fits through 2026-10-26 and are not the original fold models. Changing the anchor blend does not change the reported fold CV scores.

Until target SMP is observed, post-hoc actual cost and regret cannot be computed for a live forecast. The historical study tab retains the existing same-day retrospective G2V analysis. All displayed costs use wholesale SMP as a research proxy, not retail charging bills.

## One-time repository administrator setup

After merging this branch into the default `main` branch:

1. Open **Settings → Pages**.
2. Under **Build and deployment → Source**, select **GitHub Actions**.
3. Open **Settings → Actions → General → Workflow permissions** and confirm workflows are allowed to run. The workflow requests only `contents: read`, `pages: write`, and `id-token: write`; it does not need a write-capable repository token or API-key secrets.
4. Open **Actions** and confirm the `github-pages` environment exists or is created by the first Pages deployment. Run the workflow once manually, then inspect the run's collection status, prediction status and deployment URL.

Scheduled workflows run only from the default branch. Until this branch is merged and Pages source is switched to Actions, its workflow is not the public site's production updater. The existing branch-based Pages site should not be switched before the Pages workflow is present on `main`.
