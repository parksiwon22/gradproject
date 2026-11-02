# 전력예보 V20 공개 연구 웹 (GitHub Pages)

기존 `main/app.py`(V5 Streamlit 공개 웹), `app_v20.py`(V20 로컬 연구 웹), 운영 모델, 공식가격 발급 및 Windows 예약작업을 변경하지 않는 **새 웹 프런트엔드**입니다.

## 웹 기능

- D+1 V11 ExtraTrees, D+2/D+3 HGB + seasonal anchor의 **과거 검증 fold 예측값**을 실제 SMP와 전일 가격 기준선에 맞춰 그림
- 선택한 과거 날짜의 24시간 MAE, RMSE, 가장 저렴한 시간, 기준선 대비 MAE 표시
- 차량 배터리 용량, SOC, 충전 전력, 충전 효율, 연결 시간 설정
- 24시간 구간 내 예측가격 저가순 G2V 충전 시뮬레이션 및 실제 SMP 기준 사후 비용, oracle 대비 regret, 즉시 충전 대비 비용 계산
- 총 275개 검증일, 모델 성능표와 연구용 시연 한계 명시

HTML/CSS/JavaScript로 브라우저에서 직접 처리합니다. 데이터는 기존 저장소의
`docs/smp_improvement_ml_lgbm/final_fold_demo.csv.gz`(GitHub 추적)을 GitHub Pages에서 읽습니다. **재학습, 신규 예측, 실제 운영 발급, 원본 SQLite 실시간 조회는 구현하지 않았습니다.** 과거 평가 데이터의 최초 수신시각과 당시 모델 존재시각은 미검증입니다.

## 공개 설정 방법 (계정 소유자가 1회 실행)

GitHub 저장소 `parksiwon22/gradproject` → **Settings → Pages** → Build and deployment의 Source **Deploy from a branch**를 선택합니다.

- Branch: `web/v20-public-20261102`
- Folder: `/docs`
- Save

발행이 활성화되면 일반적으로 `https://parksiwon22.github.io/gradproject/` 형식입니다. **실제 웹이 준비됐는지는 Pages 화면의 배포 결과를 확인해야 합니다.** 기존 `https://power-forecast-v5.streamlit.app/` 웹은 별도로 보존됩니다.

Repository 설정에 대한 관리 권한이 없어 자동 활성화하지 못했습니다. 기존 Pages 발행 설정이 있는 경우에는 충돌 여부를 먼저 확인해 주세요.

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
