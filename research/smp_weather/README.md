# SMP + KMA 예보 HGB 후보

## 현재 상태

기존 SMP-only D+3 연구 모델과 보존된 316일 예측을 기준모델로 유지합니다. Weather HGB는 별도 코드와 `research/smp_weather/models/` 경로만 사용하며, 자동 운영 모델/공식 발급 gate에 연결하지 않습니다.

기존 API Hub 인증변수 `KMA_APIHUB_AUTH_KEY`는 현재 실행환경에서 읽혔습니다. 기존 키로 2026-11-14 API Hub ASOS 기간조회가 HTTP 200을 반환했고 6개 지점 144 관측행을 파싱했습니다. 관측은 미래예보 대체값으로 쓰지 않습니다.

동일 키의 단기예보 호출은 다음과 같이 모두 403 서비스 신청/승인 제한을 반환했습니다.

- 현재형 격자 API `typ01/cgi-bin/url/nph-dfs_shrt_grd`
- 최근/과거 발표시각을 지정한 호환 OpenAPI `typ02/openApi/VilageFcstInfoService_2.0/getVilageFcst`

ASOS 성공으로 키 전체의 무효나 Windows 환경변수 미전달 문제와는 구별됩니다. 현재는 단기예보 API별 신청/승인 또는 계정 권한 범위 문제로 판정했습니다. 과거 2026-11-10 20시 발행의 D+3 원본이나 2023~2025 당시 예보 기록은 확보되지 않았습니다. 따라서 실기상 모델 학습과 2026 paired 성능점수는 보류 상태입니다. 관측소별 ASOS 데이터는 기존 비공개 V11 자료에 있으나 본 후보의 예보 피처로 쓰지 않습니다.

기상청 문서는 단기 격자 자료에서 TMP(기온), REH(상대습도), WSD(풍속), POP(강수확률) 등을 제공하고, 매일 8회 발표되는 발행시각 `tmfc`와 대상시각 `tmef`를 구분합니다. 이번 후보는 이 네 변수를 6개 육지 권역 각각 별도 피처로 보관하며 권역 평균을 전국 대표값으로 가정하지 않습니다. 발표/수신/대상 시간이 갖춰진 완전한 24시간 예보만 사용합니다.

## 모델 경로와 재현

- SMP-only 보존 결과: `docs/live/walkforward_d3_2026.json.gz` (읽기 전용 비교 입력)
- 기상 입력 원본: 비공개 `automation/state/weather/forecast_vintages.jsonl`
- 기상 피처 builder: `weather_features.py`
- Weather HGB 학습기: `train_weather_hgb.py`
- 시간순 paired 평가기: `walkforward_weather.py`
- Weather 체크포인트/매니페스트: `models/` (검증된 vintage 데이터가 모이기 전에는 생성하지 않음)
- Weather 전용 결과: `results/walkforward_d3_2026.json`

Windows PowerShell, 저장소 루트에서 실행합니다.

```powershell
$env:SMP_SOURCE = (Resolve-Path ..\gradproject).Path
python -m research.smp_weather.walkforward_weather
python -m research.smp_weather.train_weather_hgb --train-before 2026-01-01
```

현재 데이터 상태에서 walk-forward 출력은 `blocked_no_forecast_vintages`이며 모든 316개 날짜의 제외 사유를 기록하고 가격/EV 점수는 `null`로 둡니다. 학습 명령은 예측시점 이전에 수신된 완전한 예보 학습일이 365일 미만이면 중단하고 체크포인트를 쓰지 않습니다.

각 대상일의 학습행은 그 행 자체의 D-3 23시 입력 원점에서 다시 구성합니다. 평가용 주간 체크포인트는 해당 ISO 주 월요일 23시에 학습하고, 월요일 전날까지의 라벨만 사용합니다. 따라서 그 주의 가장 이른 예측 원점보다 학습시각이 늦거나, 학습 라벨이 fit date와 같거나 이후인 경우 평가에서 제외합니다. 관측(`record_type=observation`), 예보 발행 후에 수신된 데이터, 예보 발행 원점 이후 자료, 결측 예보는 학습 및 평가에서 제외합니다. 주별 HGB는 기존과 같은 고정 파라미터(250회, 잎 31, 학습률 0.05, L2 2.0, seed 42)로 학습하고 5% 7/14/21/28일 seasonal anchor와 혼합합니다. 결과는 보존 기준모델과 정확히 같은 대상일의 교집합에서만 MAE/RMSE, 고가 구간 MAE, 최저가 6시간 중첩률, 동일 EV 조건 G2V/V2G regret을 계산합니다.

## 자동수집

기존 `.github/workflows/smp-public-live.yml`은 `secrets.KMA_APIHUB_AUTH_KEY`를 수집기에 전달하고, 결과를 공개 Pages가 아닌 `smp-live-data` 비공개 상태 경로에 누적합니다. ASOS와 단기예보는 별도 요청으로 처리하므로 예보 403이 SMP-only 수집과 예측을 막지 않습니다. 로컬 Windows 환경변수는 GitHub에 자동 복사되지 않으므로 repository/environment secret에 기존 키를 별도로 등록해야 합니다. API Hub 예보 서비스 승인이 해결되기 전에는 Actions에서도 ASOS만 수집될 수 있습니다.

기존 SMP-only 모델, 2024~2025 연구 결과, 2026 예측벡터, 성능 수치, 웹 파일은 이 후보에서 수정하지 않습니다.

공식 명세: [KMA API Hub 단기 격자예보](https://apihub.kma.go.kr/apiList.do?seqApi=10), [KMA API Hub 이용안내](https://apihub.kma.go.kr/apiInfo.do).
