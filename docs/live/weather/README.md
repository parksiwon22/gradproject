# 기상 데이터 수집 메모

## 구분 및 가용성 규칙

- `forecast`는 KMA API Hub 단기예보 vintage입니다. 발표시각, 응답 수신시각, SMP 예측 원점, 대상 유효시각, 격자·권역·category·API 경로를 보존합니다. 입력 자격은 예보 발표와 수신이 모두 SMP 원점 이전일 때만 인정합니다.
- ASOS는 관측 자료입니다. 관측시각과 수신시각을 별도 저장하며 예보 vintage로 취급하지 않습니다. 대상일의 사후 관측으로 과거 D+3 예보를 대체하지 않습니다.
- 기존 HGB와 보존된 2026 예측은 SMP/달력 변수만 사용합니다. 기상 피처 모델은 아직 없습니다.

## 권역 표본

수집기는 강릉(ASOS 105), 서울(108), 대전(133), 대구(143), 광주(156), 부산(159)의 육지 격자 여섯 곳을 사용합니다. 지리 분산 탐색 표본이며 전국 부하량·발전량 가중 대표값으로 검증하지 않았습니다. 모델 입력으로 승격하려면 추가 공간 대표성 검토가 필요합니다.

## API, 저장 및 Actions

- 단기예보 API Hub 경로: `typ02/openApi/VilageFcstInfoService_2.0/getVilageFcst`; ASOS 시간 자료 경로: `typ01/url/kma_sfctm3.php`.
- 인증은 환경변수 `KMA_APIHUB_AUTH_KEY`만 사용합니다. 데이터.go.kr `serviceKey`와 혼용하지 않습니다. GitHub Actions에서는 `KMA_APIHUB_AUTH_KEY` repository 또는 environment secret을 전달합니다.
- 하루의 예보 수집은 격자별 6회, ASOS 기간 수집은 6개 지점 지정 1회입니다. 응답 실패/권한 거부는 인증키나 응답 본문을 노출하지 않고 안전한 상태 사유만 남깁니다. 예보와 관측은 한 API 실패가 다른 요청을 막지 않도록 별도로 처리합니다.
- 2026-11-14 실제 호출: 기존 API Hub 키로 ASOS 기간 API가 HTTP 200을 반환했고 6개 지점 합계 144 관측행을 새 collector가 파싱했습니다. 단기예보 API Hub 경로는 HTTP 403이며 안전하게 `service_application_or_approval`로 분류됐습니다. 동일한 자격증명이 ASOS에서 승인된 사실로 키 자체의 무효 문제와 예보 서비스별 접근 제한을 구분했습니다. 응답 본문은 보존/출력하지 않았습니다.
- Actions의 비공개 저장 위치는 `automation/state/weather/forecast_vintages.jsonl` 및 `asos_observations.jsonl`이며, `smp-live-data` 상태 브랜치에 이어 저장합니다. 공개 Pages 자료에는 날씨 원자료를 넣지 않습니다.
- 2026년 당시 발행된 단기예보 원본이 없어 2026 D+3의 기상 모델 평가는 보류합니다. 실제 Actions 수집 결과가 생긴 뒤 서비스별 승인과 결측률을 확인해야 합니다.

공식 안내: [KMA 단기예보 API](https://www.data.go.kr/data/15084084/openapi.do), [KMA ASOS API](https://www.data.go.kr/data/15139432/openapi.do), [KMA API Hub](https://apihub.kma.go.kr/apiList.do).
