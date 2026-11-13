# 기상 데이터 수집 메모

## 구분 및 가용성 규칙

- `forecast` 레코드는 KMA 단기예보의 발행 vintage입니다. `issued_at_kst`, `received_at_utc`, `forecast_origin_kst`, `target_at_kst`, 격자·권역·category·원본 API 출처를 함께 기록합니다. 예측 입력으로 쓰려면 발표와 수신 모두 해당 SMP forecast origin 이전이어야 합니다.
- ASOS는 실제 관측입니다. 관측시각과 실제 수신시각을 별도 보관해야 하며 `forecast` vintage로 재분류하지 않습니다. 관측된 대상일 날씨를 과거 D+3 예보 대신 쓰지 않습니다.
- 기본 HGB와 2026 예측 결과는 SMP/달력 입력만 사용했습니다. 현재 기상 피처가 들어간 최종 모델은 없습니다.

## 권역 표본

`collect_weather.py`는 강릉(ASOS 105), 서울(108), 대전(133), 대구(143), 광주(156), 부산(159)에 대응하는 5km 단기예보 격자 여섯 곳을 사용합니다. 섬 지역 제주를 제외한 지리 분산 샘플이며, 전국 부하량·발전량으로 가중하지 않았고 국토 대표성도 검증하지 않았습니다. 기상 입력 후보의 누락률/상관/계절별 안정성을 확인할 때 쓸 시작점입니다. 지역 평균 또는 aggregate를 모델에 넣으려면 별도 검증이 필요합니다.

## 호출과 저장

- API: `VilageFcstInfoService_2.0/getVilageFcst`, data.go.kr 서비스 ID `15084084`. 공식 설명은 5km 격자, 시간별 상세 단기예보를 글피까지 제공하고 이용료는 무료라고 안내합니다.
- 수집 호출량: 각 발표 vintage당 6개 격자 요청(기본 지역 세트 기준). 응답 페이지당 `numOfRows=1000`을 요청합니다. 이번 실행 환경은 data.go.kr 단기예보용 서비스 키가 없어 실 API 수집, 계정별 요청한도, API 응답 지연 검증은 보류했습니다. 기존 KMA API Hub 경로는 계정 유형에 따라 일반회원 하루 20,000건/5GB(시스템 상황에 따라 변경)라고 안내하지만, 이 한도는 data.go.kr 단기예보 서비스의 쿼터로 간주하지 않습니다.
- 인증: 환경변수 `KMA_DATA_GO_KR_SERVICE_KEY` 또는 `KMA_SERVICE_KEY`. 키는 쿼리로 API에만 전송하고 상태 로그/응답 파일/예외 텍스트로 기록하지 않습니다. Actions에서는 repository/environment secret으로 같은 이름을 설정할 수 있습니다.
- ASOS 조회 API도 별도 `collect_asos_hourly`로 구현되어 있으며 자료는 `docs/live/weather/asos_observations.jsonl`에 `record_type=observation`, 관측시각과 수신시각을 저장합니다. forecast JSONL과 파일을 분리합니다.
- 저장: 기본 forecast 경로 `docs/live/weather/forecast_vintages.jsonl`. 한 줄에 하나의 forecast category/유효시각 값. 인증 키, 요청 URL, 원문 응답은 저장하지 않습니다.
- 예보 범위: KMA 단기예보는 발표시각에 따라 대상 유효기간이 다릅니다. 응답으로 받은 `fcstDate/fcstTime`만 저장하고 기간을 외삽하지 않습니다.
- 결측: 실패한 격자가 있으면 해당 수집을 `failed`로 중지하고 부분 결과를 성공으로 보고하지 않습니다. 모델 측 결측값 보간은 아직 없습니다. 입력이 불완전하면 기상 후보 적용을 보류하고 SMP-only 모델로 계속 동작해야 합니다.

## 과거 데이터와 이번 검증 상태

기존 V11 연구에는 여러 ASOS 관측지점의 과거 관측 자료가 있고, 일부 시점의 기상 파생량이 실험됐습니다. 관측값은 미래 대상일의 사전예보 vintage가 아닙니다. 이전 단기예보 API 접근 기록은 최근/과거 요청 모두 HTTP 403의 서비스 신청/승인 제한이었습니다. 보존된 2026년 과거 단기예보 원본과 각 issue 시점의 수신기록은 확보되지 않았으므로, 과거 관측으로 대신 채운 2026 D+3 평가를 만들지 않았습니다.

원본 API: [KMA 단기예보](https://www.data.go.kr/data/15084084/openapi.do), [KMA ASOS 지상관측](https://www.data.go.kr/data/15139432/openapi.do), [KMA API Hub 이용안내](https://apihub.kma.go.kr/apiInfo.do).
