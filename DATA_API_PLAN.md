# 데이터/API 및 시간 누출 방지 계획

## 1. 현재 확보한 것과 미확보한 것

현재 첨부 ZIP에는 2023-01-01 00:00~12-31 23:00의 8760행, 69열 통합 CSV가 있다. timestamp 중복과 시간 간격 누락은 없다. 원출처별 라이선스, 실제 발표시각, 원시 파일과 당시 예보는 없다. 따라서 연구 재현에 사용하되 운영 시점의 입력 가용성과 전체 재배포 권한은 아직 확인되지 않았다. 공개 저장소에는 이 데이터 및 실제값이 포함된 92일 예측 CSV를 새로 올리지 않는다.

현재 작업에서 기존 첨부 ZIP을 직접 읽어 private_data와 private_reference에 필요한 2개 파일을 연결했다. 두 경로는 gitignore 대상이다. `tools/import_v5_package.py`는 원본 hash를 확인하고 접근 가능한 ZIP에서 같은 작업을 재수행한다. 별도 Codex 환경이 이 첨부파일을 자동으로 볼 수 있다고 가정하지 않는다. 접근할 수 없다면 인증된 비공개 파일 전달/마운트가 다음 선행조건이며 공개 Git 전송으로 우회하지 않는다.

연구 저장소에는 원시 관측, 당시 예보, 정정 이력, 출처/라이선스, publication 시각이 필요하다. 서비스 저장소에는 최신 허용 입력과 모델, 24시간 결과, 추천과 동의한 알림 상태만 필요하다. 원시 자료와 개인정보를 서비스 공개 CSV에 넣지 않는다.

## 2. 데이터 확장

| 항목 | 우선 출처 | 확인할 조건 | cutoff 처리 |
| --- | --- | --- | --- |
| 시간별 SMP 및 수요예측 | KPX 신규 API, EPSIS 과거 공개자료 | 시장 지역, 시간 정의, 갱신일, 원 단위 | 발표 당시 버전만 |
| 실제 전력수요/예비력 | KPX 공식 수급 자료 | 시간별인지 일별인지, 정정/공개 지연 | available_at 기준 lag |
| 관측 기상 | 기상청 ASOS/AWS | 결측 코드, 지점 이동, 시간 기준 | 발표 지연 반영 |
| 다음 날 기상예보 | 기상청 단기예보/API허브 및 예보자료 아카이브 | issue-time, 유효시각, 격자, 보존기간 | 당시 확보한 최신 vintage |
| 신재생 발전량 | KPX 및 원 ZIP의 기관 원출처 | MWh/MW, 지역/기간 중복 | 지연 확인 전 제외 |
| LNG와 가스 단가 | 원자료 제공기관 확인 후 도입 | 월별 단가/일별 거래량 여부 | 당월 사후 확정치를 당월 초에 적용 금지 |
| 공휴일 | 공식 공휴일 캘린더/관보 | 임시공휴일 지정 발표일 | 발표 이후에만 알려진 정보 |

2021~2025와 2026 완료분을 후보로 조사한다. CSV 출처별 재수집이 가능하지 않으면 SMP와 캘린더만 사용한 축소 모델로 시작하고 변수 누락을 기록한다. 모델 비교는 공통 표본에서 수행한다. 확장한 데이터가 정확도를 높인다고 가정하지 않는다.

## 3. KPX API

신규 공식 목록: https://www.data.go.kr/data/15131225/openapi.do
이 안내는 육지/제주의 시간별 SMP 및 수요예측, 하루 한 번 23시경 갱신, 거래시간이 구간 끝점이라는 점을 명시한다. 구 API(https://www.data.go.kr/data/15076302/openapi.do)는 폐지 예정이며 시간 0을 00:00~01:00 시작 구간으로 설명한다. 두 API의 숫자 시간을 동일하게 join하면 1시간 오정렬이 생길 수 있다.

신규 응답의 거래시간 H는 공식 정의에 따라 `[trade_date + (H-1) hours, trade_date + H hours)`로 변환하되 실제 허용 hour 범위와 24시 표기를 명세/샘플로 확인한다. 원 raw 필드와 interval_start/end를 모두 보존한다. V5 웹의 `hour=24` for 00:00은 UI 호환 표시일 뿐 신규 KPX 시간 매핑에 재사용하지 않는다. 23:00 관측이 해당 시각에 이미 완료된 관측인지, 23~24시 구간 값인지도 확인한다.

신규 endpoint/필수 파라미터는 목록의 최신 Swagger와 키 발급 후 확인한다. 이번 조회 화면만으로 신규 endpoint를 확정할 수 없었으므로 존재하지 않는 URL을 발명하지 않는다. 인증 호출은 아직 수행하지 않았다. API 계약 테스트용 정제 샘플을 확보한 후 별도 client를 구현한다. 개발 호출한도는 안내상 100이며 소규모 캐시와 호출 예산을 둔다. 실제 승인 계정의 한도가 우선이다.

거래일을 포함해 전일 저녁부터 다음 날 새벽까지 응답 갱신을 관찰하고 `first_seen_at`을 수집한다. 공식 시장 공표시각과 이 API 갱신시각은 다를 수 있다. 거래전일 23시 당시 다음 날 가격이 확인되면 운영에는 공식 가격을 우선 제공하고 연구 예측과 구분한다. 정확한 최신 시장 공표 규칙은 별도 확인해야 한다. 다른 해의 사후조회 응답만으로 2023년 당시 공개시각을 확정하지 않는다.

## 4. 기상청 단기예보 연동

공식 서비스: https://www.data.go.kr/data/15084084/openapi.do
후보 요청: `https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst`.
키는 환경변수 `KMA_SERVICE_KEY` 또는 운영 secret store에만 둔다. JSON, base_date/base_time, nx/ny, pageNo/numOfRows를 사용하고 totalCount를 확인해 모든 페이지를 수집한다. 인증키를 URL 로그와 예외 문자열에서 마스킹한다.

원시 스키마는 `source, base_date, base_time, issued_at, first_seen_at, received_at, fcstDate, fcstTime, valid_at, nx, ny, category, fcstValue, revision_id, payload_hash`로 저장한다. TMP/REH/WSD/POP/PTY/SKY/PCP 등을 후보로 하되 최신 2609 활용가이드와 실제 응답으로 코드 의미와 단위를 확정한다. PCP의 문자열, 구간값, 강수없음과 결측은 구분한다. 모호한 정성값을 무조건 0으로 바꾸지 않는다. 일사량/발전량이 단기예보에 그대로 있다고 가정하지 않는다.

23:00 cutoff의 기본 후보는 이전 20시 발표본이다. 23시 발표본은 timestamp 표기가 23시라고 해서 23:00 정각에 확보됐다고 간주하지 않는다. `issued_at <= cutoff AND available_at <= cutoff`인 자료 중 최신 버전을 선택한다. 서비스가 지연 실행돼도 23시 이후 자료를 23시 예측 입력에 끼워 넣지 않는다. cutoff를 늦출 경우 새 origin과 새로운 실험으로 기록한다. 정규 발표시각은 최신 공식 가이드에서 다시 확인하며 발표지연을 운영 관측으로 보정한다.

우선 서울 격자 후보를 검증하고 수요권역별 지역 가중 평균을 확장 실험으로 둔다. 국가 수요와 한 도시 관측을 혼합한 V5를 전국 최적 특성으로 가정하지 않는다. 가중치는 train에서 정하거나 사전에 고정한다.

과거 예보는 최신 실시간 API에서 2023년까지 받을 수 있다고 가정하지 않는다. 기상자료개방포털과 API허브의 발표시각이 보존된 예보 아카이브를 조사한다. 과거 3시간 예보와 현재 1시간 예보의 변경도 기록한다. 당시 예보 미확보 기간은 예보 없이 학습하거나 shadow 수집부터 시작한다. 실제 관측 날씨를 미래 예보 대신 넣은 실험은 oracle upper-bound로 따로 표시한다.

## 5. as-of 데이터 계약

일반 테이블은 `source, variable, market_area/station/grid, event_start, event_end, issued_at, available_at, retrieved_at, value, unit, revision_id, quality_flag, source_hash, license_ref`를 가진다. timezone-aware Asia/Seoul로 계산하고 UTC를 저장 병기한다.

관측 입력은 `event_end <= origin AND available_at <= origin`을 만족해야 한다. 예보는 유효시각이 미래여도 `issued_at <= origin AND available_at <= origin`이면 사용 가능하다. 공식 알려진 미래가격은 연구 타깃 특성에서 금지하고 운영 price_source에만 사용한다. 과거 정정본은 당시 사용 가능한 revision으로 복원한다. 최초 다운로드 시각은 오래된 자료의 실제 publication 시각 증거가 아니다.

V5는 timestamp가 과거인지를 확인할 뿐 available_at을 저장하지 않는다. 일별 reserve/LNG 값은 하루 24행 반복이 확인되었으나 원 단위/발표시각이 불명확하다. 일별 사후 집계를 23시 입력으로 사용할 수 있는지 검증해야 한다. 확인 전 신규 모델에서는 보수적 publication lag 또는 제외를 실험한다. V5 원본을 소급 변경하지 않는다.

ffill은 변수별 max_age를 두고 stale flag를 남긴다. 특히 강수의 결측이 무강수 코드인지 확인하지 않고 장기 forward-fill하지 않는다. 풍력 결측 3600행, 강수 결측 7673행, 일사/일조 결측 3979행은 의미별 처리가 필요하다. 0 대치가 가능한 관측과 수집 실패는 구분한다. rolling/lag는 timestamp의 연속성을 확인한 뒤 계산하며 정규화 gap에 가짜 관측을 생성하지 않는다.

## 6. 실행, 실패, 저장

수집은 짧은 HTTP timeout과 재시도 3회, exponential backoff+jitter, 결과코드/스키마 검사를 사용한다. 200 HTTP라도 실패코드 또는 빈 응답이면 성공으로 처리하지 않는다. 일일 호출 예산과 rate limit을 확인한다. 원시 데이터를 append-only로 보관하고 동일 payload hash는 중복 기록하지 않는다.

매일 `origin=23:00 KST` 기준 스냅샷을 동결하고 23:10 실행을 기본 제안으로 둔다. 지연 실행은 cutoff를 바꾸지 않는다. 공식 가격을 그 시각에 따로 확인하고 조회시각을 UI에 표시한다. 매일 모델 추론만 하고 월별/수동 재학습은 검증 후 승격한다. 24행/단위/market_area/날짜/유한값 검사를 통과한 파일만 임시 경로에서 atomic rename한다. 실패 시 직전 성공본을 유지하고 stale 표시한다.

저비용 개발 스케줄러 후보는 GitHub Actions지만 정확한 실행시각 보장이 없고 schedule은 기본 브랜치에서 동작한다. 사용자 승인 전 main에 workflow를 활성화하지 않는다. 이후 실행 지연을 허용하면 Actions, 정확성이 필요하면 서버 cron/외부 scheduler를 검토한다. Streamlit session timer에 수집을 맡기지 않는다.

알림은 `user_id,target_date,forecast_version,alert_type` 중복키와 outbox로 재전송을 제어한다. 초기 dry-run 기록만 구현한다. 사용자 선택 채널과 수신 동의 후 전달하고 야간 알림/철회/실패를 처리한다. 이번 작업은 실제 정기 실행이나 전송을 활성화하지 않았다.

## 7. 계약 테스트와 공개 검토

미래 발행 예보 제외, cutoff 뒤 revision 제외, 동결 origin의 지연 실행, 시간 끝점/시작점 변환, 하루 24개 구간, 단위, 시장 구분, 페이지 누락, 중복/결측, 429/timeout/invalid key/빈 응답, stale fallback, 원자적 저장, 알림 중복을 테스트한다. fixture는 합성 또는 권한 확인된 정제 샘플만 공개한다.

데이터별 출처 URL, 저작권/재배포 조건, 기간, 공개 승인자를 기록한다. 기상청 서비스는 출처표시 조건을 안내하지만 그것만으로 ZIP의 69열 전체 공개 권한이 확정되지 않는다. 인증키 발급/계정 로그인과 공개 승인만 사용자가 처리할 선행조건으로 남긴다.

## 공식 참고

- KPX 신규 API: https://www.data.go.kr/data/15131225/openapi.do
- 구 API의 폐지/시간 정의: https://www.data.go.kr/data/15076302/openapi.do
- KMA 서비스와 최신 가이드: https://www.data.go.kr/data/15084084/openapi.do
- KMA 예보 아카이브: https://data.kma.go.kr/data/rmt/rmtList.do?code=420&pgmNo=572
- KMA API허브: https://apihub.kma.go.kr/
- Actions schedule 제한: https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule
