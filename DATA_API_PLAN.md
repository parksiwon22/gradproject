# V6 가격 공개시각, API 및 누출 방지

수정 2026-10-14. 적용은 추천 C 최소형이다. 공식 시장 사실과 원문 위치는 RESEARCH_DIRECTION.md를 따른다.

## 1. 시간과 접근을 구분

가격 산정/시장회원 공표/일반 공개웹 게시/공공 API 수신은 별개 이벤트다. 육지17시 및 제주DA18시 원칙은 회원 통지 절차와 읽어야 하며 학생 공개웹 접근권한/정확한 최초 게시를 자동으로 보장하지 않는다. 신규 API의23시경 갱신은 첫 가격결정시각이 아니다. 장애/휴일/여러날 선공개/정정version을 기록한다.

신규 API https://www.data.go.kr/data/15131225/openapi.do 는 육지/제주 시간별 SMP 및 수요예측을 안내하며 거래시간은 구간 끝점이다. 구 API https://www.data.go.kr/data/15076302/openapi.do 는 폐지 예정이며0시가 시작구간이다. endpoint/파라미터/실제 tradHour범위는 최신 Swagger와 키 승인 후 샘플로 확인한다. 이번 조사에서 인증 호출은 하지 않았고 신규 endpoint를 발명하지 않았다.

제주DA와 RT를 region 코드 하나로 합치지 않는다. RT 임시와 확정가격은 다른 product/status다. 공개 시간별 SMP API를 RT 15분 데이터라고 쓰지 않는다. 최소품은 mainland_day_ahead만 소비하고 미확인 시장 CSV는 market_area=unverified로 유지한다.

## 2. 가격 스냅샷 계약

`market_area, market_product, target_date, interval_start, interval_end, unit, price, source, source_url, version, payload_hash, issued_at, market_published_at, public_first_seen_at, service_first_seen_at, retrieved_at, timing_evidence, is_corrected, quality_flag`.

처음 다운로드한 역사자료의 retrieved_at을 당시 publication으로 바꾸지 않는다. log가 없으면 available_at는 unknown이며 rule_assumed 또는 schedule_assumed로 표시한다. 회원 공표시각을 공개웹first_seen으로 복사하지 않는다. origin<=issued/available조건을 별도로 검사하며 미래 정정본은 당시 초기값을 대체하지 못한다.

모든 datetime은 timezone-aware Asia/Seoul로 계산하고 저장에 UTC를 병기한다. time-ending H가1~24인지 인증샘플로 확인한 뒤 start=tradeDate+(H-1)h,end=tradeDate+Hh로 정렬한다. V5의00시를hour24로 표시하는 UI 규칙을 가격 interval 변환에 쓰지 않는다. 원시 시간필드를 남기고 API family별 adapter를 둔다.

## 3. 오전 예측 입력

target D, proposed origin D-1 09:00, next-day24 slots. horizon_index1~24,lead_hours15~38을 별도로 둔다. 관측은 event_end<=origin AND available_at<=origin, 예보는 issued_at<=origin AND available_at<=origin이며 valid_at는 미래일 수 있다.

가장 중요한 누출 차단은 원 V5의target-24h값이다. 오전에는 D-1 후반시간의 관측이 미래여서 전일완성24h profile로 사용하면 안 된다. 최소 모델은target-48h 및target-168h를 사용하고 시점/자료가용성을 확인한다. 이미 공개한 오늘D-1가격은 당시확보version을 따로 제공할 수 있으나 내일 타깃 가격이 이미 확보돼 있으면 prediction eligibility=false이다.

원 2023CSV는8760행69열이며 원발표로그/시장 구분/재배포권한이 미확인이다. 학습과 모델자료는 private_data/private_reference를 유지한다. timestamp연속만으로 publication leakage가 없다고 인증하지 않는다. 미확인일별 예비력/연료/사후기상은 첫 V6운영모델에서 제외한다. 역사 최소SMP/calendar 실험도 데이터 가정과출처 확인상태를 결과에 붙인다.

## 4. Replay로 먼저 개발

원본 private ZIP는 tools/import_v5_package.py로 직접 읽어 필요한 입력을 연결한다. Codex 새환경에서 첨부가 실제 보이는지 확인하고 비공개전달/마운트를 먼저 시도한다. 사용자에게 수동복사를 요구하거나 공개 Git에 raw자료를 넣지 않는다.

가격 이벤트 레코드와 EV연결이벤트를 시간순으로 처리한다. step이 official_received이면 해당 상품/거래일/버전의24개가격을 검증하고 앞으로의 계획만 갱신한다. 공식가격 없는 슬롯은 모델값으로 유지하고 모델의존 비율을 기록한다. 실행한 슬롯과 SOC trace는 append-only로 보존한다. historical final price를 initial origin에 주입하는 작업은 offline oracle에서만 허용한다.

실제first_seen 로그 없이17시/23시 이벤트를 생성하면 공표시각 가정 민감도 실험이다. 실제23시 API 접근과 그보다 앞선 회원 공표의 차이를 '불법누출 없는 미래예측 가치'로 뭉뚱그리지 않는다. 2일뒤 선공개를 만나면 해당가격도 known으로 처리한다.

공개데이터키 미확보 시 합성fixture와 공개권한 확인한 snapshot replay로 최소품을 개발한다. API 없이 실시간 운영이라 주장하지 않는다. 17시정각에 데이터가 없는 이유로 실제delay값을 발명하지 않는다.

## 5. 다음 단계의 수집

키가 확보되면 timeout/3회retry/backoff/응답코드/페이지와 호출예산/키마스킹/24구간검사를 포함한 client를 구현한다. 개발계정 안내100회 예산에 맞춰 한 거래일당 최초확인 시점 탐색은 좁은시간창과캐시를 사용한다. 가격DB에 원시payload hash를 저장하고 수정본을append한다. 개인정보와 인증키는Git/사용자화면/로그에 넣지 않는다.

기상청 단기예보: https://www.data.go.kr/data/15084084/openapi.do . origin09시에서 어떤발표본이 실제확보되는지는first_seen과vintage로 선택하며 최근표시8시예보가 즉시확보됐다고 가정하지 않는다. history실제기상으로 미래예보를 대체하지 않는다. 예보archive/detailedregion/다년데이터 확보는 확장실험이며 최소품의선행조건으로 강제하지 않는다.

일정자동화는 추후CLI와scheduler를 분리한다. 현재는 manual/replay만. 실제정기작업, 사용자알림전송, mainworkflow활성화는 하지 않는다.

## 6. 테스트와 완료조건

cutoff후가격/예보/revision 거부, market/product분리, 알려진타깃적격성 제외, holiday다일공개,1h/15min혼용거부,time-ending자정/24시,단위MW/kW/MWh/kWh,결측/중복/NaN/coverage,이미실행한계획불변,correctedprice미래구간만재계획,stalefallback,API실패/빈결과/페이지누락과키마스킹을 확인한다. 즉시충전및예측정책의평가가격은같은종류를사용한다. 공식DA가격실험을개인실정산수익실험이라고부르지않는다.
