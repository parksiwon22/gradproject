# 전력예보 V6: 공식 가격 공개를 반영한 예측과 재최적화

수정일 2026-10-14. 추천 연구 방향 C의 최소형을 개발 명세로 반영한 검토안이다. 사용자의 최종 연구 승인/운영 배포를 의미하지 않는다. 이전 문서는 commit d182e6348be0ca6b5b9868c84863ba1968c436a2에 보존된다. 근거 및 A/B/C 비교는 RESEARCH_DIRECTION.md를 읽는다.

## 1. 범위

육지 1시간 가격, EV 1대, 발표 전 다음 날 24시간 AI 예측, 공식 가격 확보 후 남은 구간 재최적화, 가상 V2G 경제성 분석을 구현한다. 연구 질문은 공표 전 예측이 실제 의사결정 비용에 기여하는 조건과, 효율/열화/출발 조건 아래 V2G의 가능한 경제성이다. MAE 개선과 판매수익을 보장하지 않는다.

V5 23시 모델, 원본 source, settings, 2023 저장 MAE 11.093786 및 benchmarks는 그대로 둔다. 현재 환경 11.417825 재현 이슈도 유지한다. 오전 모델은 v6_prepublication 별도 namespace로 만든다. main app.py/기존 outputs를 자동 overwrite하지 않는다.

## 2. 정보시점과 상태

기본 제안 origin은 D-1 09:00 KST다. D는 예측 거래일이며 target interval은 D00~D24이다. 실제 lead time은 15~38시간이고 horizon_index 1~24와 다르다. 시점 변경은 config에 남기고 데이터 공개시각 증거로 적격성을 확인한다. 휴일 등 해당 거래일 가격이 이미 알려진 경우에는 예측 필요 없음으로 분류한다.

상태는 unknown -> forecast_available -> official_received -> replanned 또는 stale이다. 공식 확보 여부는 날짜/시장/상품과 version 및 수신시각으로 결정한다. 정각이 됐다고 공식가격을 사용하지 않는다. 회원 공표, 공개 웹 게시, API 접근을 구분한다. 17/18시 원칙과 23시 API 안내를 같은 시각으로 보지 않는다.

최적화 대상은 현재 연결시점부터 출발까지 최대 48시간이다. 알려진 오늘 가격과 미공개 내일 예측을 슬롯별로 혼합하고 각 price_source를 저장한다. 이미 시행한 슬롯은 고정하고 업데이트시 실제/시뮬레이터 SOC를 초기값으로 사용한다. 미래 최종가격으로 과거 결정을 다시 작성하지 않는다. 모든 공식 가격이 확보된 후 연결한 경우는 AI 증분가치 0 가능성을 대조군으로 평가한다.

## 3. 최소 구성

| 모듈 | 요구사항 |
| --- | --- |
| PriceSnapshotStore | region, product, target intervals, version, issued_at/first_seen/retrieved_at |
| FeatureBuilderV6 | as-of 입력, lead time, lag48/168, calendar, optional verified vintage |
| Forecaster | 단순 profile와 HGB 1 family, validation 선정 |
| ChargingOptimizer | 단방향 LP 및 가상 양방향 MILP, 에너지/출발 제약 |
| ReplayEngine | 시간순 가격 이벤트/차량 연결, 실행한 행동 고정 |
| Evaluator | 비용/throughput/출발SOC/regret/제약 위반/계획 변경 |
| Streamlit V6 | 추천, SOC/충방전 차트, 정보출처 및 시뮬레이션 가정 |

새 코드는 v6/ 또는 src/v6, UI는 app_v6.py를 권장한다. 하루 forecast schema에는 origin,target_date,market_area,market_product,model_version,assumption_status를 둔다. 슬롯에는 interval_start/end,horizon_index,lead_hours,price_source,price,version을 둔다. 시장 상품은 mainland_day_ahead / jeju_day_ahead / jeju_realtime 등으로 분리하되 최소품은 첫 항목만 처리한다.

## 4. 최적화 계약

계통측 kW와 배터리 kWh를 구분한다. E_next=E+eta_c*P_c*dt-P_d*dt/eta_d-E_drive. 용량, SOC 최소/최대, 출발 목표, 연결기간, 출력, 효율, 계통 수출한도를 검사한다. 이진 mode로 동시 충방전을 금지한다. 제어 시간은 1시간/부분시간을 정확히 계산하고 조기 출발/해제 시 fail-safe 결과를 내며 물리제약을 완화해 숨기지 않는다.

목적은 구매비-가상판매대금+배터리 내부 총 throughput 기반 열화비다. 시뮬레이션 비교는 같은 말단 SOC 조건과 이동에너지를 적용한다. V1G real mode는 export_allowed=false, V2G는 simulated_mode에서만 true다. 요금/정산 계약이 없으면 수익을 실현 가능한 현금수익으로 표시하지 않는다.

가상 SMP연동 구매/판매, 고정 소매 구매, 검증한 실제 TOU 구매를 분리한다. 기본 소매요금을 SMP로 묵시 설정하지 않는다. 판매가0/높은 열화에서 방전0도 올바른 결과다. 투자비 제외 운전 마진을 사업 수익률로 부르지 않는다.

## 5. 평가와 축소 범위

공표 전 장시간 연결과 공표 후 야간 연결을 비교한다. 즉시충전, 공식정보 정책, 예측 고정계획, 예측+재최적화와 offline oracle를 같은 제약으로 비교한다. 모델은 validation에서 선택하고 test에서만 최종 확인한다. V5 11.09와 오전 MAE의 직접 우월성 비교는 금지한다.

다년 데이터, LightGBM/CatBoost/XGBoost, 당시 기상예보, D+2 예측, 제주 RT/이중정산, 차량군, 실제 알림/사용자DB/실차는 확장 기능이다. 먼저 B의 단방향 추천을 작동시키고 C의 시간순 연구를 얹는다. API 없어도 권한 확인한 공식 가격 replay와 합성 테스트로 개발하되 실시간 운영이라고 부르지 않는다.

main 병합/공개 배포/실서비스 V2G 제어는 별도 사용자 승인 전 금지한다. 최종 연구 선정, 구매/판매 요금 가정 및 실제 API/데이터 공개는 검토 대상으로 둔다.

## 근거

시장 규칙/회원 공표/API 차이와 V2G 최신 실증은 RESEARCH_DIRECTION.md의 S1~S7, 해당 조문과 미확인 항목을 따른다.
