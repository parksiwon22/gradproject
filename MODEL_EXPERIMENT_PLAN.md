# V6 발표 전 가격 예측과 의사결정 실험

수정 2026-10-14. 이 명세는 추천 C안의 실험 계획이며 이미 개선된 결과가 아니다. 원 V5 결과는 PROJECT_AUDIT.md와 benchmarks/v5를 따른다.

## 1. 분리된 세 기준

V5-archive는 D-1 23시 정보의 저장 결과 MAE11.093786, F1 test .277512를 보존한다. V5-current-environment는 재실행11.417825이며 원환경 차이가 미해결이다. V6-prepublication은 D-1 09시 제안 origin에 재학습한 새 모델이다. 처음 두 결과를 마지막 모델의 같은 난이도 baseline으로 쓰지 않는다.

V6 비교 기준은 동일 09시 정보로 만든 same-hour lag48, weekly lag168, validation에서 혼합비를 선택한 profile다. HGB의 학습/추론 계약은 같은 target 24행과 lead_hours15~38이다. 종전 V5 target-24h profile 중 origin 이후 D-1 09~23시 정보는 금지한다. target-48h, target-168h의 실제 시각과 publication을 확인한다. cutoff 이전 알려진 오늘 하루전 가격은 별도 officially_known feature family로만 사용하고 알려진 내일 타깃은 예측 적격성에서 제외한다.

## 2. 최소 모델과 데이터

첫 비교는 HGB 1 family와 profile로 제한한다. 손실 squared_error/absolute_error, leaf15/31, iteration200/400 등 최대8 조합과 seed42를 기본 제안으로 한다. V5 source는 수정하지 않는다. 데이터는 SMP와 calendar부터 시작하며 publication 확인이 안 된 일별 reserve/LNG/사후 weather 값을 억지로 넣지 않는다. 미래 날씨는 당시 forecast vintage를 얻은 공통기간에서만 추가한다.

2023 데이터는 구조검증 및 historical-assumption replay에 재사용할 수 있지만, V5 test가 이미 검토된 사실과 원 시장구분/publication 미확인을 표시한다. 2023년 결과를 untouched prospective test나 2026 제주 RT 결과로 부르지 않는다. 신규 공개권한/품질이 확인된 기간의 최근3개월을 test로 동결하고 이전 기간에서 월별 expanding folds를 구성한다. 미열람 기간이 없으면 결과는 탐색 평가로 표시한다.

각 날짜24행은 같은 split에 둔다. label interval_end와 publication은 모델 fit cutoff를 넘지 않는다. 전처리/분위수/혼합비/calibration/threshold는 fit/validation에서만 정한다. early stopping은 chronological validation을 명시하거나 끄고 iteration을 검증으로 고른다. 가짜 역보간/미래 실제치로 결측이나 과거 예보를 채우지 않는다.

## 3. 조건별 정책 실험

| 정책 | 정보/행동 규칙 |
| --- | --- |
| P0 즉시충전 | 연결 시 목표까지 가능한 출력으로 충전, 방전 없음 |
| P1 공식정보만 | 미공개 미래가격을 모르는 정의된 myopic/즉시 fallback, 공식 확보 후 최적화 |
| P2 오전 예측 고정 | origin의 알려진 오늘가격+내일예측으로 계획, 출발/물리조건 변화 외 가격갱신 없음 |
| P3 예측+재최적화 | P2와 같은 초기정보, 새 공식 version 수신시 남은 슬롯만 재계획 |
| Oracle | 최초부터 전체 미래 평가가격을 아는 offline 하한, 실행정책으로 사용 금지 |

P1을 의도적으로 충전 못 하게 기다리게 만든 뒤 AI 승리라고 해석하지 않는다. 미공개 구간의 fallback를 사전 고정하고, 필요하면 즉시충전 fallback와 오늘 알려진 가격 내 myopic 정책을 각각 제시한다. 모든 정책은 같은 연결기간/출발SOC/말단에너지/효율/열화/실행빈도/수출한도를 갖는다. 계획이 바뀌기 전 실제 시행한 출력은 반드시 SOC에 반영한다.

시나리오 S1: 공표 전부터 연결, 출발 다음 날 오전. S2: 공식 수신 후22시 연결, 출발 다음 날08시. S3: 공식값 지연/누락. S4: 조기 출발 또는 연결 중단. 날짜간 가격범위를 충분히 확보하며 없는 날 가격을 같은 날짜00시로 이어붙이지 않는다.

오전에 내일가격을 예측했더라도 실행이 공식 수신 후만이면 P3와 P1이 같을 수 있다. 이것을 'AI 가치가 없음/제한됨' 대조 결과로 보고한다. 재최적화는 P2보다 좋아도 P1보다 좋아야 AI의 증분가치를 주장할 근거가 생긴다. 실제 API 수신 로그가 없으면 17/23시 공개 이벤트는 가정 민감도이며 실측 효과로 부르지 않는다.

## 4. 지표

모델 MAE/RMSE/bias/horizon 및 피크 conditional MAE, PR-AUC/F1/recall/calibration을 보고한다. 비용은 실제 평가가격에서의 구매비, 가상판매대금, 열화비, 순운전비, P0/P1 대비 차이, oracle regret다. 제약 위반수, 출발SOC 실패율, 불가능 요청비율, 열화 throughput, 실행시간, 실행 전후 계획변경량도 보고한다.

같은 날짜/차량조건의 paired 차이를 집계하고 7일 block-bootstrap 95% CI를 선택 분석으로 둔다. 초기 소규모 탐색에서 복잡한 유의성 주장보다 개별 시나리오와 실패 원인을 우선한다. 공식가격이 업데이트된 후의 미래 행동은 새 version으로 계획하되 과거 비용은 당시의 확정된 평가정책으로 평가한다. snapshot 정정 실험과 실제 정산가격 실험은 구분한다.

모델 선정은 validation MAE를 기본으로 하되 regret와 목표SOC 결과를 함께 검토한다. 이전 2%/5% gate를 다른 cutoff의 V5 숫자에 적용하지 않는다. 개선 없으면 오전 profile을 유지해도 C의 정보공개/경제성 연구는 수행할 수 있다. test를 보고 후보/요금/threshold를 바꾸면 탐색으로 재분류한다.

## 5. 피크, 장기 및 family 확장

피크 위험은 발표 전 미공개가격에 대한 보조 출력이다. 발표 후 알려진 고가시간은 실제 가격 경고이지 예측확률이 아니다. class balancing score를 calibration 없이 발생확률로 표시하지 않는다. 최소품 이후 train-only peak weights1/2/4와 quantile를 ablation하며 평균 MAE/regret 악화를 함께 보고한다.

LightGBM/CatBoost/XGBoost는 같은 09시 schema/표본의 optional family다. 2일 이상은 target_date/lead시간/휴일 선공개 적격성을 새로 정의한다. 제주 RT 및 DA-RT 편차는 15분 RT 별도 데이터가 확보된 후 진행하며 육지 또는2023 V5 타깃을 이름만 바꿔 사용하지 않는다.

## 6. 기록

run마다 source/config/env/data hash, market/product, cutoff, eligible_dates, excluded reason, feature timestamp/availability, split manifest, policy+tariff+degradation assumptions, model parameters, action/SOC/price-vintage trace를 저장한다. benchmarks/v5와 원본 outputs는 overwrite 금지. 기록은 공개권한이 확인된 집계만 Git에 올린다.
