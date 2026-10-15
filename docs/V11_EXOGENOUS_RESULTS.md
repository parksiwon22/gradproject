# V11 수요·기상 확장 실제 실행 결과 — 2026-10-15

## 기존 V11 점검

기존 수집/실제학습은 달랐다. SMP 육지·제주2023~2026은 수집했고 육지만 학습했다. 전국2025 수요8,760시간은 보관했으나 학습에 쓰지 않았다. 기상 관측/예보, 실제 수요, 수요예측은 기존 모델에 없었다.

실제 입력21개: hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos, weekend, daily, weekly, spread, origin_price, origin_lag1, origin_lag24, origin_lag168, mean24, std24, min24, max24, mean168, std168, change24. 달력7개와 SMP14개다. V5의63피처는 별개이며 원 V5·main·app.py·outputs·원래 production.joblib을 수정하지 않았다.

## 공식 자료 실제 확보

| data | year | rows | complete_rows | resolution |
| --- | --- | --- | --- | --- |
| demand_hourly_all.csv | 2023 | 8760 | 8760 | hourly observations nationwide |
| demand_hourly_all.csv | 2024 | 8784 | 8784 | hourly observations nationwide |
| demand_hourly_all.csv | 2025 | 8760 | 8760 | hourly observations nationwide |
| power_daily_all.csv | 2023 | 365 | 365 | daily peak-state nationwide |
| power_daily_all.csv | 2024 | 366 | 366 | daily peak-state nationwide |
| power_daily_all.csv | 2025 | 365 | 365 | daily peak-state nationwide |
| power_daily_all.csv | 2026 | 287 | 287 | daily peak-state nationwide |
| weather_daily_all.csv | 2023 | 365 | 363 | daily Seoul108 |
| weather_daily_all.csv | 2024 | 366 | 362 | daily Seoul108 |
| weather_daily_all.csv | 2025 | 365 | 364 | daily Seoul108 |
| weather_daily_all.csv | 2026 | 288 | 287 | daily Seoul108 |
| demand_forecast_hourly.csv | 2023 | 8760 | 0 | hourly forecast archive mainland |
| demand_forecast_hourly.csv | 2024 | 8784 | 0 | hourly forecast archive mainland |
| demand_forecast_hourly.csv | 2025 | 8760 | 0 | hourly forecast archive mainland |
| demand_forecast_hourly.csv | 2026 | 6912 | 0 | hourly forecast archive mainland |

수요시간자료는2023 8,760/2024 8,784/2025 8,760시간을 확보했다.2026 시간별 수요실적은 미확보다. 대신2026에는 공식 일별 최대전력/공급능력/예비력/예비율을 수집했다. 이것을 시간별 실제 수요로 바꾸어 부르지 않는다. 일별 값은 과거 특정 날짜의 요약이라는 별도 피처로 시간행에 결합했다.

기상청 서울108의 평균·최고·최저기온, 평균풍속, 상대습도, 일조시간, 운량7종 일자료를 수집했다.2026 마지막날에는 일부값만 있는 잠정자료가 있어 complete_rows를 따로 기록했다. 일사량(MJ/m²)과 일조시간(hours)은 다르며 일사량은 이번 공식 신규수집에서 미확보다. 2023 원V5 CSV에 시간별 기상값은 있지만 다년 공식 API 수집 완료로 간주하지 않았다.

자료 출처:

- [공식 전국 시간별 수요 파일 및 과거년도 목록](https://www.data.go.kr/data/15065266/fileData.do). 파일별raw/해시/수집시각 보존. 전국자료는 제주를 포함하며 육지 수요예측과 같은 지리적 정의가 아니다.
- [기상청 공개 요소별 관측](https://www.weather.go.kr/w/weather/land/past-obs/obs-by-element.do): 서울108, 일별. 공개표의 풍속 표시함수는 숫자 리터럴만 추출하며 JavaScript를 실행하지 않는다.
- [EPSIS 전력수급실적](https://epsis.kpx.or.kr/epsisnew/selectEkgeEpsMepChart.do?menuId=030100): 일 최대수요 시점의 전력/공급예비력/예비율. 원응답의 중복 동일값은 제거했고 상충 중복이면 실패하도록 했다.
- [KPX 하루전 육지 수요예측](https://www.kpx.or.kr/menu.es?mid=a10404090100): 공식 과거 날짜조회198주간을 캐시하여33,216시간 확보. 연간 엑셀은 웹 방화벽 차단으로 실패했고 이를 회피하지 않고 별도 공식 공개 조회서비스를 이용했다. 시각1~24는 interval-end로 시작00~23시에 정렬. 원래 발표시각/최초version은 미확인이라 retrospective forecast-value 연구로 표시했다.
- [기상청 최신 단기예보](https://www.weather.go.kr/neng/forecast/short-term.do): 공식 페이지가 쓰는 공개 응답에서 실제 hourly 예보를 수집. 서울1159068000 예보격자는 ASOS108 관측소와 공간적으로 완전히 같지 않다. 실제20:00 단기예보·22:10 초단기예보와22:31:02 KST 최초수신을 구분해 보존했고, target16/17/18일 각24시간을 확보했다.

## API 연결과 비밀정보

Windows 사용자 환경변수에서 **KPX_SERVICE_KEY만** 확인됐다. 키는 출력하지 않았다. 공식 Swagger에서 확인한 KPX endpoint로 resultCode00,48행 수신에 성공했다. KMA_SERVICE_KEY/KMA_APIHUB_AUTH_KEY는 없었다. 공공데이터포털 계정키로 ASOS 승인 여부를 검사했으나HTTP403이라 시간별 ASOS 수집은 실패했다. [ASOS 시간자료 서비스](https://www.data.go.kr/data/15057210/openapi.do)의 활용승인 또는 별도 KMA_SERVICE_KEY 설정이 필요하다. API 키·인증query·원문 HTTP 오류 URL은 로그/Git에 저장하지 않는다. 응답에 키가 echo되는 경우도 제거한다. API 승인실패가 있다고 공개 일자료를 시간별자료라고 주장하지 않았다.

## 시간·가용성 계약과 입력 확장

기본 D-1 23시 유지. SMP21피처를 재사용하고 다음을 추가했다.

- 시간별 전국 수요4피처: target lag72/168시간,48시간 지연한24/168시간 평균. 이 연간 역사파일이 당시 실시간으로 제공됐다는 증거는 없어 후향 관측 replay다.
- 일별 서울 기상14피처:7종 x origin 날짜 기준1일 전/7일 전. 목표일 실제기상은 이 모델에 넣지 않는다.
- 일별 전력8피처:최대수요/공급능력/예비력/예비율 x origin 날짜 기준1일 전/7일 전.
- 별도 연구에는 공식 target 시간의 archived day-ahead demand1피처를 추가했다. 이것은 실제 수요가 아니라 공식 수요예측이지만 원래 발표본을 보증하지 못한다.

따라서21/25/35/39/43/44 등 피처 수를 그룹별로 기록했다. 정확한 실제 학습 목록은 V11_EXOGENOUS_FEATURES.json과 로컬 config/experiments.json에 있다. 기존63개와 이름만 맞춘 변수를 만들지 않았다. reserve의 일최대시점 값과 V5 원 hourly reserve는 같은 정의가 아니다. 원 V5 holiday·가스요금·연료·신재생발전·시간별일사까지 완전 재현한63피처 다년 자료는 여전히 없다.

과거 관측 release/vintage가 없으므로 역사실험의 운영가능성은 미인증이다. **실제 미래 실행에서는** 별도 observations.sqlite의 최초수신/revision을 검사해23시 또는 preview 실제실행시각 이전 값만 고른다. 뒤늦은 정정값은 기존 cutoff 입력에 들어가지 않는다. 과거자료 최초수신을 당시 발표시각으로 소급하지 않는다.

## 실제 모델 비교

HGB250/leaf31/lr.05 및 ExtraTrees150/min_leaf10/max_features.8,seed42 고정. 같은 연구 안에서는 모든 그룹에 동일한 완전일/시간행과 같은 train/validation/test를 사용했다. 무작위 검증/early stopping은 사용하지 않았다. 검증MAE로 각 그룹 family를 정하고 선정파일을 쓴 뒤 test 지표를 계산했다. Q3 2026은 앞선 V11 작업에서 이미 본 기간이므로 완전 미열람 테스트가 아니다. 결과를 보고 하이퍼파라미터를 재튜닝하지 않았다.

2025 비교는 학습2023~2025-06,검증2025-07~09,평가2025-10~12다.2026 비교는 학습2023~2026-03,검증2026-04~06,평가2026-07~09다. 실제 날짜경계/행수는 experiments.json에 있다.

### 그룹별 검증선정 모델의 평가

| study | group | family | MAE | RMSE | R2 | peak_MAE | peak_Bias | peak_F1 | ramp_F1 | daily_max_MAE | peak_hour_MAE | EV_regret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025_hourly_demand | SMP_hourly_demand_weather | extra_trees | 10.9202 | 15.3451 | 0.4781 | 39.2317 | -39.2317 | 0.0000 | 0.2831 | 13.1479 | 5.5870 | 9.5444 |
| 2025_hourly_demand | SMP_weather | extra_trees | 10.9991 | 15.3670 | 0.4766 | 37.9004 | -37.9004 | 0.0000 | 0.3049 | 13.2371 | 5.4783 | 9.8608 |
| 2025_hourly_demand | SMP_hourly_demand | hgb | 12.9395 | 17.4718 | 0.3234 | 34.7762 | -34.7762 | 0.0000 | 0.2752 | 12.8513 | 5.1087 | 9.7893 |
| 2025_hourly_demand | SMP | hgb | 12.8346 | 17.0140 | 0.3584 | 33.6775 | -33.6775 | 0.0000 | 0.2458 | 12.6456 | 4.9565 | 11.1330 |
| 2026_daily_exogenous | SMP_daily_power_reserve | hgb | 10.8602 | 16.4750 | 0.7441 | 12.5741 | -11.4989 | 0.8025 | 0.3806 | 12.3068 | 2.1413 | 5.0436 |
| 2026_daily_exogenous | SMP_daily_power_weather_reserve | hgb | 10.8630 | 16.5815 | 0.7408 | 11.6564 | -10.4387 | 0.8152 | 0.3806 | 11.8882 | 2.1087 | 5.2064 |
| 2026_daily_exogenous | SMP_weather | extra_trees | 10.4043 | 16.3620 | 0.7476 | 10.7528 | -9.1559 | 0.8113 | 0.3561 | 10.4951 | 2.2283 | 4.4510 |
| 2026_daily_exogenous | SMP | hgb | 10.8617 | 16.2147 | 0.7522 | 12.0705 | -11.0570 | 0.8158 | 0.3596 | 12.3658 | 2.3043 | 4.3633 |

### 동일 HGB:2026Q3,공식 수요예측 archive 추가

| group | family | MAE | RMSE | R2 | peak_MAE | peak_Bias | peak_F1 | ramp_F1 | daily_max_MAE | peak_hour_MAE | EV_regret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SMP | hgb | 10.8617 | 16.2147 | 0.7522 | 12.0705 | -11.0570 | 0.8158 | 0.3596 | 12.3658 | 2.3043 | 4.3633 |
| SMP_archived_demand_forecast | hgb | 10.3500 | 15.1994 | 0.7822 | 7.9949 | -5.0549 | 0.8575 | 0.4157 | 9.7877 | 1.8913 | 2.3226 |
| SMP_archived_demand_forecast_lagged_weather | hgb | 11.1191 | 16.1400 | 0.7544 | 8.2503 | -5.7011 | 0.8586 | 0.3658 | 9.7351 | 1.9891 | 2.2324 |
| SMP_archived_demand_forecast_lagged_weather_power | hgb | 10.5502 | 15.5775 | 0.7713 | 9.2511 | -7.6782 | 0.8561 | 0.3658 | 10.6514 | 2.0000 | 2.0974 |

수요예측archive를 넣은 HGB는 SMP 단독 대비 전체/고가MAE·regret가 개선됐지만 원본 발표시각 미확인이라는 제한이 있다. 공식 API가 수요예측과 SMP를 함께 제공하거나 수요예측을 얻을 때 이미 SMP가 공개되어 있다면 사전 SMP 예측의 추가 가치로 해석할 수 없다. 실제 최초수신 로그가 더 필요하다. 일별 관측 기상을 더했다고 모든 지표가 좋아지지 않았다. 모든 family와 모든 실패/성공, RMSE/R²,고가Bias/Precision/Recall/F1,상승·하락ramp,일최고가·발생시간,EV regret는 집계CSV에 있다. target 실제기상을 넣은 별도 연구 상한 실험은 research_oracle_metrics.csv에 저장하며 운영 성능으로 집계하지 않는다.

고가 기준150원/kWh,상승ramp15원/h,Bias=예측-실제. EV는 기존 v11.evaluate의1kW·연속2/3/4/5/6/8시간 가상 SMP 요금으로 동일 조건이다. 실제 소매요금·V2G 수익이 아니다.

## 실제 미래 출력: 변수가 들어갔는가

- 2026-10-16 / 2026-10-15T22:57:19.204283+09:00 / official_known / 43피처·실제 과거 일별 수요/기상/예비력 입력
- 2026-10-17 / 2026-10-15T22:57:19.204283+09:00 / pre23_exogenous_preview_availability_unverified / 43피처·실제 과거 일별 수요/기상/예비력 입력

43피처 출력은 실제 수신한 과거 일별 수요·기상·예비력을 사용했다. 미래 실제 수요/기상은 없다. 이 출력에는 아직 target 수요예측과 target 기상예보는 사용하지 않았음을 JSON의 weather_forecast_used/demand_forecast_used=false로 명시했다. 2026 시간별 수요실적과 과거 기상예보vintage가 없는 상태를 숨기지 않는다.

별도로 **실제로 발표된 미래 기상예보를 넣은 연구 시나리오**도 생성했다. 이26피처 모델은 과거 target일 관측 기상의5개 일요약(평균/최고/최저기온·습도·풍속)을 대용값으로 학습하고, 실제 미래 실행에서20시 발표 예보를 같은5요약으로 바꿔 넣었다. 예보 입력으로 검증된 모델이 아니며 학습/운영 분포차이와 ASOS/예보격자 공간차이가 있다. 주모델/EV추천으로 승격하지 않았다. 이름과 파일에는 OBSERVED_WEATHER_TRAINING_PROXY_NOT_VALIDATED를 명시했다. 당시 예보 archive가 확보되면 대용값을 실제 예보로 교체해 재검증해야 한다.

16일 SMP는 이미 공식가격을 확보한 날짜라 AI 미공개 미래예측으로 부르지 않는다.17일은 조회한 출처에서 가격을 관측하지 못한 후보이며 모든 경로의 미공개를 입증한 것은 아니다.23시 이전 실행은 preview로 기록했다. 모든 예측은 불변JSON, 사후평가는 별도파일이며 원예측을 덮어쓰지 않는다.

## 재사용·자동 축적·웹

기존 SMP store/features/evaluate/forecast/backup/pipeline을 재사용했다. 원래 price-only production 모델을 덮어쓰지 않고 exogenous/ 밑에 분리했다. pipeline은 외생모델이 있을 때 exogenous_refresh를 추가 실행한다. 일관측은11시 이후 하루1회 현년도 갱신,기상예보는 수집 때마다해시별원문·최초수신·발표시각을 보존한다. 역사 수요예측 주간수집은 캐시로 중복을 막는다. 자동재학습과 비공개자료 업로드는 없다. app_v11.py의 별도 실험 영역에 실사용 변수와 API상태를 연결했다. 기존 EV결정은 이 실험으로 조용히 교체하지 않았다.

## 실행

```powershell
$py='..\.venv\Scripts\python.exe'
& $py -m v11.authenticated_data --backfill
& $py -m v11.exogenous_collect
& $py -m v11.demand_forecasts
& $py -m v11.exogenous_model
& $py -m v11.forecast_covariates
& $py -m v11.exogenous_refresh
& $py -m v11.exogenous_report
& $py -m unittest tests.test_v11_exogenous tests.test_v11 -v
& $py -m streamlit run app_v11.py
```

KMA_SERVICE_KEY는 서비스 승인 후 Windows 사용자 환경변수에 설정한다. API허브 authKey는 KMA_APIHUB_AUTH_KEY이며 data.go.kr key와 다르다. 키를 채팅·코드·Git에 넣지 않는다. ASOS403에서는 현재 수집을 완료했다고 하지 않는다.

## 미완료·후속 검증

기상청 시간별 ASOS(특히 일사량) 승인,2023~2026 과거 기상예보의 original issue archive,수요예측 최초본/정정본,2026 시간별 실제 수요,전국 다지점 기상,발전원별 시간발전/신재생 전망/연료가격의 동일정의·release 기록은 미완료다. 예비력/공급능력은 실제 추가했지만 다른 자료를 임의 대체하지 않았다. 새로 생성한17일 예측의 전향오차는 실제값이 도착해야 평가할 수 있다. 전체 정확도와 피크·regret가 동시에 우월하다고 주장하지 않는다.
