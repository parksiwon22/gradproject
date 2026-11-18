# SMP 72시간 연구 입력자료 12개 그룹 (2023–2025)

예측 origin은 D일 23시(KST), 목표는 D+1~D+3 육지 SMP 72시간이다. 이 폴더는 현재 확인한 입력 원자료/패널을 번호별로 모은 검토용 사본이다. 모델 학습은 수행하지 않았다.

번호는 사용자가 확정한 12그룹 목록 순서다. 제외된 하루전 태양광 예측(solgenKpx)과 동서발전 예보는 포함하지 않는다.

| 번호 | 자료 | 이 폴더 파일 | 상태 및 주의 |
|---:|---|---|---|
| 1 | 실제 시간별 전력수요 | `01_actual_hourly_demand.csv` | 2023–2025 26,304시간. 전국+제주 후보이며 EPSIS 육지 자료와 범위가 다름. 별도 자료의 1,095시간 결측 미해결. |
| 2 | 하루전 육지 수요예측 (`mlfd`) | `02_day_ahead_demand_mlfd.csv` | 로컬 API 패널 표본 4,848/52,608행뿐. 완전한 2023–2025 육지 예측 CSV는 팀 저장소 `shared_datasets/kpx_day_ahead_demand/mainland_demand_forecast_2023_2025.csv`에 있으며 발표시각/vintage 미검증. |
| 3 | 공휴일 (`locdate`, `isHoliday`) | `03_kasi_holidays.csv` | KASI API 원자료. API 응답을 CSV로 보존. |
| 4 | 달력 특성 | `04_calendar_features.csv` | 2023–2025 시간별 파생 패널. 공통 변환 기준(요일·계절·sin/cos)은 모델 간 확정 필요. |
| 5 | 발전원별 발전량 (`tradeYmd`, `tradeNo`, `fuelTpCd`, `amgo`) | `05_generation_by_fuel_amgo.csv` | API 표본 100일(2023-01-01~2023-04-11), 전체 2023–2025 아님. 아래 첨부 연간 용량·거래량 CSV는 별도 공식 파일이며 `amgo`와 동일 필드·정의라 단정하지 말 것. |
| 6 | 지역별 태양광·풍력 실적 | `06_regional_solar_wind_actuals.csv` | API/가공 패널. 풍력은 제주만 있고 태양광은 일부 날짜 표본. 계량 ZIP은 별도 정의(ESS 충방전 제외)이며 혼합 금지. |
| 7 | 공급예비력·공급예비율 | `07_supply_reserve_mw_pct.csv` | 5분 패널. 2023–2025 1,095시각 결측/산식·정의 검토 필요. 공급예비력과 운영예비력 구분. |
| 8 | 국내 LNG 발전 연료비단가 (원/kWh) | `08_domestic_lng_fuel_cost.csv` | 원본 HOME 엑셀에서 연료비단가 표의 LNG 열 추출. 원본 시트·셀·기준월 및 조건부 동월 실험은 별도 출처 기록 필요. |
| 9 | FRED WTI·Brent (USD/barrel) | `09_fred_wti_brent.csv` | 현재 패널은 과거 값/빈티지 반복 문제가 있어 D−2 as-of 입력으로 사용 불가. 원시 시리즈 ID는 DCOILWTICO, DCOILBRENTEU. |
| 10 | FRED 국제 LNG·석탄 | `10_fred_lng_coal.csv` | 현재 패널은 과거 월값 반복으로 D−2 기준 사용 불가. PNGASJPUSDM, PCOALAUUSDM 원시 시리즈로 재구성 필요. |
| 11 | KMA ASOS 실측 TA/SI/WS | `11_kma_asos_actuals.csv` | 서울 108, 2023–2025 26,269/26,304시간. 35개 시간 누락 및 SI 결측/센티널 분류 필요. |
| 12 | KMA 중기예보 | `12_kma_midrange_temperature_seoul.csv` | 팀 GitHub에서 받은 서울 기온 중기예보 CSV. API만 있는 자료가 아니라 발표본 CSV가 있음. 2023–2026 발표 자료가 보존되어 있으며 일별 최저·최고기온(시간별 아님), 과거 발행 완전성과 72시간 연결 규칙은 별도 검증 필요. 중기 날씨(서울·인천·경기, 강수확률/상태)는 `../kma_midrange/` 원본 폴더에 있음. |

## 별도 제공된 KPX 연료원 연간 CSV

사용자 제공 파일은 이 폴더의 5번 API 표본을 대체/덮어쓰지 않도록 `05_generation_by_fuel_amgo_user_files/`에 원본 그대로 둔다. 설비용량·전력거래량 필드 자료이며 `amgo` 발전량 API 자료와 열 정의, 단위, 범위가 같다고 확인되기 전에는 병합하지 않는다.

- 한국전력거래소_연료원별 시간대별 설비용량 및 전력거래량_20231231.csv
- 한국전력거래소_연료원별 시간대별 설비용량 및 전력거래량_20241231.csv
- 한국전력거래소_연료원별 시간대별 설비용량 및 전력거래량_20251231.csv

## 원칙

- 이 디렉터리는 검토용 패키지 사본이다. 기존 프로젝트 원자료를 수정하거나 삭제하지 않았다.
- API에서만 확인되고 CSV로 보존되지 않은 응답은 각 그룹 상태에 API 전용/부분 확보로 표시한다. API 인증키는 저장하지 않는다.
- 실측은 origin 이전 과거 lag에만 사용한다. 목표시각 이후 실측은 입력에 넣지 않는다.
- 데이터 공백, 공개범위 밖, 실제 0은 서로 구분한다.
- 팀 원본 및 상세 검수 근거: `gradproject_shared_data_20261118/shared_data/`의 현지 수집 사본과 `SMP_72H_RESEARCH/reports/required_input_availability_20261118.md`.
