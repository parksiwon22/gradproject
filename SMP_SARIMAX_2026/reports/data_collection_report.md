# Data collection report

Collection used read-only copies of existing sources; API keys are not printed.

| Data | Status | Period / rows | Frequency | Unit | Missing rate | Notes |
|---|---|---|---|---|---:|---|
| aligned_hourly_all_features_2023 | copied_read_only | 2023-01-01T00:00:00 – 2023-12-31T23:00:00 / 8760 rows | 0 days 01:00:00 | KRW/kWh (SMP) | 0.00% | readable |
| HOME_전력거래_계통한계가격_시간별SMP | copied_read_only | 2023-01-01T00:00:00 – 2023-12-31T00:00:00 / 365 rows | hourly (wide date x 01–24 columns) | KRW/kWh (hourly SMP) | 0.00% | readable |
| HOME_전력거래_계통한계가격_연료원별SMP결정 | copied_read_only | 2026-01-23T00:00:00 – 2026-12-23T00:00:00 / 12 rows | 31 days 00:00:00 | unknown | 0.00% | readable |
| HOME_전력거래_전력거래량(전력시장·PPA)_연료원별 | copied_read_only | None – None / 265 rows | unknown | unknown | 95.26% | readable |
| kr_holidays_2023_events | copied_read_only | 2023-01-01T00:00:00 – 2023-12-25T00:00:00 / 18 rows | 6 days 00:00:00 | unknown | 21.48% | readable |
| powerDemandPerform_2023 | copied_read_only | None – None / 367 rows | unknown | unknown | 0.23% | readable |
| pssInLand_2023 | copied_read_only | None – None / 366 rows | unknown | unknown | 0.00% | readable |
| 태양광발전량,풍력발전량 | copied_read_only | None – None / 367 rows | unknown | MWh (source column labels) | 0.00% | readable |
| 한국가스공사_월별 발전용 천연가스 요금_23 | copied_read_only | None – None / 8 rows | unknown | unknown | 0.00% | readable |
| 한국서부발전(주)_신재생에너지발전량_20231231 | copied_read_only | 2023-01-01T00:00:00 – 2023-12-31T00:00:00 / 2189 rows | hourly (wide date x 01–24 columns) | unknown | 0.00% | readable |
| 한국전력거래소_시간별 전국 전력수요량 | copied_read_only | 2023-01-01T00:00:00 – 2023-12-31T00:00:00 / 365 rows | hourly (wide date x 01–24 columns) | unknown | 0.00% | readable |
| 지역별 시간대별 태양광 및 풍력 발전량(2309_2311) (1) | copied_read_only | 2023-09-01T00:00:00 – 2023-11-30T23:00:00 / 37128 rows | 0 days 01:00:00 | MWh (source column labels) | 7.06% | readable |
| 지역별 시간별 태양광 및 풍력 발전량_20230601_20230831 | copied_read_only | 2023-06-01T00:00:00 – 2023-08-31T23:00:00 / 37536 rows | 0 days 01:00:00 | MWh (source column labels) | 0.00% | readable |
| epsis | unavailable | — – — / — rows | unknown | unknown | — | directory not configured or not found |
| Existing project land SMP actual series (actual_smp only) | success | 2023-01-09T00:00:00 – 2025-12-31T23:00:00 / 26112 rows | hourly | KRW/kWh | 0.00% | Only timestamp and actual_smp were extracted; existing model features and predictions were not imported. |
| DCOILWTICO | connection_error | — – — / — rows | daily | USD/barrel | — | [official FRED page](https://fred.stlouisfed.org/series/DCOILWTICO); API=unverified_network_unavailable; ConnectionError |
| DCOILBRENTEU | connection_error | — – — / — rows | daily | USD/barrel | — | [official FRED page](https://fred.stlouisfed.org/series/DCOILBRENTEU); API=unverified_network_unavailable; network unavailable after first connection failure |
| PNGASJPUSDM | connection_error | — – — / — rows | monthly | USD/million Btu | — | [official FRED page](https://fred.stlouisfed.org/series/PNGASJPUSDM); API=unverified_network_unavailable; network unavailable after first connection failure |
| PCOALAUUSDM | connection_error | — – — / — rows | monthly | USD/metric ton | — | [official FRED page](https://fred.stlouisfed.org/series/PCOALAUUSDM); API=unverified_network_unavailable; network unavailable after first connection failure |
| KMA ASOS hourly at Seoul station 108 | connection_error | 2023-01-01 – 2025-12-31 / 0 rows | hourly observations | {'TA': 'C', 'SI': 'MJ/m2', 'WS': 'm/s'} | — | First failed request: connection_error (ConnectionError) |
| KPX SMP and day-ahead demand | manual_export_required | — – — / — rows | unknown | unknown | — | Official page confirms service and interval semantics but omits operation path and schema; import EPSIS CSV/Excel until official detailed guide is available. |
| KMA short-range forecast archive | not_used | — – — / — rows | unknown | unknown | — | Forecast API returns issue-time forecasts; historical vintage availability has not been verified, so it is excluded from historical model inputs. |
| East-West day-ahead solar forecast | credential_required | — – — / — rows | unknown | unknown | — | Requires a separate data.go.kr credential, not supplied; historical forecast vintage availability also needs validation. |

## Official specifications confirmed

- [KPX SMP and day-ahead demand](https://www.data.go.kr/data/15131225/openapi.do): hourly interval values are labeled by interval end; updated daily around 23:00 KST. The public page currently does not expose the operation path/schema in its page text, so automated historical querying is not attempted.
- [KMA APIHub ASOS interval endpoint](https://apihub.kma.go.kr/apiList.do?apiSeq=5): `kma_sfctm3.php`, maximum 31-day query; selected fields are `TA` (°C), `SI` (MJ/m²), `WS` (m/s). Collection uses Seoul station 108 as a single-station weather proxy.
- [FRED observations API documentation](https://fred.stlouisfed.org/docs/api/fred/series_observations.html) confirms the observations API method. The local execution environment could not resolve external hosts, so API-key validity/callability remains unverified.
- Official FRED pages confirm WTI and Brent are daily USD/barrel, and Japan LNG and Australian coal are monthly USD/million metric Btu and USD/metric ton respectively. Series pages are linked in the table above.

## Important data notes

- EPSIS hourly SMP uses the 2023 read-only local file when available. Official EPSIS CSV/Excel can be added under `EPSIS_EXPORT_DIR`; missing 2024–2025 intervals are not synthesized.
- The documented KPX day-ahead API refreshes once daily around 23:00 and timestamps each interval by its end. Only archived forecast vintages available at the forecast origin may be used.
- KMA observed weather is retained as raw data, but excluded from future exogenous variables to prevent use of post-origin observations.
- KMA forecast vintages and East-West solar prediction vintages are not used unless their historical issue-time availability can be demonstrated.
- FRED monthly series retain monthly frequency and original units. A one-month publication delay is assumed for any later model use.
