# Fuel price collection status

## Result

**No price dataset was downloaded in this run. No substitute series was used, and SARIMAX was not run.** Local access to the three relevant hostnames fails at DNS resolution (Windows `getaddrinfo` error 11001). This occurs before an HTTP request reaches FRED, EPSIS, or the public data API; it therefore does not test or reject any API credential.

| Dataset | Collected? | Local coverage | Failure cause | Additional credential? | User action |
|---|---:|---|---|---|---|
| FRED WTI `DCOILWTICO` (USD/barrel) | No | No matching FRED CSV; requested 2023-01-01–2025-12-31 | DNS lookup for `api.stlouisfed.org` failed (`gaierror`, errno 11001); no HTTP response, so key status unknown | `FRED_API_KEY` already exists locally; no additional key identified | Restore DNS/network access, then rerun `python -m src.cli collect` for vintage history. [Official WTI series page](https://fred.stlouisfed.org/series/DCOILWTICO) can provide current revised values, but a current-value CSV is not a replacement for ALFRED vintages. |
| FRED Brent `DCOILBRENTEU` (USD/barrel) | No | No matching FRED CSV; requested 2023-01-01–2025-12-31 | Same pre-HTTP DNS failure; authentication untested | Same `FRED_API_KEY` | Restore DNS/network and rerun vintage collector. [Official Brent series page](https://fred.stlouisfed.org/series/DCOILBRENTEU) is a current-value fallback only; it does not supply the required as-of history by itself. |
| FRED Japan LNG `PNGASJPUSDM` (USD/million Btu, monthly) | No | No matching FRED CSV; requested 2023-01–2025-12 | Same pre-HTTP DNS failure; authentication untested | Same `FRED_API_KEY` | Restore DNS/network and rerun vintage collector. [Official Japan LNG series page](https://fred.stlouisfed.org/series/PNGASJPUSDM) provides current revised values but is not the required as-of vintage archive. |
| FRED Australian coal `PCOALAUUSDM` (USD/metric ton, monthly) | No | No matching FRED CSV; requested 2023-01–2025-12 | Same pre-HTTP DNS failure; authentication untested | Same `FRED_API_KEY` | Restore DNS/network and rerun vintage collector. [Official Australian coal series page](https://fred.stlouisfed.org/series/PCOALAUUSDM) provides current revised values but is not the required as-of vintage archive. |
| EPSIS LNG generation **fuel cost rate** `연료비단가` (KRW/kWh) | No | No matching verified EPSIS export; requested raw months 2022-11–2025-12 (38 months) | Actual request to `epsis.kpx.or.kr` failed before HTTP: DNS `gaierror` errno 11001. Authentication does not apply to the manual export page. Whether the full historical range is downloadable was not testable. | No key for manual EPSIS page export | Open [EPSIS > 전력거래 > 연료비용](https://epsis.kpx.or.kr/epsisnew/selectEkmaFucUpfChart.do?menuId=040100). Set cadence monthly, measure **연료비단가** (not 연료단가 or 열량단가), fuel **LNG**, range 2022-11 through 2025-12; export CSV. Save as `data/monthly/epsis_lng_fuel_cost_202211_202512.csv`. |
| KPX monthly fuel cost API: `fuelType`, `day`, `untpc`, `unit` | No | No matching verified CSV/API response; raw `day=202211`–`202512` (38 monthly periods), all fuel categories | Actual requests to the catalog, API host, and official CSV page all failed before HTTP: DNS `gaierror` errno 11001. API-key acceptance and historical archive availability remain untested. The official rendered catalog specifies the base URL and fields but omits the operation suffix. | API requires data.go.kr `serviceKey`; configured local name is `DATA_GO_KR_SERVICE_KEY`. No additional key name is known. Authorization for product 15099765 cannot be tested until DNS works. | Use [KPX Monthly Fuel Cost API catalog](https://www.data.go.kr/data/15099765/openapi.do). Request every `day` from 202211 through 202512, retaining all returned `fuelType` categories, `day`, `untpc`, and `unit`. The official catalog lists base endpoint `https://apis.data.go.kr/B552115/FuelCost1`, function “월별 열량단가 조회”, and these fields, but does not expose the operation suffix in its rendered text. If the operation is not available in the authenticated portal, download the official [KPX monthly fuel cost CSV](https://www.data.go.kr/data/15069386/fileData.do) for 2022-11–2025-12, then confirm the exact columns before import. |

## Do not merge these two Korean price series

The EPSIS chart distinguishes `연료단가`, `열량단가`, and `연료비단가`. The desired EPSIS series is LNG `연료비단가`, reported in KRW/kWh; the official page describes it as the weighted-average cost rate that reflects the heat-rate and plant price-characteristic curve. The KPX open API catalog's documented function is “월별 열량단가 조회” and its sample unit is KRW/Gcal. They are different measures and must be collected in separate columns, with original units preserved.

The local `existing_한국가스공사_월별 발전용 천연가스 요금_23.xlsx` has only eight rows and has not been verified against either requested EPSIS `연료비단가` or KPX `untpc` schema. It is not used as a substitute.

## Domestic price mapping rule confirmed by the user

Fetch and retain raw monthly domestic source data for **2022-11 through 2025-12 inclusive** (38 months). For each forecast-origin month M in 2023-01 through 2025-12, attach source month **M−2**. Thus January 2023 uses November 2022, December 2025 uses October 2025, and November/December 2025 raw values are retained but are not used in these forecast-month mappings. This two-month lag is the user's conservative leakage-prevention assumption; it is not being represented as an officially verified publication delay. Assign the features using the forecast-origin month, and hold them unchanged for all 72 forecast hours even if the target timestamps enter the next month. The mapping has not been materialized because neither source file has been collected.

## FRED and ALFRED availability-time method

The current FRED values cannot be fetched until DNS works. The collector has been changed to request FRED/ALFRED `output_type=3` (new and revised observations by vintage) for the four series. It stores vintage rows separately, then builds one daily origin snapshot for 2023-01-01–2025-12-31. The as-of rule is conservative: only use observation values whose ALFRED `realtime_start` is strictly before the forecast origin date and whose observation date is also before that date. Because ALFRED vintage dates have day granularity and do not establish an intraday release time, same-origin-day releases are excluded. A 2022-10 warm-up is requested so early 2023 origins can carry the latest previously available quotes. This output has not been produced because DNS resolution failed.

Official ALFRED docs state that ALFRED retains when values were originally released and later revised; FRED's observations API supports output type 3 for new and revised observations by vintage date. See [ALFRED overview](https://fred.stlouisfed.org/docs/api/fred/alfred.html), [observations API output types](https://fred.stlouisfed.org/docs/api/fred/series_observations.html), and [real-time periods](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html).

## Local network fix

Actual HTTPS requests to the EPSIS export page, the KPX API catalog, the KPX API host, and the KPX CSV page all failed with `ConnectionError → MaxRetryError → NameResolutionError → gaierror errno=11001`. The FRED request failed with the same resolver error. No HTTPS proxy is configured. This is a DNS/network failure, before API authentication or archive lookup. Check that the computer has internet access and is connected to any required VPN; check that firewall/security software allows DNS and outbound HTTPS; then run these in PowerShell:

```powershell
Resolve-DnsName api.stlouisfed.org
Resolve-DnsName apis.data.go.kr
Resolve-DnsName epsis.kpx.or.kr
Test-NetConnection api.stlouisfed.org -Port 443
```

If DNS still fails, reconnect the network/VPN or ask the network administrator to repair the configured DNS resolver. If browsing works only through an organization proxy, configure the approved proxy for this Python process rather than placing credentials in the project. Once DNS resolves, rerun collection; only an HTTP response can distinguish API-key rejection from service authorization or request/schema errors. Keep `FRED_API_KEY` and `DATA_GO_KR_SERVICE_KEY` only in local `.env`; never put values in source, logs, or GitHub.

## Model gate

These missing selected variables remain mandatory. Do not mark the input panel ready, drop these predictors, replace them with the KOGAS workbook, or run SARIMAX until both international and domestic prices have been acquired, their periods/units verified, and availability lags built. The domestic mapping is `forecast-origin month − 2 months` and is held constant across that origin's 72-hour horizon.
