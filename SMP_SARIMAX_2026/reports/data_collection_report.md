# Data collection and readiness report

## Decision

**NOT READY — no SARIMAX training, Validation, or Final Test was run for this full-input experiment.** The processed SMP target covers 2023-01-01 00:00 through 2025-12-31 23:00 (26,304 hourly rows), but the selected exogenous input set does not. Existing results from the calendar-only implementation are an incomplete exploratory experiment, not the requested validation.

See the separate [fuel price collection report](fuel_price_collection_report.md) for the WTI/Brent/Japan LNG/Australia coal and domestic EPSIS/KPX collection status, including the current DNS error and exact export steps.

The required complete hourly panel is `data/processed/model_inputs_2023_2025.csv`. It must contain all field columns listed in `feature_selection_report.md`, `timestamp`, and `max_source_available_at`. Every row must contain all selected values and must pass the as-of check. Observed actuals after forecast origin D must not appear in D+1..D+3 features; prior actuals must be represented as lags. Forecast inputs must be historical vintages available by D.

## Failure diagnosis

- **Network:** probes to official FRED, KMA, KPX, and KASI service/catalog hosts failed with connection errors before an HTTP response. This is a network/connection failure in the current environment.
- **Authentication:** because no HTTP response was received, key acceptance/rejection could not be tested. There is no evidence that any configured key is invalid. No key value is recorded in this report.
- **API specification:** some KPX catalog pages expose a base URL and summary but not the concrete operation route/output schema needed to implement the selected fields (notably `mlfd` and fuel-code series). This is a separate specification gap.
- **Historical archive:** historical vintages/retention for day-ahead forecasts and the East-West forecast API were not verified. Connection failure is not evidence that historical querying is impossible. No complete archive is present locally.

## Required inputs, current coverage, and action

| Selected input | Current local coverage/status | Missing period / issue | Required action and official source |
|---|---|---|
| Actual hourly land SMP (target) | Complete 2023-01-01–2025-12-31, 26,304/26,304 | None for target | Preserve; source/export reference: [EPSIS hourly SMP](https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpShdChart.do?menuId=040202.) |
| Actual hourly nationwide demand (lag only) | Local aligned 2023 file only, 8,760 hours; source mapping needs verification | 2024-01-01–2025-12-31 | Download 2024 and 2025 from [KPX nationwide hourly demand file](https://www.data.go.kr/data/15065266/fileData.do); verify source fields before import |
| Day-ahead demand forecast `mlfd` | 2023 local daily demand forecast workbook exists; no verified issue-time vintage; none for 2024–2025 | 2023 vintage timestamps; all 2024-01-01–2025-12-31 | Obtain archived forecast vintages and exact operation/field mapping from [KPX SMP and demand forecast API](https://www.data.go.kr/data/15131225/openapi.do); API page alone does not prove 2023–25 vintage availability |
| Generation by fuel: `tradeYmd`, `tradeNo`, `fuelTpCd`, `amgo` (actuals lagged) | 2023 aligned fuel columns exist but mapping to required raw fields unverified | 2024-01-01–2025-12-31 and 2023 field provenance | Import exact selected fuel codes and hourly intervals via [KPX fuel generation API](https://www.data.go.kr/data/15106744/openapi.do); obtain official operation schema if route is not supplied |
| Nationwide hourly solar and wind generation (actuals lagged) | Regional CSV fragments only 2023-06-01–2023-11-30; one fragment has ~7.06% missing; existing Seoul/West columns are not nationwide substitutes | 2023-01-01–2023-05-31, 2023-12-01–2025-12-31, plus gaps | Obtain exact nationwide/regional solar and wind source exports; [KPX solar API](https://www.data.go.kr/data/15103243/openapi.do) documents solar only; wind requires its separate official dataset |
| Day-ahead solar forecast `solgenKpx` | No local vintage data | Entire 2023-01-01–2025-12-31; archive availability unknown | Obtain archived KPX forecast vintages and field schema; [KPX SMP/demand API catalog](https://www.data.go.kr/data/15131225/openapi.do) or official KPX data contact |
| Supply reserve and operating reserve, separately | 2023 daily supply reserve/rate workbook only; not hourly; no verified operating-reserve series | Hourly 2023-01-01–2025-12-31 | Export both distinct fields from [KPX reserve data](https://www.data.go.kr/data/15099819/fileData.do); if 3-month query limit applies, download 12 chunks: 2023-01-01–03-31, 04-01–06-30, 07-01–09-30, 10-01–12-31; repeat for 2024 and 2025 |
| LNG generation fuel unit cost (EPSIS) | Local KOGAS monthly generation gas tariff has 8 rows but is not verified as the selected EPSIS field | Full 2023-01–2025-12, exact series unavailable | Download exact LNG generation fuel unit-cost export from [EPSIS fuel unit cost](https://epsis.kpx.or.kr/epsisnew/selectEkmaFucUpfChart.do?menuId=040100) |
| Monthly fuel cost `fuelType`, `day`, `untpc`, `unit` | No verified exact-field monthly file | 2023-01–2025-12 (36 months) | Download full period from [KPX monthly fuel cost CSV](https://www.data.go.kr/data/15069386/fileData.do), retaining original units and monthly cadence |
| WTI `DCOILWTICO`, Brent `DCOILBRENTEU` | FRED API not reachable; no complete local series confirmed | 2023-01-01–2025-12-31 | Restore network and retry FRED, or download observations: [WTI](https://fred.stlouisfed.org/series/DCOILWTICO), [Brent](https://fred.stlouisfed.org/series/DCOILBRENTEU) |
| Japan LNG `PNGASJPUSDM`, Australian coal `PCOALAUUSDM` | FRED API not reachable; no complete local series confirmed | 2023-01–2025-12 (monthly) | Restore network and retry FRED, or download observations: [Japan LNG](https://fred.stlouisfed.org/series/PNGASJPUSDM), [Australian coal](https://fred.stlouisfed.org/series/PCOALAUUSDM) |
| KMA ASOS observed `TA`, `SI`, `WS` (lag only) | 2023 local proxy has TA complete, WS 8,723/8,760, SI 4,781/8,760; station/source mapping not verified; no 2024–25 | Complete verified hourly records 2023-01-01–2025-12-31 | Retrieve exact variables by station and dates using [KMA ASOS API](https://apihub.kma.go.kr/apiList.do?seqApi=2) (`kma_sfctm3.php`, up to 31 days/query), or official export; 36 monthly chunks cover the period |
| KMA forecast `TMP`, `WSD` with vintage | No local data | Entire period; historical vintage/archive availability unknown | Obtain issue-time historical forecast archive, or confirm archive export from [KMA short-range forecast spec](https://apihub.kma.go.kr/apiList.do?seqApi=10) (`nph-dfs_shrt_grd`) |
| East-West forecast `icsrPred`, `temp`, `ws` with vintage | No local data; service key not verified/possibly service-specific | Entire 2023-01-01–2025-12-31; API page created 2025-07-17, earlier history/archive unknown | Request historical vintages from [East-West day-ahead solar forecast API](https://www.data.go.kr/data/15144541/openapi.do); if service requires a separate key, enter it locally as `EAST_WEST_SOLAR_SERVICE_KEY` in `.env` |
| Holiday `locdate`, `isHoliday` | 2023-only file, 18 listed dates | 2024-01-01–2025-12-31 and validate full 2023 calendar | Collect 2023–25 from [KASI special date API](https://www.data.go.kr/data/15012690/openapi.do) or official annual calendar exports |
| Calendar hour/weekday/month/season | Derivable for full hourly grid | None | Generate locally; does not replace the required selected inputs |

## Credentials and safe retry

`.env` variable names (values must remain local): `DATA_GO_KR_SERVICE_KEY`, `KMA_APIHUB_KEY`, `FRED_API_KEY`; optional service-specific names are `KMA_DATA_GO_SERVICE_KEY`, `KPX_FORECAST_SERVICE_KEY`, and `EAST_WEST_SOLAR_SERVICE_KEY`. Current connection errors do not justify rotating keys. Never paste keys into source code, logs, reports, or GitHub.
A manual export should be saved locally under the project data import workflow, with source, exact fields, units, date range, and retrieval date recorded. Do not fill gaps with zero or substitute a different proxy series.

## Execution status

`python -m src.cli check-inputs` checks panel structure, field coverage, and row-level source-time bounds without fitting; it does not establish origin-specific availability for overlapping forecasts. `python -m src.cli backtest` currently refuses all fitting, including calendar-only fitting. Complete origin-specific vintage/leakage validation and full-input SARIMAX integration before producing Validation/Final Test.
