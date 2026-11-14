# Feature selection and leakage contract

## Status: incomplete full-input experiment

No selected input is being silently dropped. The current calendar-only SARIMAX is a legacy exploratory run and is not the requested model. No model may be trained until every field below is available over the complete 2023-01-01 00:00–2025-12-31 23:00 hourly grid and passes the input gate.

## Selected inputs (all required)

| Model panel column | Required source field / transformation | Availability rule |
|---|---|---|
| demand_actual_lag | Actual nationwide hourly demand | Lag only; no D+1 onward actual demand |
| demand_forecast_vintage | KPX `mlfd` day-ahead demand forecast | Historical vintage issued no later than D |
| generation_lng_lag / generation_coal_lag / generation_nuclear_lag / generation_other_fuels_lag | KPX `tradeYmd`, `tradeNo`, `fuelTpCd`, `amgo`, selected fuel groups | Historical actuals as lag features only |
| solar_generation_national_lag / wind_generation_national_lag | Nationwide/regional hourly solar and wind generation, aggregated only as specified | Historical actuals as lag features only |
| solar_generation_forecast_vintage | KPX `solgenKpx` | Historical forecast vintage available at D |
| reserve_supply_lag / reserve_operating_lag | Supply reserve and operating reserve as separate series | Historical actuals as lag features only; never conflate |
| lng_generation_fuel_unit_cost_available | EPSIS LNG generation fuel unit cost | Preserve source cadence; only available/published values |
| monthly_fuel_cost_available | `fuelType`, `day`, `untpc`, `unit` | Monthly cadence; retain units and conservative publication lag |
| wti_available / brent_available | FRED `DCOILWTICO` / `DCOILBRENTEU` | Publication-aware daily values |
| japan_lng_available / australia_coal_available | FRED `PNGASJPUSDM` / `PCOALAUUSDM` | Monthly cadence and conservative publication lag |
| weather_ta_lag / weather_si_lag / weather_ws_lag | KMA ASOS `TA`, `SI`, `WS` | Actuals as lag features only |
| weather_forecast_tmp_vintage / weather_forecast_wsd_vintage | KMA `TMP`, `WSD` | Historical forecast vintage issued by D |
| east_west_icsrpred_vintage / east_west_temp_vintage / east_west_ws_vintage | East-West `icsrPred`, `temp`, `ws` | Historical forecast vintage issued by D |
| is_holiday | KASI `locdate`, `isHoliday` | Complete 2023–2025 calendar |
| calendar | hour, weekday, month, season | Deterministic |

## Leakage and completeness rules

- At origin D, no observed value timestamped after D may enter a D+1..D+3 forecast. Actual observations are only legal as explicitly named lagged features.
- Forecast inputs use the vintage that was available by D, not a subsequently revised/current forecast.
- Monthly/daily series retain source cadence and use a documented publication lag; no zero fill or proxy substitution.
- `data/processed/model_inputs_2023_2025.csv` must include `timestamp`, `max_source_available_at`, and every required input column. All 26,304 hourly rows and all selected columns must be complete; source availability must not postdate timestamp.
- Missing one variable or period blocks model fitting. See [data collection report](data_collection_report.md) for gaps, failure categories, official sources, and required actions.

Data coverage observed: 2023-01-01 00:00:00 to 2025-12-31 23:00:00; target rows=26304/26304; SMP missing rate=0.000%.
