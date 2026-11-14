from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import OUT, PROCESSED, RAW, REPORTS


def write_collection_report() -> None:
    lines = [
        "# Data collection and readiness report", "",
        "## Decision", "",
        "**NOT READY — no SARIMAX training, Validation, or Final Test was run for this full-input experiment.** "
        "The processed SMP target covers 2023-01-01 00:00 through 2025-12-31 23:00 (26,304 hourly rows), but the selected exogenous input set does not. "
        "Existing results from the calendar-only implementation are an incomplete exploratory experiment, not the requested validation.", "",
        "The required complete hourly panel is `data/processed/model_inputs_2023_2025.csv`. It must contain all field columns listed in `feature_selection_report.md`, `timestamp`, and `max_source_available_at`. "
        "Every row must contain all selected values and must pass the as-of check. Observed actuals after forecast origin D must not appear in D+1..D+3 features; prior actuals must be represented as lags. Forecast inputs must be historical vintages available by D.", "",
        "## Failure diagnosis", "",
        "- **Network:** probes to official FRED, KMA, KPX, and KASI service/catalog hosts failed with connection errors before an HTTP response. This is a network/connection failure in the current environment.",
        "- **Authentication:** because no HTTP response was received, key acceptance/rejection could not be tested. There is no evidence that any configured key is invalid. No key value is recorded in this report.",
        "- **API specification:** some KPX catalog pages expose a base URL and summary but not the concrete operation route/output schema needed to implement the selected fields (notably `mlfd` and fuel-code series). This is a separate specification gap.",
        "- **Historical archive:** historical vintages/retention for day-ahead forecasts and the East-West forecast API were not verified. Connection failure is not evidence that historical querying is impossible. No complete archive is present locally.", "",
        "## Required inputs, current coverage, and action", "",
        "| Selected input | Current local coverage/status | Missing period / issue | Required action and official source |",
        "|---|---|---|",
        "| Actual hourly land SMP (target) | Complete 2023-01-01–2025-12-31, 26,304/26,304 | None for target | Preserve; source/export reference: [EPSIS hourly SMP](https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpShdChart.do?menuId=040202.) |",
        "| Actual hourly nationwide demand (lag only) | Local aligned 2023 file only, 8,760 hours; source mapping needs verification | 2024-01-01–2025-12-31 | Download 2024 and 2025 from [KPX nationwide hourly demand file](https://www.data.go.kr/data/15065266/fileData.do); verify source fields before import |",
        "| Day-ahead demand forecast `mlfd` | 2023 local daily demand forecast workbook exists; no verified issue-time vintage; none for 2024–2025 | 2023 vintage timestamps; all 2024-01-01–2025-12-31 | Obtain archived forecast vintages and exact operation/field mapping from [KPX SMP and demand forecast API](https://www.data.go.kr/data/15131225/openapi.do); API page alone does not prove 2023–25 vintage availability |",
        "| Generation by fuel: `tradeYmd`, `tradeNo`, `fuelTpCd`, `amgo` (actuals lagged) | 2023 aligned fuel columns exist but mapping to required raw fields unverified | 2024-01-01–2025-12-31 and 2023 field provenance | Import exact selected fuel codes and hourly intervals via [KPX fuel generation API](https://www.data.go.kr/data/15106744/openapi.do); obtain official operation schema if route is not supplied |",
        "| Nationwide hourly solar and wind generation (actuals lagged) | Regional CSV fragments only 2023-06-01–2023-11-30; one fragment has ~7.06% missing; existing Seoul/West columns are not nationwide substitutes | 2023-01-01–2023-05-31, 2023-12-01–2025-12-31, plus gaps | Obtain exact nationwide/regional solar and wind source exports; [KPX solar API](https://www.data.go.kr/data/15103243/openapi.do) documents solar only; wind requires its separate official dataset |",
        "| Day-ahead solar forecast `solgenKpx` | No local vintage data | Entire 2023-01-01–2025-12-31; archive availability unknown | Obtain archived KPX forecast vintages and field schema; [KPX SMP/demand API catalog](https://www.data.go.kr/data/15131225/openapi.do) or official KPX data contact |",
        "| Supply reserve and operating reserve, separately | 2023 daily supply reserve/rate workbook only; not hourly; no verified operating-reserve series | Hourly 2023-01-01–2025-12-31 | Export both distinct fields from [KPX reserve data](https://www.data.go.kr/data/15099819/fileData.do); if 3-month query limit applies, download 12 chunks: 2023-01-01–03-31, 04-01–06-30, 07-01–09-30, 10-01–12-31; repeat for 2024 and 2025 |",
        "| LNG generation fuel unit cost (EPSIS) | Local KOGAS monthly generation gas tariff has 8 rows but is not verified as the selected EPSIS field | Full 2023-01–2025-12, exact series unavailable | Download exact LNG generation fuel unit-cost export from [EPSIS fuel unit cost](https://epsis.kpx.or.kr/epsisnew/selectEkmaFucUpfChart.do?menuId=040100) |",
        "| Monthly fuel cost `fuelType`, `day`, `untpc`, `unit` | No verified exact-field monthly file | 2023-01–2025-12 (36 months) | Download full period from [KPX monthly fuel cost CSV](https://www.data.go.kr/data/15069386/fileData.do), retaining original units and monthly cadence |",
        "| WTI `DCOILWTICO`, Brent `DCOILBRENTEU` | FRED API not reachable; no complete local series confirmed | 2023-01-01–2025-12-31 | Restore network and retry FRED, or download observations: [WTI](https://fred.stlouisfed.org/series/DCOILWTICO), [Brent](https://fred.stlouisfed.org/series/DCOILBRENTEU) |",
        "| Japan LNG `PNGASJPUSDM`, Australian coal `PCOALAUUSDM` | FRED API not reachable; no complete local series confirmed | 2023-01–2025-12 (monthly) | Restore network and retry FRED, or download observations: [Japan LNG](https://fred.stlouisfed.org/series/PNGASJPUSDM), [Australian coal](https://fred.stlouisfed.org/series/PCOALAUUSDM) |",
        "| KMA ASOS observed `TA`, `SI`, `WS` (lag only) | 2023 local proxy has TA complete, WS 8,723/8,760, SI 4,781/8,760; station/source mapping not verified; no 2024–25 | Complete verified hourly records 2023-01-01–2025-12-31 | Retrieve exact variables by station and dates using [KMA ASOS API](https://apihub.kma.go.kr/apiList.do?seqApi=2) (`kma_sfctm3.php`, up to 31 days/query), or official export; 36 monthly chunks cover the period |",
        "| KMA forecast `TMP`, `WSD` with vintage | No local data | Entire period; historical vintage/archive availability unknown | Obtain issue-time historical forecast archive, or confirm archive export from [KMA short-range forecast spec](https://apihub.kma.go.kr/apiList.do?seqApi=10) (`nph-dfs_shrt_grd`) |",
        "| East-West forecast `icsrPred`, `temp`, `ws` with vintage | No local data; service key not verified/possibly service-specific | Entire 2023-01-01–2025-12-31; API page created 2025-07-17, earlier history/archive unknown | Request historical vintages from [East-West day-ahead solar forecast API](https://www.data.go.kr/data/15144541/openapi.do); if service requires a separate key, enter it locally as `EAST_WEST_SOLAR_SERVICE_KEY` in `.env` |",
        "| Holiday `locdate`, `isHoliday` | 2023-only file, 18 listed dates | 2024-01-01–2025-12-31 and validate full 2023 calendar | Collect 2023–25 from [KASI special date API](https://www.data.go.kr/data/15012690/openapi.do) or official annual calendar exports |",
        "| Calendar hour/weekday/month/season | Derivable for full hourly grid | None | Generate locally; does not replace the required selected inputs |", "",
        "## Credentials and safe retry", "",
        "`.env` variable names (values must remain local): `DATA_GO_KR_SERVICE_KEY`, `KMA_APIHUB_KEY`, `FRED_API_KEY`; optional service-specific names are `KMA_DATA_GO_SERVICE_KEY`, `KPX_FORECAST_SERVICE_KEY`, and `EAST_WEST_SOLAR_SERVICE_KEY`. Current connection errors do not justify rotating keys. Never paste keys into source code, logs, reports, or GitHub.",
        "A manual export should be saved locally under the project data import workflow, with source, exact fields, units, date range, and retrieval date recorded. Do not fill gaps with zero or substitute a different proxy series.", "",
        "## Execution status", "",
        "`python -m src.cli check-inputs` checks panel structure, field coverage, and row-level source-time bounds without fitting; it does not establish origin-specific availability for overlapping forecasts. `python -m src.cli backtest` currently refuses all fitting, including calendar-only fitting. Complete origin-specific vintage/leakage validation and full-input SARIMAX integration before producing Validation/Final Test.", "",
    ]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "data_collection_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_feature_report() -> None:
    path = PROCESSED / "hourly_smp_2023_2025.csv"
    from src.data_requirements import REQUIRED_MODEL_INPUTS
    lines = ["# Feature selection and leakage contract", "", "## Status: incomplete full-input experiment", "", "No selected input is being silently dropped. The current calendar-only SARIMAX is a legacy exploratory run and is not the requested model. No model may be trained until every field below is available over the complete 2023-01-01 00:00–2025-12-31 23:00 hourly grid and passes the input gate.", "", "## Selected inputs (all required)", "", "| Model panel column | Required source field / transformation | Availability rule |", "|---|---|---|",
      "| demand_actual_lag | Actual nationwide hourly demand | Lag only; no D+1 onward actual demand |", "| demand_forecast_vintage | KPX `mlfd` day-ahead demand forecast | Historical vintage issued no later than D |", "| generation_lng_lag / generation_coal_lag / generation_nuclear_lag / generation_other_fuels_lag | KPX `tradeYmd`, `tradeNo`, `fuelTpCd`, `amgo`, selected fuel groups | Historical actuals as lag features only |", "| solar_generation_national_lag / wind_generation_national_lag | Nationwide/regional hourly solar and wind generation, aggregated only as specified | Historical actuals as lag features only |", "| solar_generation_forecast_vintage | KPX `solgenKpx` | Historical forecast vintage available at D |", "| reserve_supply_lag / reserve_operating_lag | Supply reserve and operating reserve as separate series | Historical actuals as lag features only; never conflate |", "| lng_generation_fuel_unit_cost_available | EPSIS LNG generation fuel unit cost | Preserve source cadence; only available/published values |", "| monthly_fuel_cost_available | `fuelType`, `day`, `untpc`, `unit` | Monthly cadence; retain units and conservative publication lag |", "| wti_available / brent_available | FRED `DCOILWTICO` / `DCOILBRENTEU` | Publication-aware daily values |", "| japan_lng_available / australia_coal_available | FRED `PNGASJPUSDM` / `PCOALAUUSDM` | Monthly cadence and conservative publication lag |", "| weather_ta_lag / weather_si_lag / weather_ws_lag | KMA ASOS `TA`, `SI`, `WS` | Actuals as lag features only |", "| weather_forecast_tmp_vintage / weather_forecast_wsd_vintage | KMA `TMP`, `WSD` | Historical forecast vintage issued by D |", "| east_west_icsrpred_vintage / east_west_temp_vintage / east_west_ws_vintage | East-West `icsrPred`, `temp`, `ws` | Historical forecast vintage issued by D |", "| is_holiday | KASI `locdate`, `isHoliday` | Complete 2023–2025 calendar |", "| calendar | hour, weekday, month, season | Deterministic |", "", "## Leakage and completeness rules", "", "- At origin D, no observed value timestamped after D may enter a D+1..D+3 forecast. Actual observations are only legal as explicitly named lagged features.", "- Forecast inputs use the vintage that was available by D, not a subsequently revised/current forecast.", "- Monthly/daily series retain source cadence and use a documented publication lag; no zero fill or proxy substitution.", "- `data/processed/model_inputs_2023_2025.csv` must include `timestamp`, `max_source_available_at`, and every required input column. All 26,304 hourly rows and all selected columns must be complete; source availability must not postdate timestamp.", "- Missing one variable or period blocks model fitting. See [data collection report](data_collection_report.md) for gaps, failure categories, official sources, and required actions.", ""]
    if path.exists():
        df = pd.read_csv(path, parse_dates=["timestamp"])
        lines += [f"Data coverage observed: {df.timestamp.min()} to {df.timestamp.max()}; target rows={df.smp.notna().sum()}/{len(df)}; SMP missing rate={df.smp.isna().mean():.3%}.", ""]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "feature_selection_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_model_report() -> None:
    exp = OUT / "metrics" / "validation_experiments.json"
    lines = ["# Model experiment report", "", "**Status: incomplete exploratory calendar-only experiment; not a full-input SARIMAX evaluation.** The project now blocks any new model fitting until all selected inputs are complete and point-in-time safe. Existing experiment artifacts are retained only as historical context.", "", "SARIMAX seasonal period is fixed at 24 hours. The legacy experiment used calendar-only exogenous features and omitted all selected operational and market inputs.", "", "## Legacy Validation candidates", ""]
    if exp.exists():
        lines += ["```json", exp.read_text(encoding="utf-8"), "```", "", "These are legacy calendar-only metrics. They are not Validation/Final Test results for the selected-input experiment and must not be used as its conclusion.", ""]
    else:
        lines += ["No new full-input model experiment has completed. First satisfy every requirement in the data collection report; the current calendar-only implementation remains disabled.", ""]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "model_experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_final_report() -> None:
    metric_path = OUT / "metrics" / "metrics_by_split.csv"
    data_path = PROCESSED / "hourly_smp_2023_2025.csv"
    lines = ["# Evaluation status", "", "## Full-input experiment: BLOCKED", "", "No Validation or Final Test was run for the selected-input SARIMAX because required explanatory data are incomplete and some forecast-vintage histories remain unverified. This is an intentional fail-closed stop. See [data collection report](data_collection_report.md) for the input-by-input gap table and official retrieval actions.", "", "## Legacy calendar-only experiment: incomplete, exploratory only", "", "The metrics currently stored in the output directory came from the prior calendar-only SARIMAX. They are not complete-input Validation or Final Test results and must not be represented as the requested final model. No new results were generated in this run.", "", "Target SMP is complete from 2023-01-01 through 2025-12-31 (26,304 hourly observations); target availability alone is insufficient to authorize training.", ""]
    if data_path.exists():
        d = pd.read_csv(data_path, parse_dates=["timestamp"])
        lines += [f"Observed target coverage in processed file: {d.timestamp.min()} through {d.timestamp.max()}. Total hourly grid rows: {len(d)}; observed SMP: {int(d.smp.notna().sum())}; missing SMP: {int(d.smp.isna().sum())} ({d.smp.isna().mean():.2%}).", ""]
        for year, part in d.groupby(d.timestamp.dt.year): lines.append(f"- {year}: {int(part.smp.notna().sum())}/{len(part)} observed hourly SMP")
        lines.append("")
    if metric_path.exists():
        m = pd.read_csv(metric_path)
        overall = m[m.group.isin(["overall_sarimax", "previous_day_smp", "previous_week_smp"])][["split", "group", "count", "mae", "rmse", "r2"]]
        table = ["| " + " | ".join(overall.columns) + " |", "|" + "|".join(["---"] * len(overall.columns)) + "|"]
        table += ["| " + " | ".join(str(v) for v in row) + " |" for row in overall.itertuples(index=False, name=None)]
        lines += ["## Archived legacy metrics (calendar-only; not valid full-input results)", "", *table, "", "These values are preserved for provenance only. They were produced without the selected exogenous variables and do not satisfy the current input requirements. Do not cite them as the project's Validation or Final Test performance.", ""]
    else:
        lines += ["No new Validation or Final Test metrics exist for the full-input experiment. Target coverage and exogenous input coverage are assessed separately; see the data collection report.", ""]
    lines += ["## Figures", "", "Existing figures, if any, belong to the archived calendar-only experiment and are not full-input model results.", ""]
    prediction_files = [OUT / "predictions" / "validation_predictions.csv", OUT / "predictions" / "final_test_predictions.csv"]
    prediction_frames = [pd.read_csv(p, parse_dates=["target_timestamp", "forecast_origin"]) for p in prediction_files if p.exists()]
    if prediction_frames:
        predictions = pd.concat(prediction_frames, ignore_index=True).dropna(subset=["absolute_error"])
        worst = predictions.nlargest(10, "absolute_error")
        lines += ["Largest individual forecast errors across validation and final test (descriptive only; no causal attribution):", "", "| split | forecast origin | target time (KST) | lead | actual SMP | predicted SMP | absolute error |", "|---|---|---|---|---:|---:|---:|"]
        lines += [f"| {r.split} | {r.forecast_origin} | {r.target_timestamp} | {r.forecast_day} hour {int(r.horizon)} | {r.actual_smp:.2f} | {r.predicted_smp:.2f} | {r.absolute_error:.2f} |" for r in worst.itertuples()]
        lines.append("")
    else:
        lines += ["No prediction rows are available yet. Any causal explanation of future large errors must be labelled as a hypothesis unless verified against contemporaneous source records.", ""]
    lines += ["## Next step", "", "Complete every field and missing period in the collection report, preserve source publication/issue timestamps, build the full panel, and finish full-input SARIMAX wiring before running `python -m src.cli backtest`. Calendar-only SARIMAX remains an incomplete exploratory experiment.", ""]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "final_evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_all() -> None:
    write_collection_report(); write_feature_report(); write_model_report(); write_final_report()
