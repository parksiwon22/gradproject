from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import OUT, PROCESSED, RAW, REPORTS


def write_collection_report() -> None:
    manifest = RAW / "collection_manifest.json"
    records = json.loads(manifest.read_text(encoding="utf-8")).get("records", []) if manifest.exists() else []
    lines = ["# Data collection report", "", "Collection used read-only copies of existing sources; API keys are not printed.", "", "| Data | Status | Period / rows | Frequency | Unit | Missing rate | Notes |", "|---|---|---|---|---|---:|---|"]
    for r in records:
        period = f"{r.get('period_start', '—')} – {r.get('period_end', '—')} / {r.get('rows', '—')} rows"
        missing = r.get("smp_missing_rate", r.get("missing_rate"))
        missing = f"{missing:.2%}" if isinstance(missing, (int, float)) else "—"
        unit = r.get("unit", r.get("units", "unknown"))
        note = str(r.get("reason", r.get("inspection_reason", r.get("note", r.get("inspection_status", ""))))).replace("|", "/")
        if r.get("series_page"):
            note = f"[official FRED page]({r['series_page']}); API={r.get('api_status',r.get('status','unknown'))}; {note}"
        lines.append(f"| {r.get('dataset','')} | {r.get('status','')} | {period} | {r.get('frequency','unknown')} | {unit} | {missing} | {note} |")
    if not records: lines += ["", "No collection manifest exists yet. Run `python -m src.cli collect`."]
    lines += ["", "## Official specifications confirmed", "", "- [KPX SMP and day-ahead demand](https://www.data.go.kr/data/15131225/openapi.do): hourly interval values are labeled by interval end; updated daily around 23:00 KST. The public page currently does not expose the operation path/schema in its page text, so automated historical querying is not attempted.", "- [KMA APIHub ASOS interval endpoint](https://apihub.kma.go.kr/apiList.do?apiSeq=5): `kma_sfctm3.php`, maximum 31-day query; selected fields are `TA` (°C), `SI` (MJ/m²), `WS` (m/s). Collection uses Seoul station 108 as a single-station weather proxy.", "- [FRED observations API documentation](https://fred.stlouisfed.org/docs/api/fred/series_observations.html) confirms the observations API method. The local execution environment could not resolve external hosts, so API-key validity/callability remains unverified.", "- Official FRED pages confirm WTI and Brent are daily USD/barrel, and Japan LNG and Australian coal are monthly USD/million metric Btu and USD/metric ton respectively. Series pages are linked in the table above.", "", "## Important data notes", "", "- EPSIS hourly SMP uses the 2023 read-only local file when available. Official EPSIS CSV/Excel can be added under `EPSIS_EXPORT_DIR`; missing 2024–2025 intervals are not synthesized.", "- The documented KPX day-ahead API refreshes once daily around 23:00 and timestamps each interval by its end. Only archived forecast vintages available at the forecast origin may be used.", "- KMA observed weather is retained as raw data, but excluded from future exogenous variables to prevent use of post-origin observations.", "- KMA forecast vintages and East-West solar prediction vintages are not used unless their historical issue-time availability can be demonstrated.", "- FRED monthly series retain monthly frequency and original units. A one-month publication delay is assumed for any later model use.", ""]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "data_collection_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_feature_report() -> None:
    path = PROCESSED / "hourly_smp_2023_2025.csv"
    lines = ["# Feature selection report", "", "## Selected model inputs", "", "- Calendar only: hour, weekday, month, and annual seasonality (sin/cos terms). These features are known for all forecast timestamps.", "- Target: official land SMP, aligned to KST hour-start timestamps. For sources with 01–24 interval-end labels, hour 01 maps to 00:00–01:00 and hour 24 maps to 23:00–24:00.", "", "## Excluded candidate inputs", "", "| Candidate | Decision | Reason |", "|---|---|---|", "| Actual demand, generation, reserve, ASOS weather | Excluded from future exog | Future actuals are unavailable at forecast origin; no point-in-time archive was verified for every origin. |", "| KPX mlfd demand forecast | Excluded pending archive verification | API refresh and historical vintage issue-time availability are not independently verified for every origin. |", "| KMA short-range weather forecast | Excluded pending vintage verification | Current API forecasts do not establish the historical forecast vintage that would have been available at each old origin. |", "| East-West solar forecast fields | Excluded | Separate service credential was not supplied and historical vintage/field mapping remains unverified. |", "| Fuel costs and FRED oil/LNG/coal | Excluded from current model | Monthly/daily series require verified publication lags and alignment. FRED data are retained separately and never expanded as independent hourly observations. |", "| Holiday indicator | Excluded pending complete calendar source | Existing holiday file is limited to 2023; no complete authenticated calendar for 2023–2025 is currently assembled. |", "", "No requested field is replaced with a similar field name. The selected exogenous matrix is deliberately limited to calendar features to avoid look-ahead leakage.", ""]
    if path.exists():
        df = pd.read_csv(path, parse_dates=["timestamp"])
        lines += [f"Data coverage observed: {df.timestamp.min()} to {df.timestamp.max()}; target rows={df.smp.notna().sum()}/{len(df)}; SMP missing rate={df.smp.isna().mean():.3%}.", ""]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "feature_selection_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_model_report() -> None:
    exp = OUT / "metrics" / "validation_experiments.json"
    lines = ["# Model experiment report", "", "SARIMAX seasonal period is fixed at 24 hours. Candidate order space is intentionally small; each model uses the same calendar-only exogenous features.", "", "## Validation candidates", ""]
    if exp.exists():
        lines += ["```json", exp.read_text(encoding="utf-8"), "```", "", "The candidate with the lowest walk-forward validation MAE was selected. During a split, model parameters are fixed; observed SMP is appended only after each 72-hour forecast origin advances, with refitting disabled.", ""]
    else:
        lines += ["No model experiment has completed. The required train/validation SMP coverage is incomplete or statsmodels is not installed. Run `python -m src.cli backtest` after resolving the reported blocker.", ""]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "model_experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_final_report() -> None:
    metric_path = OUT / "metrics" / "metrics_by_split.csv"
    data_path = PROCESSED / "hourly_smp_2023_2025.csv"
    lines = ["# Final evaluation report", "", "## 1. Research objective and period", "", "Forecast land SMP in KRW/kWh at 23:00 KST from D+1 00:00 through D+3 23:00 (72 hours). Requested period: 2023-01-01 through 2025-12-31; train through 2024-12-31, validation 2025-01-01 through 2025-06-30, final test 2025-07-01 through 2025-12-31.", "", "## 2–3. Data and preprocessing", "", "See [data collection report](data_collection_report.md) and [feature selection report](feature_selection_report.md). Missing target SMP is never replaced with zero or model output. Post-origin actual demand, generation, weather, and SMP are not supplied as future exogenous variables.", ""]
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
        lines += ["## 4–6. Model and performance", "", "Validation-selected SARIMAX is reported in `artifacts/selected_model.json`. Split metrics are below; D+1/D+2/D+3 details are in the metrics CSV.", "", *table, "", "The selected candidate did not converge within the configured 25 optimizer iterations. SARIMAX also scored worse than the previous-day and previous-week persistence baselines on both validation and final test, so these results are preliminary and should not be presented as a successful forecast model. Detailed per-day and horizon metrics are saved in `outputs/metrics/metrics_by_split.csv`. Forecast rows remain distinct by `forecast_origin` and `target_timestamp` in `outputs/predictions/`.", ""]
    else:
        lines += ["## 4–6. Model and performance", "", "No Validation or Final Test performance is reported because the target series does not yet cover the requested split. The current local source inventory contains an existing 2023 aligned hourly series; it does not establish 2024 and 2025 EPSIS land SMP. Validation/test scores and figures are intentionally absent rather than fabricated.", "", "After importing official EPSIS exports, run `python -m src.cli preprocess`, `python -m src.cli backtest`, and `python -m src.cli report`.", ""]
    lines += ["## 7. Graphs", "", "Monthly actual-versus-predicted graphs (one per month), six-month hourly overview graphs (interactive HTML and PNG), and error-analysis graphs are in `outputs/figures/`. Daily 72-hour comparison graphs were omitted at the user's request.", "", "## 8. Large-error forecasts", ""]
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
    lines += ["## 9. Limitations and extensions", "", "Current exogenous variables are restricted to deterministic calendar features because historical point-in-time vintages for operating forecasts/weather were not established. The train, validation, and final-test date windows are fixed. Add complete source data, preserve publication vintages, and rerun the same SARIMAX protocol before interpreting performance. XGBoost, LightGBM, LSTM, and web integration are outside this project's scope.", ""]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "final_evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_all() -> None:
    write_collection_report(); write_feature_report(); write_model_report(); write_final_report()
