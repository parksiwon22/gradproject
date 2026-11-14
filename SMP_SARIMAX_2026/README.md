# SMP_SARIMAX_2026

Python pipeline for 72-hour land SMP forecasts with a seasonal SARIMAX model. It is isolated from the existing V5/V6 and web projects. It reads source files in the shared folder without changing them and copies selected files into this project's ignored `data/raw/` folder.

## Setup

1. Create a virtual environment and install dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\python -m pip install -r requirements.txt
   ```

2. Keep credentials in `.env` (already locally configured for this workspace). `.env.example` lists supported names; `.gitignore` excludes `.env`.
3. Export the official EPSIS hourly land-SMP CSV or Excel for 2023–2025 and set `EPSIS_EXPORT_DIR` in `.env`. Existing local data are picked up through `EXISTING_SMP_DATA_DIR`.

## Commands

Run from this folder:

```powershell
python -m src.cli collect
python -m src.cli preprocess
python -m src.cli backtest
python -m src.cli visualize
python -m src.cli report
```

`check-inputs` checks the complete selected input panel and source-time leakage contract without training. `backtest` is fail-closed and currently disabled: it cannot fit the legacy calendar-only model or produce Validation/Final Test scores. Full-input SARIMAX wiring is required after every selected input is collected. Missing fields are never zero-filled or replaced by proxies.

`visualize` creates monthly comparisons, Validation and Final Test six-month hourly overview graphs, and error-analysis plots. Per-day prediction graphs are omitted per the user's instruction. CSV predictions retain distinct forecast origins, target timestamps, and leads.

## Input and leakage policy

All previously selected inputs are required: hourly actual demand and selected generation/reserve series as lagged values; archived KPX demand/solar forecasts; EPSIS/KPX fuel costs; FRED WTI, Brent, Japan LNG, and Australian coal; KMA observed weather as lags and historical forecast vintages; East-West forecast vintages; and KASI holidays. Actual observations after origin D are prohibited for D+1..D+3. Forecast vintages must have been issued by D. Monthly series retain cadence and publication lag. See `reports/data_collection_report.md` for current gaps, exact date ranges, failure diagnosis, credential variable names, and official downloads. Existing calendar-only SARIMAX outputs are incomplete exploratory results.

## Outputs

- `data/processed/hourly_smp_2023_2025.csv`: fixed hourly grid with missing target periods left blank.
- `outputs/predictions/`: Validation and Final Test forecast rows.
- `outputs/metrics/`: overall, per-day, per-month, per-lead, and per-horizon metrics plus model search log.
- `outputs/figures/monthly/`, `six_months/`, `errors/`: PNG and interactive HTML visualizations (no daily comparison graphs).
- `reports/`: collection, variable selection, experiment, and final evaluation reports.

No 2024–2025 SMP result is claimed until those actual data have been imported and the backtest runs successfully.
