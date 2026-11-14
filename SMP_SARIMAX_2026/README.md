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

`backtest` fits the limited candidate set on the fixed training period, chooses by Validation MAE, and uses a fixed-parameter daily state update for Validation and Final Test. It requires all hourly target SMP values in each prescribed split. It never fills missing SMP with zero.

`visualize` creates monthly comparisons, Validation and Final Test six-month hourly overview graphs, and error-analysis plots. Per-day prediction graphs are omitted per the user's instruction. CSV predictions retain distinct forecast origins, target timestamps, and leads.

## Input and leakage policy

Calendar features are the only exogenous variables currently included in the model. Actual demand, generation, reserve, weather, and SMP after each origin are never used as future inputs. Observed KMA weather and FRED data are archived separately; they are not included until an auditable point-in-time forecast/publication vintage and lag are available. Monthly values preserve their source cadence, and `src.preprocessing.monthly.merge_monthly_with_conservative_lag` applies a conservative two-month start-of-month availability rule for a one-month release delay.

## Outputs

- `data/processed/hourly_smp_2023_2025.csv`: fixed hourly grid with missing target periods left blank.
- `outputs/predictions/`: Validation and Final Test forecast rows.
- `outputs/metrics/`: overall, per-day, per-month, per-lead, and per-horizon metrics plus model search log.
- `outputs/figures/monthly/`, `six_months/`, `errors/`: PNG and interactive HTML visualizations (no daily comparison graphs).
- `reports/`: collection, variable selection, experiment, and final evaluation reports.

No 2024–2025 SMP result is claimed until those actual data have been imported and the backtest runs successfully.
