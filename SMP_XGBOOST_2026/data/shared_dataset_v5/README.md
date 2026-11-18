# Shared dataset smp72h_common_v5_generation_v2_xgb_20261119

This version was built from `gradproject` branch `codex/smp-72h-data` at a90141b81ecd841c524eb75a72f4bcc4f39989f5 using the selected files listed in `manifest.json`.

## Target and time rules

- Forecast origin: local D 23:00 (Asia/Seoul).
- Targets: D+1 00:00 through D+3 23:00, 72 hourly values.
- One observed cutoff per issue: origin minus 72 hours. Observed input groups and historical SMP use the same cutoff; interval aggregates are available only after their interval end.
- Evaluation blocks crossing split boundaries are excluded. Train: 2023-01 through 2024-12; Validation: 2025-01 through 2025-06; Final Test: 2025-07 through 2025-12.

## Tables

- `hourly.csv`: raw hourly panel. It preserves source NaNs and adds missing flags; observed interval aggregates are aligned to their corresponding interval start for presentation.
- `forecast_inputs.csv`: one row per forecast origin and target hour. The repeated historical features are computed at the origin's common cutoff. Numeric missing values are filled using per-feature Train median; `__is_missing` columns retain the pre-fill state. All-missing Train features remain NaN for XGBoost's native handling.
- `evaluation_origins.csv`: the common issue schedule and cutoffs.
- `evaluation_targets.csv`: fixed common target schedule, actual land SMP, and daily-repeat/weekly baselines.
- `demand_reserve_missing_timestamps.csv`: hourly interval-end comparison showing which EPSIS demand and reserve aggregates are missing.
- `feature_manifest.csv`: input source, group, units, availability, pre-imputation missing rate, and imputation value.
- `quality_report.json`: source quality, coverage, leakage checks and limitations.

## Important conditions

The domestic LNG feature is matched by target month under the agreed conditional historical experiment. It is not known to have been available at issue time and is not an operational forecast claim. Historical demand forecasts are used only as past forecast history because their publication timestamps are unverified. Medium-range weather is selected by archived issue time and exact target day; absence is retained.

Actual observation release times and hourly SMP publication times were not verified. The 72-hour cutoff is a conservative study assumption, not a claim that every source is published within 72 hours. The EPSIS comparison page provides five-minute current-load samples but no row-level geography field; the separate Jeju page supports treating the general page as mainland with this limitation recorded.
