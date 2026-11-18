# Data collection and shared dataset audit

Dataset version: `smp72h_common_v5_generation_v2_xgb_20261119`  
Source branch: `codex/smp-72h-data` at `a90141b81ecd841c524eb75a72f4bcc4f39989f5`  
Shared evaluation origins: 1090 ({'final_test': 182, 'train': 729, 'validation': 179})  
Targets: 78480; missing SMP labels: 0.

| Group | Use and verified scope |
|---:|---|
| 1 | EPSIS actual hourly load, mean of 12 five-minute snapshots, MW; 1,095 hourly gaps retained and imputed for model input. |
| 2 | Complete mainland day-ahead demand forecast history; publication time unverified, so only historical forecast values up to the shared cutoff are used. |
| 3 | KASI official holidays, separate from weekends. |
| 4 | Asia/Seoul calendar and hour/weekday/month cyclical features; checked against the supplied calendar panel. |
| 5 | KPX generation input `20261119_v2`: 26,304 hourly rows for 2023–2025 from annual 2023, API 2024, and annual 2025 sources. Includes LNG, bituminous coal, anthracite coal, nuclear, and combined coal generation (MWh); 2024 source gaps remain missing and flagged. |
| 6 | Official regional solar/wind sources; Jeju excluded where explicit; missing provincial sums remain missing; PPA/BTM coverage is not asserted. |
| 7 | Supply reserve MW and percent; 5-minute means require all 12 samples, 1,085 absent samples and 676 incomplete hours retained as missing. |
| 8 | Domestic LNG generation fuel cost in KRW/kWh matched by target month under the agreed conditional experiment. |
| 9 | FRED WTI and Brent values selected from the source vintage panel by D-2 as-of date. |
| 10 | FRED international LNG and coal monthly values selected from the source vintage panel by D-2 as-of date. |
| 11 | Seoul ASOS 108 temperature, wind and solar observations; night flag, observed zero and missing SI are separate. |
| 12 | KMA Seoul temperature and metro weather/precipitation, selecting the latest archived issue no later than the origin; issue time, target day and lead are retained. |

## Critical data choices

- Group 1 uses the EPSIS general current-load page. Its historical rows have no explicit geography column; the separate Jeju monitoring page supports treating the general system page as mainland. The old KPX nationwide-plus-Jeju hourly file was hashed and explicitly excluded.
- EPSIS dates 2023-07-02, 2024-03-21, 2024-03-22, and 2025-02-28 returned no rows; their intervals remain missing.
- Group 5 uses the specified `20261119_v2` hourly panel, with source kinds `annual_2023`, `api_2024`, and `annual_2025`. It covers the full 26,304-hour 2023–2025 timeline. Its 2024 gaps remain NaN in the raw panel and are flagged; no missing generation is changed to zero. The source defines market-traded sent-out energy and does not distinguish mainland from Jeju.
- Group 6 originals span 2023-01 through 2025-12 in seven official CSVs: 2023 Jan–Feb, Mar–May, Jun–Aug, Sep–Nov, and Dec; full 2024; and full 2025. Regional definitions vary by year and are recorded in the source manifest.
- Group 7 has 1085 missing five-minute observations across the complete expected 2023–2025 range. Only full 12-sample hours receive an hourly mean.
- EPSIS and reserve gaps were compared by interval end: 675 shared hours, 420 demand-only, and 1 reserve-only. Timestamp details are in `demand_reserve_missing_timestamps.csv`.
- Group 11 has 11921 missing solar observations, including 11887 during calculated night and 34 during daylight. Missing values remain missing; observed zero counts separately.
- Group 12 forecast availability by D+1/D+2/D+3 is in `quality_report.json`. Exact-target forecasts use the latest issue at or before the origin; absent forecasts are not replaced with future observations.

## Availability and missing data

The un-imputed source panel is `hourly.csv`. `forecast_inputs.csv` has one row per origin and target hour. It retains a `__is_missing` flag for each numeric feature; Train-only medians fill columns with observed Train values, and all-missing Train columns remain NaN for XGBoost's native missing path. Selected KMA issue timestamps are kept as metadata columns, excluded from model features.

## Leakage checks

- Every issue has exactly 72 targets and one `observed_cutoff = forecast_origin - 72h`.
- History uses interval-end timestamps and only values at or before that common cutoff.
- A future-observation perturbation check passed for 18 history source series at three issue cutoffs each (54 feature comparisons).
- Calendar alignment and the D-2 FRED and KMA issue-time checks passed.
- Domestic LNG is the one intentionally conditional target-month input; it is not claimed available at the historical issue time.

## Limitations

The 72-hour observation cutoff is a conservative assumption because each source's actual publication timestamp is not verified. EPSIS geography is inferred from the general system page and separately exposed Jeju page, not row-coded. Demand-forecast issue timestamps are unknown. Generation and renewable market scope may exclude PPA, behind-the-meter or other non-market production. Domestic LNG same-month cost is a conditional historical experiment only.
