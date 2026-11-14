# Feature selection report

## Selected model inputs

- Calendar only: hour, weekday, month, and annual seasonality (sin/cos terms). These features are known for all forecast timestamps.
- Target: official land SMP, aligned to KST hour-start timestamps. For sources with 01–24 interval-end labels, hour 01 maps to 00:00–01:00 and hour 24 maps to 23:00–24:00.

## Excluded candidate inputs

| Candidate | Decision | Reason |
|---|---|---|
| Actual demand, generation, reserve, ASOS weather | Excluded from future exog | Future actuals are unavailable at forecast origin; no point-in-time archive was verified for every origin. |
| KPX mlfd demand forecast | Excluded pending archive verification | API refresh and historical vintage issue-time availability are not independently verified for every origin. |
| KMA short-range weather forecast | Excluded pending vintage verification | Current API forecasts do not establish the historical forecast vintage that would have been available at each old origin. |
| East-West solar forecast fields | Excluded | Separate service credential was not supplied and historical vintage/field mapping remains unverified. |
| Fuel costs and FRED oil/LNG/coal | Excluded from current model | Monthly/daily series require verified publication lags and alignment. FRED data are retained separately and never expanded as independent hourly observations. |
| Holiday indicator | Excluded pending complete calendar source | Existing holiday file is limited to 2023; no complete authenticated calendar for 2023–2025 is currently assembled. |

No requested field is replaced with a similar field name. The selected exogenous matrix is deliberately limited to calendar features to avoid look-ahead leakage.

Data coverage observed: 2023-01-01 00:00:00 to 2025-12-31 23:00:00; target rows=26304/26304; SMP missing rate=0.000%.
