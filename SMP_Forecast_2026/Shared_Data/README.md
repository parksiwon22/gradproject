# Shared data

Both models read the existing local SMP_XGBOOST_2026 feature panel and its underlying hourly target, calendar, and Seoul ASOS panels. No raw or processed source dataset is copied into this repository. The LightGBM run records file sizes and SHA-256 hashes in [data_manifest.csv](data_manifest.csv).

The source panel was 13,308,506 bytes when used. It contains 26,112 hourly rows from 2023-01-09 through 2025-12-31, 27 model features, actual SMP, forecast-origin timestamps, and source-cutoff audit fields. It was kept local because redistribution terms for the inherited SMP and ASOS extracts have not been confirmed.

## Local source paths

From the workspace root:

- SMP_XGBOOST_2026/data/processed/day_ahead_supervised_panel_2023_2025.csv
- SMP_XGBOOST_2026/data/processed/calendar_features_hourly_2023_2025.csv
- SMP_XGBOOST_2026/data/processed/kma_asos_108_hourly_2023_2025.csv
- SMP_SARIMAX_2026/data/processed/hourly_smp_2023_2025.csv

Set SMP_XGBOOST_PROJECT and SMP_SARIMAX_PROJECT if your local paths differ. Then run python LightGBM/src/models/lightgbm_day_ahead.py from this project root. The script reads the saved XGBoost panel and rebuilds the panel in memory to confirm every row and column is identical before fitting.

## Official sources and retrieval notes

- SMP target: [EPSIS hourly SMP](https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpShdChart.do?menuId=040202). EPSIS provides CSV/XLSX export controls for the hourly SMP view. The source-project report says its hourly target series is complete for 2023-01-01–2025-12-31, but the inherited labels do not retain price-product or publication-vintage fields. The original extraction log was not available in this project, so the exact export selections cannot be fully reconstructed.
- Weather: [KMA ASOS API](https://apihub.kma.go.kr/apiList.do?seqApi=2) and [KMA ASOS observation availability](https://data.kma.go.kr/data/grnd/selectAsosList.do?pgmNo=36). The existing normalized panel covers station 108 in hourly monthly requests. The original project records 26,269 of 26,304 timestamps and leaves unavailable cells missing. The KMA API requires a personal authentication key; no key is stored here.
- Calendar: generated deterministically from timestamps and the existing normalized calendar panel.

EPSIS's hourly SMP page offers exports, but an explicit redistribution licence was not established for the inherited extract. KMA states that API weather data are subject to the applicable Korea Open Government License (Korea Open Government License, or KOGL) conditions; the specific licence type for this ASOS extract was not recorded. Since redistribution permission for the two source extracts is not confirmed, neither they nor the derived full panel are uploaded. Recheck each dataset's current licence and attribution requirements before redistributing it. The official KMA API terms and service information are at [API Hub service information](https://apihub.kma.go.kr/apiInfo.do) and [API Hub terms](https://apihub.kma.go.kr/policy.do).

This comparison run reuses the local files and does not call an API or require a key. Their hashes, byte sizes, official-source links, retrieval notes, and local paths are in data_manifest.csv.
