# Forecast-distance bias and error analysis

Bias is `mean(predicted_smp - actual_smp)`: positive means overprediction; negative means underprediction.
Prediction standard deviation is compared with actual standard deviation to show compression or excess spread.
High/low-price and sharp-change slices are descriptive post-evaluation slices based on actual values; they are not used to choose the model or weights.

| Split | Experiment | Bias | Overprediction share | Underprediction share | Actual SD | Predicted SD | SD ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| final_test | E0_xgb | 18.5129 | 0.9040 | 0.0960 | 23.0657 | 20.3215 | 0.8810 |
| final_test | E1_recent_daily_blend | 15.0323 | 0.8751 | 0.1249 | 23.0657 | 19.6217 | 0.8507 |
| final_test | E1_selected_blend | 13.3395 | 0.8586 | 0.1414 | 23.0657 | 19.9301 | 0.8641 |
| final_test | E1_weekly_blend | 13.3395 | 0.8586 | 0.1414 | 23.0657 | 19.9301 | 0.8641 |
| final_test | E2_train_plus_validation | 11.7275 | 0.7861 | 0.2139 | 23.0657 | 16.3487 | 0.7088 |
| final_test | E3_retrained_blend | 8.5897 | 0.7497 | 0.2503 | 23.0657 | 17.1503 | 0.7435 |
| final_test | baseline_recent_daily | 1.1096 | 0.5188 | 0.4780 | 23.0657 | 22.9007 | 0.9928 |
| final_test | baseline_weekly | 1.2681 | 0.5277 | 0.4663 | 23.0657 | 22.8934 | 0.9925 |
| validation | E0_xgb | -1.1431 | 0.4706 | 0.5294 | 20.7679 | 19.6070 | 0.9441 |
| validation | E1_recent_daily_blend | -1.0275 | 0.4744 | 0.5256 | 20.7679 | 18.4112 | 0.8865 |
| validation | E1_selected_blend | -0.9635 | 0.4796 | 0.5204 | 20.7679 | 18.8217 | 0.9063 |
| validation | E1_weekly_blend | -0.9635 | 0.4796 | 0.5204 | 20.7679 | 18.8217 | 0.9063 |
| validation | baseline_recent_daily | -0.5651 | 0.4932 | 0.5041 | 20.7679 | 21.2731 | 1.0243 |
| validation | baseline_weekly | -0.5442 | 0.4969 | 0.5005 | 20.7679 | 21.3071 | 1.0260 |

## D+1/D+2/D+3 MAE

| Split | Experiment | Day | MAE | RMSE | R² | Bias |
|---|---|---:|---:|---:|---:|---:|
| final_test | E0_xgb | D+1 | 19.9978 | 23.8607 | -0.0688 | 18.4676 |
| final_test | E0_xgb | D+2 | 19.9769 | 23.8677 | -0.0708 | 18.4309 |
| final_test | E0_xgb | D+3 | 20.2244 | 24.1408 | -0.0969 | 18.6402 |
| final_test | E1_recent_daily_blend | D+1 | 16.9597 | 20.8129 | 0.1868 | 14.9650 |
| final_test | E1_recent_daily_blend | D+2 | 17.1200 | 20.9200 | 0.1774 | 14.9630 |
| final_test | E1_recent_daily_blend | D+3 | 17.2878 | 21.0766 | 0.1639 | 15.1687 |
| final_test | E1_selected_blend | D+1 | 15.7894 | 19.5118 | 0.2853 | 13.3102 |
| final_test | E1_selected_blend | D+2 | 15.8287 | 19.5742 | 0.2798 | 13.2753 |
| final_test | E1_selected_blend | D+3 | 16.0019 | 19.7733 | 0.2641 | 13.4329 |
| final_test | E1_weekly_blend | D+1 | 15.7894 | 19.5118 | 0.2853 | 13.3102 |
| final_test | E1_weekly_blend | D+2 | 15.8287 | 19.5742 | 0.2798 | 13.2753 |
| final_test | E1_weekly_blend | D+3 | 16.0019 | 19.7733 | 0.2641 | 13.4329 |
| final_test | E2_train_plus_validation | D+1 | 15.1030 | 19.2223 | 0.3064 | 11.6123 |
| final_test | E2_train_plus_validation | D+2 | 15.1795 | 19.3030 | 0.2996 | 11.7201 |
| final_test | E2_train_plus_validation | D+3 | 15.3187 | 19.3729 | 0.2936 | 11.8502 |
| final_test | E3_retrained_blend | D+1 | 12.8843 | 16.7258 | 0.4748 | 8.5115 |
| final_test | E3_retrained_blend | D+2 | 12.9523 | 16.8085 | 0.4690 | 8.5778 |
| final_test | E3_retrained_blend | D+3 | 13.0526 | 16.8712 | 0.4643 | 8.6798 |
| final_test | baseline_recent_daily | D+1 | 14.5026 | 21.2939 | 0.1488 | 0.9547 |
| final_test | baseline_recent_daily | D+2 | 13.2877 | 19.9414 | 0.2526 | 1.0916 |
| final_test | baseline_recent_daily | D+3 | 11.4722 | 17.8236 | 0.4021 | 1.2824 |
| final_test | baseline_weekly | D+1 | 11.3917 | 17.7093 | 0.4113 | 1.2763 |
| final_test | baseline_weekly | D+2 | 11.4362 | 17.7814 | 0.4057 | 1.2457 |
| final_test | baseline_weekly | D+3 | 11.4722 | 17.8236 | 0.4021 | 1.2824 |
| validation | E0_xgb | D+1 | 10.4449 | 14.3604 | 0.5296 | -0.8598 |
| validation | E0_xgb | D+2 | 10.4596 | 14.2193 | 0.5277 | -1.1697 |
| validation | E0_xgb | D+3 | 10.5224 | 14.2746 | 0.5233 | -1.4000 |
| validation | E1_recent_daily_blend | D+1 | 10.0613 | 14.1412 | 0.5438 | -0.7744 |
| validation | E1_recent_daily_blend | D+2 | 10.1109 | 14.0525 | 0.5387 | -1.0564 |
| validation | E1_recent_daily_blend | D+3 | 10.1506 | 13.9957 | 0.5417 | -1.2518 |
| validation | E1_selected_blend | D+1 | 10.1112 | 14.2892 | 0.5342 | -0.7128 |
| validation | E1_selected_blend | D+2 | 10.0662 | 14.0534 | 0.5387 | -0.9998 |
| validation | E1_selected_blend | D+3 | 10.1090 | 14.1148 | 0.5339 | -1.1777 |
| validation | E1_weekly_blend | D+1 | 10.1112 | 14.2892 | 0.5342 | -0.7128 |
| validation | E1_weekly_blend | D+2 | 10.0662 | 14.0534 | 0.5387 | -0.9998 |
| validation | E1_weekly_blend | D+3 | 10.1090 | 14.1148 | 0.5339 | -1.1777 |
| validation | baseline_recent_daily | D+1 | 16.3568 | 24.0130 | -0.3154 | -0.4327 |
| validation | baseline_recent_daily | D+2 | 14.5082 | 21.6719 | -0.0971 | -0.6033 |
| validation | baseline_recent_daily | D+3 | 12.1767 | 18.9402 | 0.1607 | -0.6592 |
| validation | baseline_weekly | D+1 | 12.3139 | 19.2103 | 0.1581 | -0.3700 |
| validation | baseline_weekly | D+2 | 12.1727 | 18.9229 | 0.1636 | -0.6035 |
| validation | baseline_weekly | D+3 | 12.1767 | 18.9402 | 0.1607 | -0.6592 |

## Price and rapid-change slices

See `outputs/metrics/event_slice_analysis.csv` for MAE/RMSE by split and experiment across the upper/lower actual-price deciles and the upper decile of absolute hour-to-hour actual-price changes.

| Final Test slice | Experiment | Rows | MAE | RMSE |
|---|---|---:|---:|---:|
| high_price_top_decile | E0_xgb | 1312 | 10.7234 | 12.8157 |
| low_price_bottom_decile | E0_xgb | 1329 | 31.2666 | 35.8673 |
| sharp_hourly_change_top_decile | E0_xgb | 1303 | 23.2405 | 30.1750 |
| high_price_top_decile | E2_train_plus_validation | 1312 | 8.3515 | 11.5541 |
| low_price_bottom_decile | E2_train_plus_validation | 1329 | 28.5673 | 32.4744 |
| sharp_hourly_change_top_decile | E2_train_plus_validation | 1303 | 19.7475 | 26.7531 |
| high_price_top_decile | E3_retrained_blend | 1312 | 8.8774 | 12.8771 |
| low_price_bottom_decile | E3_retrained_blend | 1329 | 23.3597 | 27.8413 |
| sharp_hourly_change_top_decile | E3_retrained_blend | 1303 | 18.2701 | 24.5767 |
| high_price_top_decile | baseline_weekly | 1312 | 12.9897 | 20.4111 |
| low_price_bottom_decile | baseline_weekly | 1329 | 12.6680 | 23.3299 |
| sharp_hourly_change_top_decile | baseline_weekly | 1303 | 18.9817 | 27.2763 |
