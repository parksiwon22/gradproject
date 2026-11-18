# Validation/Test shift diagnostics

Target distribution uses unique target hours so overlapping 72-hour forecast blocks do not multiply the same SMP hour. Missing-input rates use the common dataset's pre-imputation missing flags; these are diagnostics only and do not change preprocessing.

| Split | Unique target hours | SMP mean | SMP SD | P10 | Median | P90 | P99 | Max | Measured-input missing-flag rate | SMP-history lag missing rate | SMP-history latest mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 17544 | 144.3263 | 44.0041 | 96.7400 | 140.0900 | 213.8810 | 271.7928 | 310.4500 | 0.0281 | 0.0004 | 148.9145 |
| validation | 4344 | 117.3826 | 20.9407 | 88.9600 | 118.2600 | 139.8870 | 157.3142 | 236.5500 | 0.0161 | 0.0000 | 118.1245 |
| final_test | 4416 | 104.2534 | 23.0565 | 71.6500 | 110.7800 | 130.4950 | 147.3935 | 193.7400 | 0.0171 | 0.0000 | 108.1383 |

Input-group missing-flag rates are in `outputs/metrics/input_missing_by_group.csv`; individual SMP history lag/rolling distributions are in `outputs/metrics/smp_history_feature_drift.csv`.
Validation-to-Final-Test unique-hour SMP mean changed by -13.13 KRW/kWh (-11.18%). The `smp_history_latest` mean changed from 118.12 to 108.14 KRW/kWh, while its missing rate remained 0.00%/0.00%.
The overall pre-imputation indicator rate changed from 1.61% to 1.71%; group-level changes are in the CSV. E0 Final Test monthly bias rose from 12.36 in July to 27.79 KRW/kWh in December.
Selected group missing-flag rates: group 1: Train 1.14%, Validation 2.12%, Final Test 2.31%; group 6: Train 7.82%, Validation 0.00%, Final Test 0.00%; group 7: Train 0.89%, Validation 0.95%, Final Test 1.87%; group 11: Train 9.75%, Validation 9.09%, Final Test 9.09%.
This pattern is consistent with a falling SMP level and growing positive E0 bias; missingness changed only modestly overall and the recent SMP lag remained populated. These diagnostics do not establish a single cause.
