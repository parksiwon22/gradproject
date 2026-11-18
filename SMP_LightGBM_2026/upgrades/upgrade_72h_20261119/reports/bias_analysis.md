# LightGBM bias analysis

Bias is mean(prediction - actual); positive values indicate overprediction. The full summary includes over/underprediction rates and prediction/actual variances.

Validation biases:

```text
                             experiment_id      split     n  bias_pred_minus_actual  overprediction_rate_pct  underprediction_rate_pct  actual_mean  prediction_mean  actual_variance  prediction_variance
                      E0_existing_lightgbm validation 12888                0.670910                53.545934                 46.454066   117.538864       118.209774         431.3037           305.455990
             E1_existing_plus_recent_daily validation 12888                0.547311                53.026071                 46.973929   117.538864       118.086175         431.3037           288.190959
                   E1_existing_plus_weekly validation 12888                0.427881                52.731223                 47.268777   117.538864       117.966745         431.3037           296.718846
               E1_existing_best_validation validation 12888                0.427881                52.731223                 47.268777   117.538864       117.966745         431.3037           296.718846
  E2_validation_selected_config_train_only validation 12888                0.440810                52.932961                 47.067039   117.538864       117.979674         431.3037           322.184725
E3_train_only_model_plus_selected_baseline validation 12888                0.243800                52.133768                 47.866232   117.538864       117.782664         431.3037           310.049971
                     baseline_recent_daily validation 12888               -0.565081                49.324953                 50.411235   117.538864       116.973783         431.3037           452.544208
                           baseline_weekly validation 12888               -0.544237                49.689634                 50.054314   117.538864       116.994627         431.3037           453.992043
```

Final Test biases:

```text
                     experiment_id      split     n  bias_pred_minus_actual  overprediction_rate_pct  underprediction_rate_pct  actual_mean  prediction_mean  actual_variance  prediction_variance
              E0_existing_lightgbm final_test 13104               18.764032                89.606227                 10.393773   104.237124       123.001156       532.027009           320.326675
     E1_existing_plus_recent_daily final_test 13104               16.998586                88.507326                 11.492674   104.237124       121.235710       532.027009           313.339610
           E1_existing_plus_weekly final_test 13104               15.264855                87.034493                 12.965507   104.237124       119.501978       532.027009           323.239929
       E1_existing_best_validation final_test 13104               15.264855                87.034493                 12.965507   104.237124       119.501978       532.027009           323.239929
E2_lightgbm_train_validation_refit final_test 13104               13.743652                82.165751                 17.834249   104.237124       117.980776       532.027009           235.152877
   E3_refit_plus_selected_baseline final_test 13104               11.248550                79.677961                 20.322039   104.237124       115.485674       532.027009           254.432198
             baseline_recent_daily final_test 13104                1.109573                51.884921                 47.802198   104.237124       105.346696       532.027009           524.440940
                   baseline_weekly final_test 13104                1.268143                52.770147                 46.634615   104.237124       105.505267       532.027009           524.106334
```

The large change between splits is treated as a distribution-shift warning. No Final Test-based offset was used.
