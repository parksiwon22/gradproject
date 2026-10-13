# SMP Predictor V5 Day-Ahead

V5 is the first deployable day-ahead experiment for the EV charging web application. It forecasts all 24 SMP values for the next calendar day from the 23:00 cutoff and does not use recursive predicted SMP inputs.

## Forecast Definition

At 23:00 on day D, V5 forecasts day D+1 from 00:00 through 23:00. Each horizon is predicted directly from the same cutoff history. The baseline is same-hour previous-day SMP, and the residual target is `actual_smp(t+h) - smp(t+h-24)`.

## Deployability

The deployable model uses historical SMP, historical demand/weather/renewable/reserve/gas features, same-hour previous-day and previous-week profiles, and known target-day calendar features. It does not use realized target-day demand, weather, renewable generation, reserve, or SMP setter fields.

가격 날씨는 SMP 예측 기반 위험도이며 실제 기상예보와 다릅니다.

## Selected Model

- mode: `full`
- selected model by validation MAE: `validation_selected_profile_hgb_blend`
- final-test selected MAE: 11.0938
- final-test daily-profile MAE: 13.5758
- mean final-test charging regret across durations: 2.4643

## Output Schema

`generate_day_ahead_forecast(history_df, target_date, future_forecast_df=None)` returns exactly 24 rows with:

`timestamp, horizon, predicted_smp, baseline_smp, hgb_predicted_smp, ramp_probability, risk_level, price_weather, model_name, source`

## Runtime Summary

- elapsed seconds: 33.841
- daily train/validation/test counts: {'train': 235, 'validation': 30, 'test': 92}
- hourly row train/validation/test counts: {'train': 5640, 'validation': 720, 'test': 2208}
