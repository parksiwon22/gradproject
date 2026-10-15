# V5 63피처의 V11 이식 점검

원 V5 소스와 모델은 수정하지 않았다. 생성식은 model/v5.py의 preprocess_hourly/build_one_day_rows를 기준으로 분류했다. 공식 자료의 event time과 실제 발표/수신시각은 별개다.

| V5 피처 | 의미/생성 | 2024~2026 이식 상태 |
|---|---|---|
| horizon | horizon, 과거 동일시간24/168h SMP 또는 origin lag/rolling/change | 공식 육지 SMP로 재구성 가능; V11은 독립21피처; 역사 발표 vintage 미확인 |
| same_hour_previous_day_smp | horizon, 과거 동일시간24/168h SMP 또는 origin lag/rolling/change | 공식 육지 SMP로 재구성 가능; V11은 독립21피처; 역사 발표 vintage 미확인 |
| same_hour_previous_week_smp | horizon, 과거 동일시간24/168h SMP 또는 origin lag/rolling/change | 공식 육지 SMP로 재구성 가능; V11은 독립21피처; 역사 발표 vintage 미확인 |
| profile_daily_weekly_spread | horizon, 과거 동일시간24/168h SMP 또는 origin lag/rolling/change | 공식 육지 SMP로 재구성 가능; V11은 독립21피처; 역사 발표 vintage 미확인 |
| cutoff_to_daily_profile_delta | horizon, 과거 동일시간24/168h SMP 또는 origin lag/rolling/change | 공식 육지 SMP로 재구성 가능; V11은 독립21피처; 역사 발표 vintage 미확인 |
| target_hour | origin/target 달력의 시간·요일·월 sin/cos와 주말 | 재구성 가능; 목표 달력 외 미래 관측은 사용 안 함 |
| target_hour_sin | origin/target 달력의 시간·요일·월 sin/cos와 주말 | 재구성 가능; 목표 달력 외 미래 관측은 사용 안 함 |
| target_hour_cos | origin/target 달력의 시간·요일·월 sin/cos와 주말 | 재구성 가능; 목표 달력 외 미래 관측은 사용 안 함 |
| target_dayofweek_sin | origin/target 달력의 시간·요일·월 sin/cos와 주말 | 재구성 가능; 목표 달력 외 미래 관측은 사용 안 함 |
| target_dayofweek_cos | origin/target 달력의 시간·요일·월 sin/cos와 주말 | 재구성 가능; 목표 달력 외 미래 관측은 사용 안 함 |
| target_month_sin | origin/target 달력의 시간·요일·월 sin/cos와 주말 | 재구성 가능; 목표 달력 외 미래 관측은 사용 안 함 |
| target_month_cos | origin/target 달력의 시간·요일·월 sin/cos와 주말 | 재구성 가능; 목표 달력 외 미래 관측은 사용 안 함 |
| target_is_weekend | origin/target 달력의 시간·요일·월 sin/cos와 주말 | 재구성 가능; 목표 달력 외 미래 관측은 사용 안 함 |
| target_is_public_holiday | 공휴일/대체/임시휴일 달력 플래그 | 정확한 당시 선공개 휴일 달력 미확보; V11 제외 |
| target_is_substitute_holiday | 공휴일/대체/임시휴일 달력 플래그 | 정확한 당시 선공개 휴일 달력 미확보; V11 제외 |
| target_is_temporary_holiday | 공휴일/대체/임시휴일 달력 플래그 | 정확한 당시 선공개 휴일 달력 미확보; V11 제외 |
| origin_smp | horizon, 과거 동일시간24/168h SMP 또는 origin lag/rolling/change | 공식 육지 SMP로 재구성 가능; V11은 독립21피처; 역사 발표 vintage 미확인 |
| origin_national_demand_mw | origin 실제 관측/파생값; demand lag1/24, renewable 합, net_load=수요-신재생 | 같은 지역·정의의 다년 관측 및 available_at 미확보; V11 제외. 전국2025 수요는 별도 보존 |
| origin_seoul_solar_generation_mwh | origin 실제 관측/파생값; demand lag1/24, renewable 합, net_load=수요-신재생 | 같은 지역·정의의 다년 관측 및 available_at 미확보; V11 제외. 전국2025 수요는 별도 보존 |
| origin_seoul_wind_generation_mwh | origin 실제 관측/파생값; demand lag1/24, renewable 합, net_load=수요-신재생 | 같은 지역·정의의 다년 관측 및 available_at 미확보; V11 제외. 전국2025 수요는 별도 보존 |
| origin_west_renewable_generation_sum | origin 실제 관측/파생값; demand lag1/24, renewable 합, net_load=수요-신재생 | 같은 지역·정의의 다년 관측 및 available_at 미확보; V11 제외. 전국2025 수요는 별도 보존 |
| origin_weather_temp_c | origin 실제 관측/파생값; demand lag1/24, renewable 합, net_load=수요-신재생 | 같은 지역·정의의 다년 관측 및 available_at 미확보; V11 제외. 전국2025 수요는 별도 보존 |
| origin_weather_rainfall_mm | origin 실제 관측/파생값; demand lag1/24, renewable 합, net_load=수요-신재생 | 같은 지역·정의의 다년 관측 및 available_at 미확보; V11 제외. 전국2025 수요는 별도 보존 |
| origin_weather_wind_speed_ms | origin 실제 관측/파생값; demand lag1/24, renewable 합, net_load=수요-신재생 | 같은 지역·정의의 다년 관측 및 available_at 미확보; V11 제외. 전국2025 수요는 별도 보존 |
| origin_weather_humidity_percent | origin 실제 관측/파생값; demand lag1/24, renewable 합, net_load=수요-신재생 | 같은 지역·정의의 다년 관측 및 available_at 미확보; V11 제외. 전국2025 수요는 별도 보존 |
| origin_weather_sunshine_hr | origin 실제 관측/파생값; demand lag1/24, renewable 합, net_load=수요-신재생 | 같은 지역·정의의 다년 관측 및 available_at 미확보; V11 제외. 전국2025 수요는 별도 보존 |
| origin_weather_solar_radiation_mj_m2 | origin 실제 관측/파생값; demand lag1/24, renewable 합, net_load=수요-신재생 | 같은 지역·정의의 다년 관측 및 available_at 미확보; V11 제외. 전국2025 수요는 별도 보존 |
| origin_reserve_power_mw | origin 실제 관측/파생값; demand lag1/24, renewable 합, net_load=수요-신재생 | 같은 지역·정의의 다년 관측 및 available_at 미확보; V11 제외. 전국2025 수요는 별도 보존 |
| origin_reserve_rate_percent | origin 실제 관측/파생값; demand lag1/24, renewable 합, net_load=수요-신재생 | 같은 지역·정의의 다년 관측 및 available_at 미확보; V11 제외. 전국2025 수요는 별도 보존 |
| origin_fuel_trade_LNG | 월별/일별 가스·연료 단가의 origin 값 | 원자료 정의/발표 vintage 미확보; V11 제외 |
| origin_gas_일반_원_GJ_공급비 | 월별/일별 가스·연료 단가의 origin 값 | 원자료 정의/발표 vintage 미확보; V11 제외 |
| origin_gas_일반_원_GJ_원료비 | 월별/일별 가스·연료 단가의 origin 값 | 원자료 정의/발표 vintage 미확보; V11 제외 |
| origin_gas_일반_원_Nm3_공급비 | 월별/일별 가스·연료 단가의 origin 값 | 원자료 정의/발표 vintage 미확보; V11 제외 |
| origin_gas_일반_원_Nm3_원료비 | 월별/일별 가스·연료 단가의 origin 값 | 원자료 정의/발표 vintage 미확보; V11 제외 |
| origin_gas_집단_원_GJ_공급비 | 월별/일별 가스·연료 단가의 origin 값 | 원자료 정의/발표 vintage 미확보; V11 제외 |
| origin_gas_집단_원_GJ_원료비 | 월별/일별 가스·연료 단가의 origin 값 | 원자료 정의/발표 vintage 미확보; V11 제외 |
| origin_gas_집단_원_Nm3_공급비 | 월별/일별 가스·연료 단가의 origin 값 | 원자료 정의/발표 vintage 미확보; V11 제외 |
| origin_gas_집단_원_Nm3_원료비 | 월별/일별 가스·연료 단가의 origin 값 | 원자료 정의/발표 vintage 미확보; V11 제외 |
| origin_is_public_holiday | 공휴일/대체/임시휴일 달력 플래그 | 정확한 당시 선공개 휴일 달력 미확보; V11 제외 |
| origin_is_substitute_holiday | 공휴일/대체/임시휴일 달력 플래그 | 정확한 당시 선공개 휴일 달력 미확보; V11 제외 |
| origin_is_temporary_holiday | 공휴일/대체/임시휴일 달력 플래그 | 정확한 당시 선공개 휴일 달력 미확보; V11 제외 |
| origin_hour_sin | origin/target 달력의 시간·요일·월 sin/cos와 주말 | 재구성 가능; 목표 달력 외 미래 관측은 사용 안 함 |
| origin_hour_cos | origin/target 달력의 시간·요일·월 sin/cos와 주말 | 재구성 가능; 목표 달력 외 미래 관측은 사용 안 함 |
| origin_dayofweek_sin | origin/target 달력의 시간·요일·월 sin/cos와 주말 | 재구성 가능; 목표 달력 외 미래 관측은 사용 안 함 |
| origin_dayofweek_cos | origin/target 달력의 시간·요일·월 sin/cos와 주말 | 재구성 가능; 목표 달력 외 미래 관측은 사용 안 함 |
| origin_month_sin | origin/target 달력의 시간·요일·월 sin/cos와 주말 | 재구성 가능; 목표 달력 외 미래 관측은 사용 안 함 |
| origin_month_cos | origin/target 달력의 시간·요일·월 sin/cos와 주말 | 재구성 가능; 목표 달력 외 미래 관측은 사용 안 함 |
| origin_is_weekend | origin/target 달력의 시간·요일·월 sin/cos와 주말 | 재구성 가능; 목표 달력 외 미래 관측은 사용 안 함 |
| origin_smp_lag_1h | horizon, 과거 동일시간24/168h SMP 또는 origin lag/rolling/change | 공식 육지 SMP로 재구성 가능; V11은 독립21피처; 역사 발표 vintage 미확인 |
| origin_smp_lag_2h | horizon, 과거 동일시간24/168h SMP 또는 origin lag/rolling/change | 공식 육지 SMP로 재구성 가능; V11은 독립21피처; 역사 발표 vintage 미확인 |
| origin_smp_lag_24h | horizon, 과거 동일시간24/168h SMP 또는 origin lag/rolling/change | 공식 육지 SMP로 재구성 가능; V11은 독립21피처; 역사 발표 vintage 미확인 |
| origin_smp_lag_168h | horizon, 과거 동일시간24/168h SMP 또는 origin lag/rolling/change | 공식 육지 SMP로 재구성 가능; V11은 독립21피처; 역사 발표 vintage 미확인 |
| origin_smp_rolling_24h_mean | horizon, 과거 동일시간24/168h SMP 또는 origin lag/rolling/change | 공식 육지 SMP로 재구성 가능; V11은 독립21피처; 역사 발표 vintage 미확인 |
| origin_smp_rolling_24h_std | horizon, 과거 동일시간24/168h SMP 또는 origin lag/rolling/change | 공식 육지 SMP로 재구성 가능; V11은 독립21피처; 역사 발표 vintage 미확인 |
| origin_smp_change_1h | horizon, 과거 동일시간24/168h SMP 또는 origin lag/rolling/change | 공식 육지 SMP로 재구성 가능; V11은 독립21피처; 역사 발표 vintage 미확인 |
| origin_smp_change_24h | horizon, 과거 동일시간24/168h SMP 또는 origin lag/rolling/change | 공식 육지 SMP로 재구성 가능; V11은 독립21피처; 역사 발표 vintage 미확인 |
| origin_demand_lag_1h | origin 실제 관측/파생값; demand lag1/24, renewable 합, net_load=수요-신재생 | 같은 지역·정의의 다년 관측 및 available_at 미확보; V11 제외. 전국2025 수요는 별도 보존 |
| origin_demand_lag_24h | origin 실제 관측/파생값; demand lag1/24, renewable 합, net_load=수요-신재생 | 같은 지역·정의의 다년 관측 및 available_at 미확보; V11 제외. 전국2025 수요는 별도 보존 |
| origin_renewable_generation | origin 실제 관측/파생값; demand lag1/24, renewable 합, net_load=수요-신재생 | 같은 지역·정의의 다년 관측 및 available_at 미확보; V11 제외. 전국2025 수요는 별도 보존 |
| origin_net_load | origin 실제 관측/파생값; demand lag1/24, renewable 합, net_load=수요-신재생 | 같은 지역·정의의 다년 관측 및 available_at 미확보; V11 제외. 전국2025 수요는 별도 보존 |
| origin_wind_data_available | origin 실제 관측/파생값; demand lag1/24, renewable 합, net_load=수요-신재생 | 같은 지역·정의의 다년 관측 및 available_at 미확보; V11 제외. 전국2025 수요는 별도 보존 |
| origin_solar_radiation_missing | origin 실제 관측/파생값; demand lag1/24, renewable 합, net_load=수요-신재생 | 같은 지역·정의의 다년 관측 및 available_at 미확보; V11 제외. 전국2025 수요는 별도 보존 |
| origin_solar_generation_missing | origin 실제 관측/파생값; demand lag1/24, renewable 합, net_load=수요-신재생 | 같은 지역·정의의 다년 관측 및 available_at 미확보; V11 제외. 전국2025 수요는 별도 보존 |

전국 수요 MWh는 1시간 평균 MW와 수치가 같아도 원 V5 national_demand_mw와 동일 원천/정의라고 인증하지 않는다. 현재 모든 연도의 원63피처 완비 날짜는 0일로 표시한다. 기상청 최신예보 키와 과거 forecast-vintage archive가 없어 실제기상을 미래예보로 대체하지 않는다.