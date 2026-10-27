# SMP 예측 연구 최종 후보 보고

작성일: 2026-10-27. 이 보고의 새 모델 평가는 **과거 fold 시점 재현**이며, 실시간 운영 예측 성능이 아니다.

## 최종 연구 후보

| 기간 | 후보 | fold 평가일 | MAE | RMSE | G2V regret | 기준선 대비 MAE / RMSE / regret 감소 |
|---|---|---:|---:|---:|---:|---:|
| D+1 | V11 구조 ExtraTrees, fold별 재학습 | 275 | 9.248 | 13.200 | 69.699 | 23.9% / 30.3% / 56.7% |
| D+2 | HGB 잔차 + seasonal anchor 혼합 | 275 | 10.175 | 14.223 | 65.408 | 36.5% / 38.1% / 67.4% |
| D+3 | HGB 잔차 + seasonal anchor 혼합 | 275 | 10.429 | 14.621 | 63.175 | 35.7% / 36.6% / 66.4% |

기준선은 같은 23시 입력으로 계산한 persistence이다. MAE/RMSE는 24시간 오차를 합친 pooled 값, G2V regret은 하루별 regret의 평균(원)이다. 충전 평가는 60kWh, SOC 30%→80%, 충전 7kW, 효율 90%, 하루 전체 연결 조건을 동일하게 적용했다. 웹 시연에서는 차량 조건을 조정해 다시 계산할 수 있다.

## 학습과 검증 설계

각 기간은 순서대로 학습, 보정, 평가를 분리했다. 평가일은 서로 겹치지 않는다.

| Fold | 학습 종료 | anchor 혼합 보정 기간 | 검증 기간 |
|---|---|---|---|
| 2024 가을 | 2024-06-30 | 2024-07-01–09-30 | 2024-10-01–12-31 |
| 2025 봄 | 2024-12-31 | 2025-01-01–03-31 | 2025-04-01–06-30 |
| 2025 가을 | 2025-06-30 | 2025-07-01–09-30 | 2025-10-01–12-31 |

HGB와 LightGBM은 일별 SMP를 예측하는 대신 공통 V11/V17 입력에서 daily 기준 가격에 대한 잔차를 학습한다. seasonal anchor는 대상일과 같은 요일·시간의 과거 7/14/21/28일 SMP 중앙값이다. 혼합 비중은 직전 보정 구간에서만 고른다. LightGBM은 V12 설정의 300 estimator, learning rate 0.05, 31 leaves, seed 42를 사용했다. 급등 가중 실험은 실제 SMP가 150 이상인 학습 표본에 6배 가중치를 주었다. D+1 ExtraTrees는 V11 설정(150 trees, min leaf 10, max features 0.8)으로 fold마다 다시 적합했다.

## 후보 선택 및 실험 결과

- **D+1:** ExtraTrees가 새 HGB 및 단순 기준선보다 MAE/RMSE가 낮았다. HGB 혼합은 G2V regret이 조금 더 낮았지만 MAE 우선 원칙에 따라 ExtraTrees를 후보로 선택했다.
- **D+2:** HGB/anchor 혼합이 LightGBM/anchor 혼합보다 전체 세 fold에서 MAE 0.290, RMSE 0.412, regret 약 3.66원 낮았다. LightGBM 단독과 6배 급등 가중은 혼합 후보를 개선하지 못했다. 최신 2025Q4 fold에서도 HGB/anchor 혼합이 LGB 혼합보다 MAE가 낮았다.
- **D+3:** 세 fold pooled 결과에서 LGB/anchor 혼합은 HGB/anchor보다 MAE 약 0.033, RMSE 0.056, regret 약 0.33원 앞섰다. 차이는 작고 2025Q4에서는 HGB/anchor의 MAE가 근소하게 더 낮으며 RMSE도 더 낮았다. 작은 평균 우위를 일반화 개선으로 볼 근거가 부족하여 HGB/anchor를 최종 후보로 선택했다. LightGBM 비교 결과는 모두 보존했다.
- 급등 표본 6배 가중은 일반 MAE 및 G2V regret을 대체로 악화시켰다. 이를 최종 후보에 사용하지 않았다.

기존 성능 중 2025Q4 자료는 이미 여러 차례 선택/분석에 쓰인 기간이다. 보존된 V12/V14/V15 Q4 시간별 점수와 새 fold 점수는 `common_2025q4_model_summary.csv`와 이전 `docs/smp_improvement_ml/d2_2025q4_common_date_comparison.csv`, `d3_v17_v19_common_date_comparison.csv`에 따로 기록했다. 기존 V17/V19의 보존 점수는 일별 RMSE 평균, 새 후보는 pooled 시간별 RMSE인 산출물이 있어 숫자를 직접 우열 주장에 섞지 않았다. 공정한 요약표는 동일 scoring 구현으로 재계산된 새 모델 275일과 persistence를 사용했다. 기존 V12/V14/V15의 별도 Q4 점수는 참고용으로만 읽어야 한다.

2025Q4(92일)에서 모델 계보를 함께 볼 때 저장 결과는 다음과 같다. 이 기간은 이미 모델 선택·분석에 사용되어 독립 테스트가 아니다. D+2는 시간별 저장 예측을 같은 기간으로 다시 점수화했다. D+3 V17/V19 행은 보존한 일별 측정값으로, MAE와 regret은 같은 전일·EV 조건이며 RMSE는 보존 결과의 **일별 RMSE 평균**이다. 새 후보의 RMSE는 pooled 값이므로 D+3 RMSE는 직접 비교하지 않는다.

| 기간 | 모델 | MAE | RMSE | G2V regret (원/일) | 자료 구분 |
|---|---|---:|---:|---:|---|
| D+2 | V12 LightGBM peak | 13.297 | 17.837 | 84.315 | 보존 시간별 검증 예측 |
| D+2 | V14 shape-level blend | 13.415 | 17.564 | 70.788 | 보존 시간별 검증 예측 |
| D+2 | V15 event-features LightGBM | 12.827 | 17.210 | 112.688 | 보존 시간별 검증 예측 |
| D+2 | 새 HGB/anchor | 12.143 | 15.223 | 79.473 | fold 재현 |
| D+2 | 새 LightGBM/anchor | 12.728 | 17.256 | 86.103 | fold 재현 |
| D+3 | V17 joint | 15.328 | 18.322* | 107.355 | 보존 일별 측정값 |
| D+3 | V19 independent HGB | 15.620 | 18.316* | 94.958 | 보존 일별 측정값 |
| D+3 | 새 HGB/anchor | 12.995 | 15.880 | 80.278 | fold 재현 |
| D+3 | 새 LightGBM/anchor | 13.008 | 17.832 | 80.437 | fold 재현 |

* 기존 V17/V19 RMSE는 일별 RMSE 평균이라 fold 모델 pooled RMSE와 산식이 다르다. V17/V19 대비 MAE/regret 감소는 이 반복 선택 기간의 사후 진단이지 독립적인 개선 입증은 아니다.

## 시점 제한과 누출 검증

모든 새 fold feature는 명목상 발급 시각인 KST 23:00 이하의 SMP timestamp로만 생성했다. D+1/D+2/D+3 builder는 origin 및 필요한 lag SMP가 빠지면 대상일 feature를 완성하지 않고, `v20/ml_shadow_adapter.py`는 입력 24시간이 완성되지 않거나 finite하지 않으면 오류로 중단한다. 23:00 가격 누락 시 미래 관측으로 대체하지 않는다. D+1~D+3의 누락 origin 입력을 각각 제거하는 회귀 테스트를 추가했고 통과했다.

SQLite의 과거 `first_seen`은 현재 수집 시각을 뜻하며 과거 공개/수신 시각의 원본 증거가 아니다. 그래서 feature timestamp는 23시 이전으로 제한했지만, 그 데이터가 당시 실제로 수신 가능했는지(vintage)는 검증되지 않았다. 새 fold와 과거 저장 예측을 운영 기록/사전예측으로 취급하지 않는다. 2026-10-26까지 학습한 shadow bundle은 과거 시연에서 호출하지 않는다. 각 과거 검증일보다 앞선 fold 학습 종료일을 가진 fold 예측 결과만 시연한다. 다만 fold joblib이 당시 실제로 존재했는지는 입증되지 않았다.

## 웹 시연 및 보존

`app_v20.py`의 **최종 연구 fold 시연** 메뉴는 275일의 저장된 검증 fold 예측으로 D+1~D+3, 기준선, 실제 SMP를 그린다. 각 날짜의 MAE/RMSE, 시간별 오차와 같은 EV 조건의 G2V 충전 시간 및 fold 실제 SMP 기준 비용/regret 재생을 표시한다. 화면은 retrospective fold replay와 미검증 vintage를 표시한다. 기존 공식 발급 gate, scheduler, 운영 모델 호출 경로는 수정하지 않았다.

- 대표 예측 그래프: `final_representative_forecast.png` (2025-12-31, 세 horizon)
- 선정 성능표: `final_recommended_models.csv`
- 시간대별 MAE: `final_hourly_mae.csv`
- Streamlit 시연 데이터: `final_fold_demo.csv.gz`
- LightGBM/격리 환경 잠금: `requirements-lock.txt`

`requirements-lock.txt`는 `.venv-smp-research`에서 `pip freeze`로 기록했다. LightGBM 4.7.0, scikit-learn 1.7.2, numpy 2.2.6, pandas 2.3.3, Streamlit 1.66.1, Altair 6.3.0 및 matplotlib 3.11.2를 포함한다. 저장된 기존 sklearn 모델 일부는 1.9.0에서 pickle되어 로딩 때 버전 경고가 발생했다. 비교는 지정된 격리 버전에서 재생했으며 이 호환성 경고를 보존했다.

## 한계 및 다음 검증

275일은 세 개의 과거 validation fold이며 완전히 독립적인 unseen test가 아니다. fold 날짜 자체는 disjoint지만 이 프로젝트의 반복된 연구·모델선정 맥락 때문에 선택 편향이 남는다. SMP 수신 vintage와 과거 joblib 가용성도 미확인이다. 따라서 현재 후보는 졸업작품의 **연구용 시연 모델**로 확정하고 운영 자동 발급 모델로 승격하지 않는다. 실제 운영 적합성 판단에는 이후 시점에 저장한 입력 snapshot, 모델 hash/학습 종료, 발급 timestamp, 확정 대상 SMP를 함께 보존하는 전향적 비교가 필요하다.

재현 명령 (PowerShell, 저장소 루트):

```powershell
.venv-smp-research\Scripts\python.exe -m streamlit run app_v20.py
```
