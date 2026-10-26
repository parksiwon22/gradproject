# V20 shadow 모델 연결 안내

`final_candidate.json`은 rolling calibration 알고리즘 설정이고 `v20/research_model_adapter.py`는 기존 V20의 as-of 가격 snapshot을 받는 adapter입니다. 두 파일은 연구 브랜치에서만 사용할 수 있으며, V20 운영 발급 코드나 gate에 연결되어 있지 않습니다.

```python
from v20.research_model_adapter import load_config, predict

# prices는 v20.research._cutoff_prices(issue_time) 같은 immutable 23시 snapshot이어야 합니다.
# timestamp와 smp 열 외에 first_seen이 포함되면 adapter가 issue origin 이후 수신 행을 버립니다.
research_config = load_config()
forecast = predict(prices, target_date="2026-11-01", horizon_days=2,
                   config=research_config)
```

adapter는 D+1/D+2/D+3 각각에서 대상일 직전 90개의 완료 target day만 사용해, persistence와 같은 요일 2/4/8주 median 또는 4주 mean 앵커 사이의 가중치를 고릅니다. 검증 구간에서 MAE 및 G2V regret이 모두 horizon persistence보다 나쁜 후보는 선택하지 않습니다. 유효 후보가 없거나 이력의 24시간 벡터가 부족하면 persistence로 돌아갑니다.

Prediction dataframe은 timestamp, predicted_smp, horizon_days, origin_timestamp, 알고리즘·앵커·가중치·보정 일수와 수신 vintage metadata를 담습니다. 앱 화면과 정식 발급 경로에 연결하기 전, forward issue 기록에서 과거 수신 vintage와 actual SMP를 대조하는 shadow 기간을 먼저 확보해야 합니다.

## 재현 명령

```powershell
..\.venv\Scripts\python.exe -m experiments.smp_performance.run_study
..\.venv\Scripts\python.exe -m experiments.smp_performance.evaluate_v20_replay
..\.venv\Scripts\python.exe -m experiments.smp_performance.render_results
```

이 명령은 새 모델을 학습하지 않고 기존 보존자료로 timestamp 기반 재현을 만듭니다. 과거 `first_seen`과 과거 모델 잠금은 복구하지 않으므로 결과를 운영 성능으로 표시하지 않습니다.
