# V20 연구 shadow 모델 연결 안내

세 모델은 기존 운영 예측·발급 gate와 분리된 joblib 파일이다. V20 웹에 비교용 연구 결과를 표시할 때 `v20/ml_shadow_adapter.py`를 별도 shadow 경로에서 호출하면 된다.

```python
from v11.store import connect, latest
from v20.ml_shadow_adapter import predict

origin = "2026-10-27T23:00:00+09:00"
db = connect()
prices = latest(db, "land", as_of="2026-10-27T14:00:00+00:00")
db.close()

forecast = predict(prices, target_date="2026-10-29", horizon_days=2)
```

`latest(..., as_of=...)`에는 origin을 UTC로 변환한 값이 들어가야 한다. Adapter는 추가로 price timestamp와 `first_seen`이 해당 origin 이후인 행을 버리고, target_date 24시간을 만들지 못하거나 입력 lag가 부족하면 오류로 종료한다. Bundle별 학습 종료일, anchor weight, origin, 모델명을 결과에 붙인다.

현재 생성된 bundle:

| Bundle | 학습 종료 | 기본 모델 | 최근 calibration anchor weight |
|---|---|---|---:|
| `d1_recommended_shadow.joblib` | 2026-10-26 | peak-weighted HGB residual | 0.00 |
| `d2_recommended_shadow.joblib` | 2026-10-26 | peak-weighted HGB residual | 0.05 |
| `d3_recommended_shadow.joblib` | 2026-10-26 | HGB residual | 0.05 |

화면에서는 shadow forecast를 운영 예측과 별도로 그려야 하며, 연구 목적·vintage 미검증·학습 종료일을 함께 표시해야 한다. 이 번들/adapter는 `v20/service.py`, publisher, scheduler, 공식 발급 gate에 연결되지 않았다. 운영 전환은 독립적인 23시 input snapshot / 이후 확정 실적 validation을 마친 뒤 별도 변경으로 진행한다.

저장된 joblib은 trusted local artifact만 사용한다. 새 bundle을 만들려면 프로젝트 Python 환경의 scikit-learn 버전을 동일하게 유지하고 다음 명령을 실행한다.

```powershell
python -m experiments.smp_performance.train_ml_candidates
python -m experiments.smp_performance.render_ml_results
```
