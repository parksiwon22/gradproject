"""Build aggregate-only report from executed local checkpoints, without refitting."""
import json
import importlib.metadata
import numpy as np
import pandas as pd
from scripts.research_local import ROOT, OUT

def markdown(frame, columns):
    def value(x): return f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x)
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+['| '+' | '.join(value(r[c]) for c in columns)+' |' for _,r in frame.iterrows()])

def run():
    a=pd.read_csv(OUT/'metrics.csv'); b=pd.read_csv(OUT/'price_only/metrics.csv')
    pred=pd.read_csv(OUT/'historical_test_predictions.csv')
    mask=pred.actual_smp>=pred.actual_smp.quantile(.95)
    for n in pred.columns[3:]:
        a.loc[(a.split=='historical_test')&(a.model==n),'descriptive_top5_MAE']=float(np.mean(abs(pred.loc[mask,n]-pred.loc[mask,'actual_smp'])))
    a.to_csv(OUT/'metrics.csv',index=False)
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    manifest['packages'].update({n:importlib.metadata.version(n) for n in ['lightgbm','catboost','xgboost']})
    manifest['diagnostic_amendment']='Initial training-q95=268.9125 had zero historical-test events; initial metrics preserved; fixed150 follow-up uses inherited V5 threshold; 2023 historical test exploratory.'
    manifest['derived_models']={'daily_profile':'previous same-hour','fixed_profile_069':'.69 daily + .31 weekly inherited V5 profile','hour_residual_correction':'July per-hour mean residual + HGB','ensemble':'.5 HGB + .5 weighted HGB','peak_gate':'HGB and q90 mixed by uncalibrated high-price classifier score'}
    manifest['quantile_coverage']={}
    for split in ('calibration','validation','historical_test'):
        p=pd.read_csv(OUT/(split+'_predictions.csv'))
        manifest['quantile_coverage'][split]=float(np.mean(p.actual_smp<=p.hgb_quantile90))
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
    winner=manifest['selection']['winner']
    pred.assign(error=pred[winner]-pred.actual_smp).groupby('target_date').agg(actual_max=('actual_smp','max'),predicted_max=(winner,'max'),Bias=('error','mean')).to_csv(OUT/'daily_maxima.csv')
    q=json.loads((OUT/'price_only/evaluation_manifest.json').read_text(encoding='utf-8'))
    columns=['model','MAE','RMSE','R2','peak_MAE','peak_Bias','peak_Precision','peak_Recall','peak_F1','false_alarms','EV_regret']
    report='''# V6–V10 로컬 재개발 실행 결과 — 2026-10-15

## 결론

V5 유지. 새 연구 코드는 독립적으로 작성했고 실제 로컬 데이터로 실행했다. 이전 클라우드 36+6 실험을 복구하거나 동일 재현한 결과가 아니다. V5 소스, 원본 모델, app.py, outputs/는 변경하지 않았다. main 및 V5 브랜치도 유지했다.

23시 기준 V5 전체 재학습을 별도 artifacts/research_local/v5_reproduction/에 실행했다. 2023 Q4 2,208개 예측은 보존 결과와 atol=1e-8로 일치했고 MAE 11.0937861480, R² 0.6429194849를 확인했다. 원자료 SHA-256과 피처 순서는 로컬 manifest에 기록했다.

## 실험 및 선택 계약

모든 실험 seed42, 전일23시 -> 다음날24시간. 원 V5는 1~8월 학습, 9월 검증이다. 새 실험은 2023년 1~6월 학습, 7월 잔차보정, 8~9월 모델 선택이다. 동일 날짜는 같은 split에 유지했다. HGB random early stopping은 껐다. 새 모델은 테스트 확인 후 재학습하거나 선정하지 않았다. V5와 새 모델의 학습기간이 달라 엄밀한 동일학습표본 알고리즘 우위 비교는 아니다.

선택 규칙: 검증 MAE가 고정 프로파일 MAE의 105% 이하인 후보 중 피크 MAE 최소, 동률이면 전체 MAE. 고정 프로파일 .69 일간 + .31 주간 비율은 기존 V5에서 상속된 기준이며 새롭게 사전 등록된 값은 아니다. 2023년 Q4는 과거 V5 연구에서 이미 본 기간이며 새 연구의 탐색 평가다.

초기 학습 상위5% 임계값 268.9125는 2023 Q4 피크 사례가 0건이라 평가에 부적합했다. initial_train_q95_*로 결과를 보존했고, V5의 기존 150원/kWh 기준으로 후속 실험을 수행했다. 아래 peak 지표는 실제값>=150, 알림은 예측값>=150이다. Bias는 예측-실제. 각 평가기간 실제 상위5% MAE는 사후 기술통계로 별도 기록했고 알림/선택 임계값 조정에는 사용하지 않았다.

## V6–V8: 2023 Q4 결과

'''+markdown(a[a.split=='historical_test'],columns)+'''

검증 선택은 `'''+winner+'''`. ExtraTrees는 탐색 Q4 전체 MAE와 피크 MAE에서 V5보다 좋았지만 검증 선택 모델이 아니므로 테스트 성능만 보고 최종 승격하지 않았다. 가중 손실·고가 분류기 결합·잔차 보정·앙상블·q90을 실제 실행했다. 고가 classifier score는 확률 보정되지 않았으며 운영 급등 확률로 인증하지 않는다. q90은 잔차 조건부 분위수 모델이며 검증/테스트 coverage는 manifest에 기록한다. 예측가격>=150 지표와 +15원/h 상승 ramp는 다른 사건이다. 상승/하락 ramp MAE, 일 최대오차, 시간별 MAE/Bias도 로컬 metrics와 진단 CSV에 있다.

## V9: EV 평가

가상 SMP 연동 요금, 1kW 정출력, 매일24시간 연결 가능, 연속2/3/4/5/6/8시간 충전이다. 기간별 평균 실제 충전비, 실제가격 oracle 최저비, regret(원), 최저구간 overlap 및 놓친 피크에서 충전한 비율을 기록했다. duration은 동일 가중. 실제 개인 소매요금/V2G 판매수익/실차 SOC 제어 결과가 아니다. 이 단순 시나리오는 밤새 연결/차량제약/공식가격 수신 후 재최적화를 평가하지 않는다.

## V10: 2024 공식 자료 독립 기간 평가

프로젝트 폴더에서 2024 자료를 찾지 못해 KPX EPSIS 공식 육지 시간별 SMP를 조회했다. 2023-12 이력 포함397일, 2024년 윤년 8,784시간과 Q4 2,208시간의 완전성·중복·시간연속을 검증했다. 원응답은 로컬에만 저장하며 hash는 '''+q['official_payload_sha256']+'''.

[공식 조회 페이지](https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpShdChart.do?menuId=040202)의 POST 형식을 읽어 조회했다. c1=01시 끝 구간은 시작00시, c24=24시 끝 구간은 시작23시로 정렬했다. 다운로드한 역사 정정 최종 스냅샷이며 당시 공개시각 증거는 아니다. 원2023 CSV의 시장구분은 미검증이므로 2024 육지와 동일 시장이라고 인증하지 않는다.

V5 원피처에 필요한 2024 기상·수요·연료·예비력 자료가 없어 **원 V5 및 63피처 새 모델의 2024 결과는 미완료**다. 미래 관측을 채워 넣지 않고 SMP 이력/달력 전용 별도 모델6개와 프로파일을 학습했다. 이들은 V5 대체 재현 모델이 아니다. 동일한 2023 시간순 학습/선택 기준을 사용하고 `selection_before_2024.json`을 쓴 후 2024 정량 평가를 수행했다. 다운로드 파싱 중 원응답 일부는 점검했다. Q4는 과거 클라우드 결과를 본 기간이므로 완전 미열람 테스트가 아니다. 평가 결과를 이용한 재선정은 하지 않았다.

2024 평가에 앞서 선택된 가격전용 모델: `'''+q['winner']+'''`.

### 2024 전체

'''+markdown(b[b.split=='2024_full'],columns+['descriptive_top5_MAE'])+'''

### 2024 Q4

'''+markdown(b[b.split=='2024_Q4'],columns+['descriptive_top5_MAE'])+'''

고정 프로파일은 전체 MAE에 유리하지만 일부 학습모델은 고가 MAE/regret를 낮추는 대신 허위 경보와 전체오차를 늘렸다. 클라우드 보고 수치(전체 앙상블12.2114/프로파일10.6683, 상위5%15.9059/21.4839)와 이번 결과는 별개이며 이전 실험의 재현으로 주장하지 않는다.

## 재현 명령 (저장소 루트 PowerShell)

```powershell
$py = '..\\.venv\\Scripts\\python.exe'
& $py -m pip install -r requirements-research.txt
$env:SMP_RESULTS_DIR='artifacts/research_local/v5_reproduction/results'
$env:SMP_MODELS_DIR='artifacts/research_local/v5_reproduction/models'
& $py -m model.v5 --full
Remove-Item Env:SMP_RESULTS_DIR,Env:SMP_MODELS_DIR
& $py -m scripts.research_local
& $py -m scripts.download_kpx_2024
& $py -m scripts.research_2024
& $py -m scripts.research_report
& $py -m unittest tests.test_research_local tests.test_v5.UnitTests tests.test_v5.IntegrationTests.test_full_reproduction tests.test_v5.IntegrationTests.test_saved_model_inference_and_cutoff -v
& $py -m pip check
```

권한 있는 원2023 CSV와 보존 V5 결과가 필요하다. 재실행은 research_local 내부 결과를 덮어쓰므로 이전 체크포인트는 먼저 별도 백업한다. 다운로더는 기존 공식 스냅샷을 보존한다. 연구선정은 코드 고정 후 평가하며 보고서 재생성은 재학습하지 않는다.

## 로컬 산출물과 미완료

- `artifacts/research_local/manifest.json`: 입력hash, 환경, 63피처, 기간, seed, 모델별 파라미터/성공여부, 선정규칙, q90 coverage.
- `metrics.csv`, `historical_test_predictions.csv`, `hourly_errors.csv`, `daily_maxima.csv`: 집계와 상세 비교.
- `quarter.png`, `first_week.png`: 2023 실제/V5/검증선정 모델.
- `price_only/metrics.csv`, `2024_predictions.csv`, `evaluation_manifest.json`, `2024_Q4.png`, `2024_first_week.png`: 공식가격 전용 평가.
- `tests_final.log`: 최종 검증. 초기 파서 빈응답 KeyError를 ValueError로 수정한 이력은 tests.log에 보존.
- 검증 범위: 미래값 변경 시 cutoff 피처 불변, 공식 파서 자정/24시·빈응답·중복 거부, oracle regret0, V5 전체지표 및 저장모델 추론. 기존 웹은 변경하지 않았고 웹 산출물 덮어쓰는 테스트는 실행하지 않았다.
- 미완료: 원 V5/63피처 2024 평가, 원자료 시장/피처 실제 공개시각·vintage 검증, 다년 rolling 검증, 실사용 EV 제약·재최적화, 확률 calibration. 원 V5 교체 및 배포는 하지 않는다.

raw 데이터, 개별 예측, 모델 바이너리, 로컬 경로 자산목록은 Git 제외한다. 집계 보고서와 소스/테스트만 공개 개발 브랜치에 저장한다. 개인 parksiwon22 GCM 계정 확인 후 해당 username을 지정해 Push했다. swpark3622 인증은 사용하지 않았다.
'''
    (ROOT/'docs/LOCAL_V6_V10_RESULTS.md').write_text(report,encoding='utf-8')
    print('Aggregate report written; no refitting')

if __name__=='__main__': run()
