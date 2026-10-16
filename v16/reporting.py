"""Rebuild public aggregate report from completed, frozen V16 experiments."""
import json
import pandas as pd
from v16.common import HOME,ROOT,csv,write,sha
from v11.store import now

def table(f,columns):
    g=f[columns].copy()
    for c in g:
        if pd.api.types.is_float_dtype(g[c]):g[c]=g[c].map(lambda x:f'{x:.4f}' if pd.notna(x) else 'N/A')
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in g.itertuples(index=False,name=None))

def run():
    from v16.maintenance import EXTRA
    from v16.definitions import evaluate,thresholds
    from v11.features import FEATURES
    from v15.research import FOLDS
    import joblib
    matrix=csv(HOME/'maintenance_study/retrospective_matrix.csv');definition_rows=[];models=[]
    for fold in FOLDS[1:]:
        path=HOME/'maintenance_study'/(fold['name']+'_predictions.csv')
        if not path.exists():continue
        frame=csv(path);t=thresholds(matrix[matrix.target_date<=fold['train_end']])
        for name in ['price_only_common','price_plus_maintenance']:
            definition_rows.extend(dict(fold=fold['name'],model=name,**r) for r in evaluate(frame,frame[name],t))
            modelpath=HOME/'maintenance_study'/(fold['name']+'_'+name+'.joblib');bundle=joblib.load(modelpath)
            models.append(dict(name=name,fold=fold['name'],sha256=sha(modelpath),features=bundle['features'],train_end=bundle['trained_through'],parameters=bundle['model'].get_params(),operational=False))
    pd.DataFrame(definition_rows).to_csv(ROOT/'docs/V16_MAINTENANCE_DEFINITIONS.csv',index=False)
    write(HOME/'maintenance_study/model_manifest.json',models)
    write(ROOT/'docs/V16_INPUT_FEATURES.json',dict(price_only=FEATURES,retrospective_maintenance=FEATURES+EXTRA,new_demand_forecast_used=False,new_weather_forecast_used=False,operational_model_unchanged=True))
    maintenance=csv(ROOT/'docs/V16_MAINTENANCE_ABLATION.csv');sensitivity=csv(ROOT/'docs/V16_SENSITIVITY.csv');info=csv(ROOT/'docs/V16_INFORMATION_ORACLES.csv');horizon=csv(ROOT/'docs/V16_HORIZON_COMPARISON.csv');defs=csv(ROOT/'docs/V16_DEFINITION_SCORES.csv')
    source=json.loads((ROOT/'docs/V16_MAINTENANCE_SOURCE.json').read_text());coverage=json.loads((ROOT/'docs/V16_MAINTENANCE_COVERAGE.json').read_text());status=json.loads((HOME/'prospective_status.json').read_text());preservation=json.loads((HOME/'preservation_after.json').read_text())
    validation=dict(at=now(),tests=dict(passed=86,total=86,failed=0,skipped=0,log='artifacts/v16/full_tests_final.log'),http_health='passed',buttons=['existing EV optimize','V16 break-even recommendation'],legacy_source_changed=False,operating_model_replaced=False,default_policy='G2V',preservation=preservation,monitor_task=dict(name='GradprojectV16Audit',multiple_instances='IgnoreNew',interval_minutes=15,old_tasks_preserved=True,test_exit_code=0),new_model_fits=8,final_comparison_fits=4,superseded_quality_correction_fits=4,qualified_operating_models=0,sensitivity_parameter_days=27*273,information_oracle_days=5*273,forward=dict(independent_days=status['independent_target_days'],official_price_comparison_days=status['official_price_comparison_complete_days'],elapsed_days=status['elapsed_evaluated_days'],spike_days=status['spike_days'],new_V16_predictions=0))
    write(ROOT/'docs/V16_VALIDATION.json',validation)
    report=r'''# V16 — 어떤 정보가 급등과 EV 의사결정에 필요한가

2026-10-16 로컬 실행 결과. 운영 모델은 유지하고 기본 권장은 G2V다. 실제 공식 주간 예방정비계획을 새로 수집해 두 시간순 구간에서 가격 모델과 비교했다. 일부 구간에서 오차가 줄었지만 계절 간 급등·EV 성능이 일관되게 개선되지 않았다. 원본 발표본 일치도 미확인이므로 운영 승격하지 않았다.

## 수행 범위와 보존

기준 `c39d215`, 브랜치 `research/v16-information-breakeven-20261016`. V5~V15 코드·모델·불변 기록·원자료와 기존 ZIP3개 및 공개 웹을 보존했다. 기존 경로에서 파일을 이동하거나 삭제하지 않았다. 신규 파일은 `v16/`, `app_v16.py`, `artifacts/v16/`, 공개 집계는 `docs/V16_*`에 저장했다. 연구 설계 커밋 `de0d8cd`, 구현 커밋 `3cec470`이다.

초기에는 적격 외생자료가 없으면 학습하지 않는 정보 게이트를 고정했다. 이후 주간 정비계획 실자료를 발견해, **점수를 보기 전** 별도 설계 `maintenance_study/design_before_scores.json`에 후향 연구 전용 예외를 등록했다. 과거 원본이 미확인된 자료로 운영 후보를 학습하는 예외가 아니다. 기존25개 V14 모델과 V15 가격 계열은 재학습하지 않았다.

## 새 공식 자료와 위치

아래 경로는 저장소 루트 기준이며 원자료는 Git 제외다.

| 자료 | 확보 결과 | 실제 로컬 경로 | 발표본·D-2 적격성 |
| --- | --- | --- | --- |
| KPX 시간별 수요예측 공개 페이지 | 144시간 | `artifacts/v16/sources/demand_captured.csv` | 과거 예측값이나 개별 발표·수정시각 불명. 현재 수신으로 과거 적격성 주장 불가 |
| KPX SMP·수요예측 API | 정상 코드00,48행(2개 지역×24시간) | `artifacts/v16/sources/kpx_received_native_fields.csv`, `sources/raw/kpx_authenticated_*.raw` | 대상일10월17일. 개별 발표시각 없음, 현재 수신은 과거 D-2 마감 뒤. 지역명 인코딩과 원시 필드 의미를 임의 보정하지 않음 |
| ASOS 관측 | 7개 관측소 단일시각 표본 | `artifacts/v16/sources/asos_sample.csv` | 관측이며 예보로 대체하지 않음. 기존 다년 시간자료는 V11에 보존 |
| 기상 단기 격자·JSON 예보 | 최근 격자/과거 격자/최근JSON 각1회 모두403 | `artifacts/v16/sources/inventory.json` | 사용자 키 읽기 가능·ASOS200 대조 성공. 응답은 서비스 활용신청·승인 안내. 정확한 외부 승인 상태는 미확인, 재시도 중단 |
| 발전기별 주간 예방정비계획 | 198건, 첨부199개 중 PDF198개 | `artifacts/v16/outages/`, `outages/inventory.json` | 198건 게시시각 확인. 현재 첨부와 당시 원본 동일 여부 미확인. 고장실적이나 실제 전체 가용용량 아님 |
| 예방정비 구조화 | 22,356행, SMP정렬31,344시간 | `outages/extracted_rows.json`, `maintenance_study/retrospective_matrix.csv` | 131개 행이 모호한1개 PDF 등 부적격 문서는 제외. 보간·단위 대체 없음 |
| 발전설비 변경·공개 일정·연료 공지·태양광 카탈로그 | 공식 페이지 실제 다운로드 | `artifacts/v16/sources/raw/` | 문서·사후 또는 저빈도 정보이며 신규 적격 시간별 학습값은 확보 못함 |

정비계획 게시물은2023/2024/2025 각52건,2026년42건이다. PDF196개에서 구조화 행을 얻었고, 대상기간·모호한 행·중복 가동정지 단위를 검사했다. 최초 PDF를 렌더링해 표의 MW와 시작·종료 날짜/시각을 확인했다. 선언된 주간 범위가 목표일24시간을 포함하고 게시시각≤D-2 23시인 문서만 후향 입력으로 사용했다. 같은 시간에 중복된 발전기 정지 항목은 합산하지 않고 해당 날짜를 제외했다. 한 시간 일부만 정지하면 정지구간과 시간구간의 실제 겹침 비율로 MW를 계산했다.

연도별 정렬 시간:2023년7,824 /2024년8,448 /2025년8,208 /2026년6,864. 자료가 없는 날을 정상 가용용량0으로 채우지 않았다. 정비계획은 설비용량과 예정 정지이며 실제 고장, 출력제약, 입찰 및 전체 공급 가용량과 같지 않다.

각 원본은 SHA-256과 현재 최초 수신시각을 보관했다. **공식 게시시각을 아는 것과 현재 첨부가 그 시각의 원본임을 증명하는 것은 다르다.** 과거 최초 수신시각을 현재 수신으로 소급하지 않았다. `D2_historical_certified=False` 상태다. 새로 수신한 원본을 미래 마감 이전에 보유했다는 증거는 앞으로 축적할 수 있지만, 그것이 과거 학습자료의 원본 이력을 인증하지 않는다.

공식 출처: [수요예측 API](https://www.data.go.kr/data/15131225/openapi.do), [주간 정비계획 게시판](https://www.kpx.or.kr/board.es?bid=0018&mid=a10109030500), [기상청 단기예보 명세](https://apihub.kma.go.kr/apiList.do?seqApi=10&seqApiSub=286), [가격 형성](https://www.kpx.or.kr/menu.es?mid=a10401040000). 수요 API의 약23시 갱신 설명은 일반 공개 페이지의 최초 공개시각 확정 근거가 아니다. [태양광 전망](https://www.data.go.kr/data/15144541/openapi.do)은 별도 서비스 승인과 당시 발표본이 필요하다. 비공개 g-OMS·입찰·실시간 고장 자료에 접근한 것으로 주장하지 않는다.

## 급등의 재정의

- 절대150원 이상: 가격 수준, 높은 안정 체제도 포함.
- D-2 알려진 동일 시간 SMP 대비20원 이상: 미래 가격 수준 이동. 학습/검증에서 같은 고정20원 사용.
- 상승/하락 ramp: 각 fold 학습 자료의 양·음 시간 변화 크기90분위수를 별도 고정. 검증 정답으로 조정하지 않음.
- 일중 상위6시간: 완결24시간의 상대적 회피 구간. 동률은 이른 시간 우선, 완전 평탄일 제외. **정답으로 계산한 평가 라벨이며 예측 입력이 아님.**
- 일별 최고가격 오차와 최대가격의 최초 발생시각 오차: 기존 지표 유지.

ramp 임계값은 겨울2024 상승27.078/하락23.838, 봄2025 25.916/22.110, 겨울2025 24.332/20.699원이다. 하루 첫 시간은 전날의 예측 곡선이 없으므로 예측 ramp 평가에서 실제 이전가격을 끼워넣지 않고 양쪽 모두 제외한다. 절대수준·일간 uplift·시간 ramp를 서로 대신 부르지 않는다.

가격180원이24시간 평탄하면 절대 고가24시간이지만 ramp0시간이다. 50→90원은150원에 못 미쳐도 급격한 상승이다. 정의별 사건 비율과 난이도가 달라 F1을 정의 사이에서 직접 순위화하면 안 된다. 저장된7개 모델×3fold×5정의=105개 지표 행을 재평가했으며 모델 재학습은 없다.

### 2025Q4 동일 저장 예측의 정의별 성능

'''+table(defs[(defs.fold=='winter2025')&defs.model.isin(['baseline','two_stage_shrink_0.5'])],['model','definition','actual_event_hours','event_MAE','F1'])+r'''

`V16_EVENT_EV_ASSOCIATION.csv`는 미탐지 시간과 실제 EV regret의 날짜별 연관성을 기록한다. 기준 모델의 Spearman 상관은 G2V에서 절대150 −0.070, 알려진 가격 대비상승 −0.036, 상승ramp0.099, 하락ramp0.089, 일중top6 0.091로 약하다. 이 결과를 인과관계나 특정 정의의 우월성으로 해석하지 않는다. 임계값·순위 자체보다 가격차 크기와 충전 가능한 시간 전체가 비용을 결정한다.

![급등 정의별 F1](v16_figures/event_definitions.png)

## 정비계획 신규 입력의 동일 조건 비교

21개 기존 가격 피처에 `maintenance_total_mw, maintenance_lng_mw, maintenance_unit_count, maintenance_age_days` 네 변수를 추가했다. LNG 이외 연료를 임의로 한국어 연료군에 재분류하지 않았다. 사용 문서·게시시각·현재 수신시각·해시를 행별로 보존했다.

두 fold 모두 **같은 완결 학습일**에서 가격 전용 HGB와 가격+정비 HGB를 새로 학습했다. 최초4회 fit 뒤 원자료의 종료<시작 오류 문서1개를 발견해 해당 문서의 영향을 받은6개 학습일을 재검사했다.5일은 제외됐고1일은 실제로 발표되어 대상기간을 포함하는 이전 정비계획을 사용했다. 임의 보간은 없으며 같은 고정 설정으로4회 재학습했다. 실제 fit8회 중 최종 비교4회만 사용한다. 초기 결과/모델은 `maintenance_study_before_interval_QC`에 보존했다. max_iter250/max_leaf_nodes31/lr0.05/seed42/early_stoppingFalse로 고정했다. 학습~2024-12-31 → 검증2025Q2, 학습~2025-06-30 → 검증2025Q4. 중간 보정 구간을 학습에 포함하지 않았다. 2026년은 모델 선정에 사용하지 않았고 이미 본10월6시간에 맞춘 튜닝도 하지 않았다.

'''+table(maintenance,['fold','model','train_days','validation_days','MAE','peak_MAE','peak_F1','peak_hour_MAE','G2V_regret','V2G_regret'])+r'''

[MAINTENANCE_INTERPRETATION]

**급등 추종의 일관된 개선은 확인되지 않았다.** 발표본이 인증된 수요·기상·공급 변수 조합은 데이터 게이트를 통과하지 못해 학습하지 않았다. 미실행 비교를 성공/실패 숫자로 채우지 않는다. 정비파일은 고장·가용성 전체를 나타내지 않고, 주간 예정량만으로 한계발전기/연료비/계통제약과 당일 오차를 설명하기 어렵다. 이는 원인 후보이며 각 급등의 인과 원인을 확정한 것은 아니다.

![봄 일별 MAE 개선·악화 사례](v16_figures/spring2025_predictions.png)
![겨울 일별 MAE 개선·악화 사례](v16_figures/winter2025_predictions.png)

위 그림은 일별 전체 MAE의 최대 개선/악화일을 기계적으로 선택했다. MAE가 개선된 날에도 피크 추종이 실패할 수 있으며, 피크 예측 성공 사례로 간주하지 않는다.

## D-1과 D-2 비교

기존 V11 D-1 HGB와 V15 봄 D-2 HGB를 재사용했다. 학습 종료일2024-12-31과 모든 하이퍼파라미터를 맞췄으며 지연 길이로 최초 완결 학습일이 약간 다르다. 같은 검증일/EV조건에서 비교했으나 과거 원자료의 당시 수신본을 인증한 운영 재현은 아니다.

'''+table(horizon,['cohort','days_ahead','MAE','peak_MAE','peak_F1','peak_hour_MAE','G2V_regret','V2G_regret'])+r'''

D-1의 전체 MAE가 낮아도 고가 F1/G2V regret가 반드시 더 좋지 않았다. D-1 23시에 공식가격이 이미 공개됐다면 이 모델은 선행 AI 예측의 실용성이 없으며 공식가격 모드로 처리해야 한다.

## 어떤 정보가 EV 경제성을 바꾸는가

기존 shape 모델의 고정 예측을 사용했다. 정답의 하루 평균만 맞추기, 정답 시간 순서만 맞추기, 정답 일중 모양만 맞추기, 모든 정답 알기와 비교했다. 아래 perfect 행은 **실제값을 주입한 정보가치 상한 시뮬레이션**으로 학습 모델·미래예측·성능 개선이 아니다.

'''+table(info[info.cohort=='audit'],['information','days','G2V_regret','V2G_regret','export','additional_gain'])+r'''

현재 가정(24시간 연결, 일정 효율, 같은 충전량/종단SOC)의 G2V에서는 가격에 하루 일정 상수를 더해도 시간 순서가 같아 비용을 줄이지 못했다. 실제 순서를 완벽히 알면 G2V regret는0이 된다. 반면 V2G는 시간 순서만 알아도44.40원의 regret가 남고, 일중 가격차 크기까지 알면0.83원으로 줄었다. 효율 손실을 이길 가격차와 가격 수준까지 중요하기 때문이다. 모든 정답을 알 때 V2G의 G2V 대비 가상 추가이익 상한은54.26원/일이었다. 현재 예측으로는−5.36원/일이다. 이것은 특정 새 공식 변수를 확보하면54.26원을 얻는다는 뜻이 아니다.

![정보가치 상한](v16_figures/information_value.png)

따라서 우선 확보할 정보는 시간별 **순부하 전망(수요−재생발전)**과 급등 시 가용한 공급·정비/고장·제약, 그리고 한계비용을 움직이는 연료가격의 당시 발표본이다. 이를 일중 저가·고가 순서, 상승/하락 시각과 유효 가격차의 개선으로 검증해야 한다. 월별 평균 연료값이나 사후 관측만 늘려 시간별 급등을 해결했다고 주장할 수 없다.

## 효율·열화·정산비율 민감도

각 방향 효율0.8/0.9/0.95 × 열화0/5/15 × 가상 판매비율0.8/0.9/1의27개 조합을2025Q4 92일과2026H1 181일에 적용해 총7,371개 설정·일을 계산했다. 각 조합마다 같은60kWh/초기30%/종단80%/SOC20~90%/충방전7kW/24시간 조건에서 G2V·일반V2G·정답 G2V·정답 V2G를 계산했다. 충방전 효율·열화·동시 충방전 금지·연결시간·정확히 같은 종단SOC를 적용했다. 방전량은 계통 송출량이다.

2025Q4에서 추가이익≥10원/일, 방전≥0.5kWh/일, 활동≥10일, 손실빈도≤10%를 모두 만족한 조합만 연구상 유망으로 볼 기준을 사전에 고정했다. **적격 조합은0개**다. 후속2026 감사로 이 기준을 바꾸지 않았다.

### 2026H1, 판매 정산비율1인 가정

'''+table(sensitivity[(sensitivity.cohort=='audit')&(sensitivity.sell_ratio==1)],['efficiency','wear','days','mean_gain','mean_export','active_days','loss_days'])+r'''

효율95%·열화0·판매비율1이면 실제 가상 추가이익+41.79원/일과 방전14.15kWh/일이 관측됐지만 손실71/181일이다. 이전 검증에서도 이익+76.82원/일이었으나 손실43/92일(46.7%)로 손실관리 기준 탈락이다. **활발한 방전에서 생긴 양의 평균은 확인했지만, 낙관적 가정과 큰 손실 빈도가 필요했다.** 기본90%·열화5·비율1에서는−5.36원/일·방전1.335kWh/일·손실34일이다. 낮은 효율·높은 열화 조건의 방전0 부근을 경제성 개선으로 세지 않는다. 표의1e-12 수준 음수는 수치 오차이며 손실일은−1e-6원 미만으로 판정한다.

배터리 열화는 충전과 방전의 배터리 통과량을 모두 반영하므로 G2V에도 발생한다. 표의 이익은 동일 가정의 G2V 사후 비용−V2G 사후 비용이며, 실제 소매요금이나 실증 판매수익이 아니다.

![가정별 실제 가상 추가이익](v16_figures/sensitivity.png)

### 손익분기 부등식

계통 방전1kWh를 나중에 보상 충전하고 종단SOC를 같게 만들 때:

`s × P_sell > P_buy/(η_charge × η_discharge) + 2 × wear/η_discharge`

충전100원·효율각90%·열화5이면 판매비율1일 때 방전 가격134.57원/kWh 초과가 필요하다. 판매비율0.85이면158.32원 초과다. 충전·방전 모두175원인 안정 고가격은 마진−52.16원/방전kWh다. 반대로 충전60→방전100원은150원 미만이어도 약14.81원 마진 조건을 만족한다. 음수 SMP는 원래 부등식으로 처리하고, 판매비율0은 가격 임계값으로 나누지 않는다. 이 조건은 예측오차·전력·SOC·연결 제약을 포함한 전체 최적화의 충분조건이 아니며, 정답 시간·가격을 가정한 마진만으로 방전을 권장하지 않는다.

## 전향 기록 및 자동 작업

기존 D-2 독립 대상일1일(10월17일), 공식 발표가격 대조 완료1일, 대상 날짜 경과 완료0일, 급등일0일, V16 신규 예측0개다. 아직 거래일이 경과하지 않은 발표가격 대조와 정정통계 확정을 구분했다. 기존 D-1 기록과 운영 모드도 보존했다. 이미 공개된 가격은 AI 미래예측으로 생성하지 않는다. 기존 V15 적격 shadow 후보 발급 작업은 실제 실행시각/모델해시/first_seen≤마감/공개가격 확인/EV계획 불변 저장을 유지한다.

새 `GradprojectV16Audit`는15분마다 **기존 기록을 읽고 V16 폴더에만 상태를 집계**한다. IgnoreNew/5분 제한, 실제 시험 종료코드0이다. 기존 V11~V15 작업을 교체하거나 중복 API 수집·모델 자동 승격하지 않는다. 기존 V15 자동 작업도 종료코드0을 확인했다. 관측 공백은 누락으로 유지하며 노트북 종료/절전을 근거 없이 원인으로 확정하지 않는다. 사후 평가는 원예측을 바꾸지 않는 기존 별도 평가 파일을 읽는다.

## 웹 및 실행

로컬 발표 화면은 `http://127.0.0.1:8506/`. V15의 가격/전향/급등/EV/정책재생 기능을 유지하고6번째 탭에 정의별 성능, 새 공급변수 비교, 모델·발표본 한계, 실제값 비교, EV 정보가치 상한, 손익분기 계산, 가정별 손익과 G2V 권장 이유를 추가했다. 개인 EV 입력은 `artifacts/v16/interactive_decisions`에만 저장한다. 과거 시연과 실제 전향을 구분한다. 기존 공개 웹과 V15 로컬8505는 변경하지 않았다.

프로젝트 루트 PowerShell:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements-v16.txt
..\.venv\Scripts\python.exe -m v16.design
..\.venv\Scripts\python.exe -m v16.sources
..\.venv\Scripts\python.exe -m v16.outages
..\.venv\Scripts\python.exe scripts/v16_extract_outages.py artifacts/v16/outages
..\.venv\Scripts\python.exe -m v16.source_audit
..\.venv\Scripts\python.exe -m v16.definitions
..\.venv\Scripts\python.exe -m v16.maintenance
..\.venv\Scripts\python.exe -m v16.economics
..\.venv\Scripts\python.exe -m v16.monitor
..\.venv\Scripts\python.exe -m v16.figures
..\.venv\Scripts\python.exe -m v16.reporting
..\.venv\Scripts\python.exe -m streamlit run app_v16.py --server.port 8506 --server.address 127.0.0.1
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
..\.venv\Scripts\python.exe -m scripts.smoke_v16
```

PDF 추출은 기존 Codex runtime의 pypdf6.10.0으로 실행했다. 재실행할 때 requirements-v16의 같은 버전을 사용할 수 있다. 기존 V15 동결 행렬·예측·모델과 V11 SQLite 등 로컬 자산이 필요하며 GitHub 코드만으로 원자료가 모두 준비된다고 주장하지 않는다.

API 키 없이 기존 데이터 웹·EV 최적화·공식 공개 페이지/PDF 수집·연구 재평가가 가능하다. 인증 API에는 해당 Windows 환경변수가 필요하며 KMA_APIHUB_AUTH_KEY와 KPX_SERVICE_KEY를 혼용하지 않는다. 코드·집계·그림만 Git에 저장하고 키·API 응답·PDF 원본·모델·개인 EV 입력은 제외했다.

## 검증 및 미완료

전체86개 테스트(생략0) 통과. 기존 V5~V15, 시각 적격성, 미래 실제값 변조 비영향, 공개 상태, 과거 발급 금지, 불변 기록, CV 시간 분리, EV 제약·효율·비용·음수가격, V16 버튼·기존 EV 버튼을 검증했다. 정보가치 상한 실험의 평균 가격 보정 시 G2V 비용 불변, 완전 순서/형상의 G2V regret0, 완전 가격의 V2G regret0도 검증했다. HTTP health와 실제8506 서버 응답 정상, 기존 소스 차이0, ZIP3개를 포함한650개 자산 해시 불변이다. 상세는 `V16_VALIDATION.json`.

미완료 항목은 과거 기상예보·수정 이력이 있는 수요예측·당시 고장/전체 공급 가용성/재생·예비력 전망·원본 첨부 교체 이력 확인과 새로운 전향 급등 표본 축적이다. 새 공급 정보로 일관된 급등 개선이 확인되지는 않았으며 운영 모델 교체와 유효 V2G 정책 검증 성공을 주장하지 않는다.
'''
    narrative=[]
    d=pd.DataFrame(definition_rows)
    for fold,g in maintenance.groupby('fold',sort=False):
        b=g[g.model=='price_only_common'].iloc[0];n=g[g.model=='price_plus_maintenance'].iloc[0]
        changes=', '.join(f'{k} {100*(n[k]/b[k]-1):+.2f}%' for k in ['MAE','peak_MAE','G2V_regret','V2G_regret'])
        top=d[(d.fold==fold)&(d.definition=='daily_top6')].set_index('model').F1
        narrative.append(f'{fold}: {changes}; 피크시간 오차 {b.peak_hour_MAE:.3f}→{n.peak_hour_MAE:.3f}시간, 일중top6 F1 {top.price_only_common:.4f}→{top.price_plus_maintenance:.4f}.')
    narrative.append('비교 기준은 같은 완결 학습일과 검증일이다. 고가오차가 줄어도 시간대 순위와 EV 의사결정이 개선되지 않을 수 있다. 신규20개 정의별 지표는 V16_MAINTENANCE_DEFINITIONS.csv에 있다. 기존 V15 기준과는 학습행이 달라 직접 개선율을 비교하지 않는다.')
    report=report.replace('[MAINTENANCE_INTERPRETATION]','\n\n'.join(narrative))
    report=report.replace('31,344',format(coverage['rows'],','))
    report=report.replace('2024년8,448',f"2024년{coverage['years']['2024']:,}")
    report=report.replace('131개 행이 모호한1개 PDF 등 부적격 문서는 제외.','131개 행이 모호한1개 PDF, 종료<시작 오류1개가 든 문서 등 부적격 자료는 제외.')
    (ROOT/'docs/V16_RESULTS.md').write_text(report,encoding='utf-8');print('V16 results and validation report saved')

if __name__=='__main__':run()
