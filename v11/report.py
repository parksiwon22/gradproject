"""Publish aggregate metrics and execution instructions, never private rows."""
import json
import pandas as pd
from v11.store import ROOT,HOME

def table(f,columns):
    def val(x): return f'{x:.4f}' if isinstance(x,float) else str(x)
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+['| '+' | '.join(val(r[c]) for c in columns)+' |' for _,r in f.iterrows()])

def main():
    coverage=pd.read_json(HOME/'coverage.json')
    m=pd.read_csv(HOME/'training/metrics.csv')
    selection=json.loads((HOME/'training/selection_before_test.json').read_text(encoding='utf-8'))
    audit=json.loads((HOME/'source_audit.json').read_text(encoding='utf-8'))
    public=json.loads((HOME/'public_availability.json').read_text(encoding='utf-8'))
    historical=json.loads((HOME/'training/2023_context.json').read_text(encoding='utf-8'))
    future=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((HOME/'forecasts').glob('*.json'))]
    m.to_csv(ROOT/'docs/V11_AGGREGATE_METRICS.csv',index=False)
    hgb=m[(m.split=='test_2026Q3')&(m.model=='hgb')]
    selected=m[(m.split=='test_2026Q3')&(m.stage==selection['stage'])&(m.model==selection['family'])]
    profile=m[(m.split=='test_2026Q3')&(m.model=='profile')]
    text='''# V11 실제 실행 보고 — 2026-10-15

## 현재 결과

공식 다년 SMP 수집·revision DB·모델24개 시간순 실험·불변 예측기록·사후비교·별도 Streamlit 웹·로컬 예약실행을 구현했다. 원 V5/main/기존 app.py/outputs는 보존했다. 공개 배포나 main 병합은 하지 않았다. V11은 운영 기능을 갖춘 연구 시스템이며 원자료의 역사적 발표 vintage와 미래예측 적격성은 아직 완전 검증되지 않았다.

## 실제 확보량

'''+table(coverage,['market','year','hours','days','expected_full_year','missing_between_endpoints','latest'])+'''

육지2024는8,784시간,2025는8,760시간 완전하다. 2026의10월16일24시간은 미래 거래일이지만 이미 공개된 공식가격이다. 이를 관측 수요나 AI 미공개가격 예측으로 부르지 않는다. 제주2024는3월21일24시간 누락이며 해당 날짜 단독 재조회도 빈 응답이었다. 억지 보간하지 않았다. 제주 음수가격은 원값을 보존한다. 2024년 제주 제도변화 전후의 market product는 추가 확인이 필요하며 육지 모델에는 혼합하지 않는다.

기존2023 CSV와 공식 육지 SMP를 같은시각8,760개 비교했으며 모두 일치했다(가격 MAE0). 역사적 실제 발표시각·정정본 가용성을 입증한 것은 아니다.

전국2025 수요 공식 CSV8,760시간(MWh,육지+제주)을 실제 확보하고 통합 데이터셋에 별도 결합했다. 원 V5 national_demand_mw와 동일 원천/지역 정의라고 인증하지 않는다. 발표 vintage가 없어 학습에는 제외했다. 63개 원피처 각각의 생성/이식 여부는 V11_FEATURE_AUDIT.md 참조. 동일한63피처가 완비된2024~2026 날짜는0일이다. V11은 독립21개 SMP/달력 피처만 사용한다.

## 출처와 공개시각

- [KPX EPSIS 시간별 SMP](https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpShdChart.do?menuId=040202): 공식 공개 조회 AJAX의 실제 요청 형식을 읽어 사용. 인증키 불필요. c1=01시 끝 구간→시작00시, c24→시작23시, Asia/Seoul, 원/kWh. 원응답hash·source·retrieved_at·first_seen·revision 저장.
- [KPX 일반 공개 육지 SMP](https://www.kpx.or.kr/smpInland.es?mid=a10404080100): issue_date별 마지막 날짜열을 파싱해 실제 웹에서 이용 가능한 가격을 별도로 확인. EPSIS의 후행 업데이트를 미래 미공개로 오인하지 않음.
- [공공데이터포털 SMP/수요예측 API](https://www.data.go.kr/data/15131225/openapi.do): 공식 설명은1시간 끝점·하루1회23시경 갱신·무료·개발100호출. 활용신청 인증키 미설정이라 인증 API는 호출하지 않았다. 검증 안 된 REST endpoint를 발명하지 않고 실제 공개 조회를 수집 경로로 택했다.
- [전국 시간별 수요 CSV](https://www.data.go.kr/data/15065266/fileData.do): 무료 공식 파일. 실제 다운로드 링크를 페이지 메타데이터에서 해석해 다운로드했고 로컬 원문 보존.
- [기상청 단기예보](https://www.data.go.kr/data/15084084/openapi.do): 키/과거forecast-vintage 미확보. 관측기상을 미래 예보로 대체하지 않음.
- [KPX 공개절차 설명](https://www.kpx.or.kr/board.es?act=view&bid=0048&list_no=73368&mid=a10504030000): 통상17시 절차와 휴일 예외 안내. 23시 API 갱신을 최초 공개시각으로 해석하지 않음. 현재 일반 공개 웹에서 실제 수신한 시각과 회원 공표규정은 별개.

웹에서10월16일 가격을 이번 실행 중 이미 받았다. 10월17일 페이지에서는24개 가격을 관측하지 못했다. 이것이 모든 접근 경로의 미공개를 입증하지는 않는다. 실패/빈페이지/all-zero placeholder는 official_known으로 인증하지 않는다. 공개 검사 원문 및 당시hash를 로컬에 보존한다.

## 시간순 실험과 성능

모든 모델 seed42, 잔차 target=실제가격-관측 가능한 동일시간 과거가격. HGB/피크가중HGB/ExtraTrees/CatBoost/LightGBM/XGBoost 각각4개 학습범위에서 실제 실행했다. 가중HGB는 학습가격>=150일 때4배. HGB 랜덤 early stopping 비활성, 다른 모델 고정 iterations. 2023만/2023~24/2023~25/2026Q1까지 학습을 같은 검증·테스트 구간에서 비교했다.

검증2026-04-01~06-30(2,184시간), 평가2026-07-01~09-30(2,208시간). 모델선정은 검증 전체MAE 최소, 동률 피크MAE. 선정기록을 쓴 뒤 처음으로 이 실행의 Q3 지표를 계산했으며 이후 성능을 보고 모델을 바꾸지 않았다. 역사자료 다운로드는 모델선정 전에 이뤄졌으므로 암호학적 블라인드 테스트는 아니다. 발표 로그 없는 과거 replay이며 당시 사용 가능했던 데이터임을 완전히 보증하지 않는다.

검증 선정: `'''+selection['stage']+' / '+selection['family']+'''`.

### 같은 HGB에 데이터를 추가한 영향 — 공통2026Q3 평가

'''+table(hgb,['stage','MAE','RMSE','R2','peak_MAE','peak_Bias','peak_F1','daily_max_MAE','peak_hour_MAE','EV_regret'])+'''

### 검증선정 모델과 프로파일 — 공통2026Q3

'''+table(pd.concat([selected,profile]),['stage','model','MAE','peak_MAE','peak_Bias','peak_Precision','peak_Recall','peak_F1','false_alarms','ramp_F1','EV_regret'])+'''

선정모델의 Q3 전체MAE는 프로파일보다 나쁘지만 가상 충전regret는 개선됐다. 무조건 V5/프로파일보다 우수하다고 결론 내리지 않는다. 과거평가용 모델과, 선택 family를 최신 허용 이력으로 refit한 production 모델은 다른 바이너리로 보존한다. production refit의 미래성능은 아직 미측정이다. 전체24개 모델의 모든 지표는 V11_AGGREGATE_METRICS.csv, 설정/파라미터/기간/성공여부는 로컬 training/config_before_evaluation.json 및 experiments.json에 있다.

가격>=150의 high-price F1와 상승15원/h의 ramp F1는 별개다. 피크Bias=예측-실제. 최고가 발생시각오차는 argmax의 절대 시각차(시간), 동률이면 첫 시간이다. EV는1kW, 하루24시간 접속, 연속2/3/4/5/6/8시간을 동일 가중한 가상 SMP 연동 비용이며 실제 EV 소매요금/V2G 수익이 아니다. 웹은 별도 SOC/출력/효율90%의 부분충전 계산을 제공하므로 백테스트 평균regret와 같은 시나리오가 아니다.

### V5와2023 동일 평가기간의 참고 비교

V5 MAE11.093786, R²0.642919(원63피처·기존검증정책). 새21피처 ExtraTrees를2023년1~8월에 학습한 탐색 비교 MAE '''+f"{historical['metrics']['MAE']:.6f}"+''', 피크MAE '''+f"{historical['metrics']['peak_MAE']:.6f}"+''', 피크F1 '''+f"{historical['metrics']['peak_F1']:.6f}"+'''. V5의 피크MAE12.886623/F1 .585757보다 나아졌지만 새 ExtraTrees의regret '''+f"{historical['metrics']['EV_regret']:.6f}"+'''은 V5 2.464330보다 소폭 나빴다. 이 모델은2026검증에서 선택된 HGB와 다르고, 이미 본2023 구간의 탐색 참고다. 원 V5 63피처가 없어2026년의 실제 V5 추론과 V11 동일조건 비교는 미완료다.

## 실제 생성 기록

'''+table(pd.DataFrame([dict(target=r['target_date'],issued_at=r['issued_at'],mode=r['mode'],model=r['model_family'],days_ahead=r['days_ahead']) for r in future]),['target','issued_at','mode','model','days_ahead'])+'''

기본 D-1 23시 next-day24시간 정책은 유지한다. 아직23시 이전인 실제 실행은 preview로 명시했다. 별도2일 horizon ExtraTrees는 D-2 23시 기준 독립 피처/모델로 학습했고 선공개되지 않은 다음다음날 후보를 보조 출력한다. 23시 이전 preview는 현재 이미 받은 공식가격만 사용하며 이후23시 관측값을 썼다고 주장하지 않는다. 실제23시 실행과 동일한 실험으로 섞지 않는다. 공개 적격성 미확인이라는 제한을 파일과 웹 모두 표시한다.

예측파일은 exclusive-create로 저장하고 동일 대상/모델/모드의 기존 파일을 재사용한다. 생성시각·target·모델hash·학습datahash·입력snapshot버전·공개probe를 포함한다. 공식가격 도착 후 별도 evaluation 파일을 추가하며 원예측을 덮어쓰지 않는다. 실제값이0개인 미래기록은MAE를 만들지 않는다. 이미 알려진 가격의 모델오차는 미래 예측 점수에서 구분한다. 불확실성은 검증 잔차 절대값90분위수 구간이며 미래구간 coverage나 급등확률을 보장하지 않는다.

## 자동 축적과 저장

SQLite에시장/시간/정정version을 append한다. 동일값은 새 price row를 만들지 않고 last_seen만 갱신한다. 원응답은SHA256별 저장해 바이트 동일 파일을 중복 생성하지 않는다. 증분조회는 최근7일을 재확인하고 장기정정은 --backfill 재감사한다. 요청3회 재시도, 실패로그와 마지막 성공시각을 기록한다. 네트워크/무결성 실패 시 기존가격 보존. 새로운 수집이 기존학습모델을 자동으로 바꾸지는 않는다.

Windows 작업 `GradprojectV11Local`을 현재 개인 사용자로 등록했다. 매시간 수집 및 매일23:05 KST 작업이며 중복실행은 무시한다. 첫 자동실행은2026-10-15 22:18:46 KST에 시작했고22:19:31에 파이프라인4단계 모두 종료코드0으로 완료했다. 작업 스케줄러 LastTaskResult=0, Ready 및 다음실행23:05를 확인했다. 컴퓨터가 꺼져 있거나 로그아웃하면 실행되지 않을 수 있고 StartWhenAvailable로 복귀 후 실행한다. 모델 자동재학습·데이터 외부업로드는 포함하지 않는다.

[GitHub Actions 과금 안내](https://docs.github.com/en/billing/concepts/product-billing/github-actions)에 따르면 공개저장소 표준 hosted runner는 무료 사용 범위가 있으나 private 저장소/runner 종류/저장량에는 제한·과금이 적용될 수 있다. 노트북이 꺼져도 hosted runner는 동작하지만 원자료 비공개 저장·키권한·artifact노출·영구저장소를 별도로 설계해야 한다. 이번에는 Actions workflow나 원자료 자동커밋을 활성화하지 않았다.

로컬 backup 명령은 SQLite online backup과 원응답·생산모델·불변예측·manifest를ZIP으로 보존하고 닫은 후CRC와SHA256을 검사한다. 소스 Git Push는 데이터 백업이 아니다. 기본 백업은 같은 노트북의 artifacts/v11_backups이며 장치 고장 대비를 위해 사용자가 승인한 암호화 외장디스크/개인 비공개 저장소로 옮겨야 한다. 외부 전송은 하지 않았다.

## 실행 명령 — 저장소 루트 PowerShell

```powershell
$py = '..\\.venv\\Scripts\\python.exe'
& $py -m pip install -r requirements-research.txt
& $py -m v11.collect --backfill
& $py -m v11.public_prices
& $py -m v11.extras
& $py -m v11.audit
& $py -m v11.train
& $py -m v11.forecast --preview
& $py -m v11.pipeline
& $py -m v11.report
& $py -m v11.backup
& $py -m streamlit run app_v11.py
& $py -m unittest tests.test_v11 -v
& $py -m scripts.smoke_v11
Get-ScheduledTaskInfo -TaskName GradprojectV11Local
# 예약 등록/해제
.\\scripts\\schedule_v11.ps1 -Install
.\\scripts\\schedule_v11.ps1 -Remove
```

키를 발급받으면 채팅에 붙이지 말고 로컬 환경변수 KPX_SERVICE_KEY / KMA_SERVICE_KEY에 설정한다. 인증 API adapter는 최신 Swagger와 실제 응답 검증 후 추가한다. 현재 공개 조회 경로는 키가 필요 없다.

## 검증과 미완료

테스트14개가 최종 통과했다. 중복·시장분리·정정과 as-of vintage, API3회실패 시 기존자료 보존, 부분일·NaN·중복 거부, 음수가격, 두 horizon의 미래값 불변성, 공개상태 구분, 예측 불변, oracle regret0, 시간순 분리, Streamlit 조작을 검증했다. V5 기존 재현·추론도 재확인했다. pip check와 V11 Streamlit 실제 HTTP health도 통과했다. 초기 샌드박스 임시폴더 권한/HTTP 제약과 백업 writer 종료 전 CRC 검사에서 실패했으며 수정 후 재검증했다. 로그는 v11_tests_verified.log, v11_smoke_final.log, v11_backup_final.log에 보존했다.

미완료: 전체 경로의 최초 공개시각 증명, 원63피처의 다년 자료와 예보 vintage, 제주2024-03-21 누락 및 시장상품 전환 감사, 10월17일 가격 도착 후 전향오차, 원V5의2026 동일조건 평가, 노트북 종료 중에도 동작할 비공개 서버. 23시 이전 preview를 23시 본실행으로 주장하지 않는다.
'''
    # Keep the report in Korean; technical identifiers remain unchanged.
    (ROOT/'docs/V11_RESULTS.md').write_text(text,encoding='utf-8')
    print('Aggregate report:',ROOT/'docs/V11_RESULTS.md')

if __name__=='__main__': main()
