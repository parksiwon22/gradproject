"""Public aggregate report; source prices, user dispatches and models stay local."""
import json
import numpy as np
import pandas as pd
from v11.store import ROOT,now
from v12.research import HOME

def table(frame,columns):
    result=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for row in frame[columns].itertuples(index=False,name=None): result.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return result

def main():
    docs=ROOT/'docs'; study=HOME/'study'; analysis=HOME/'analysis'
    main_scores=pd.read_csv(study/'selection_metrics.csv'); follow=pd.read_csv(HOME/'relative_followup/selection_metrics.csv')
    selected=json.loads((study/'selection_before_audit.json').read_text(encoding='utf-8'))
    scores=pd.concat([main_scores,follow,pd.read_csv(study/'audit_metrics.csv')],ignore_index=True)
    scores.to_csv(docs/'V12_MODEL_COMPARISON.csv',index=False)
    experiments=json.loads((study/'experiments.json').read_text(encoding='utf-8'))+json.loads((HOME/'relative_followup/experiments.json').read_text(encoding='utf-8'))
    (docs/'V12_EXPERIMENTS.json').write_text(json.dumps(experiments,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    ev=pd.read_csv(analysis/'ev_summary.csv'); ev.to_csv(docs/'V12_EV_COMPARISON.csv',index=False)
    original=pd.read_csv(study/'selection_predictions.csv'); extra=pd.read_csv(HOME/'relative_followup/selection_predictions.csv'); names=main_scores.name.tolist()+follow.name.tolist()
    pred={n:(original[n].to_numpy() if n in original else extra[n].to_numpy()) for n in names}; unique=[]
    for n in names:
        if not any(np.allclose(pred[n],pred[u]) for u in unique): unique.append(n)
    known=json.loads((analysis/'official_EV_example.json').read_text(encoding='utf-8')); peak=json.loads((analysis/'known_peak_EV_example.json').read_text(encoding='utf-8'))
    risk=json.loads((analysis/'risk_metrics.json').read_text(encoding='utf-8'))
    summaries=json.loads((HOME/'evaluation_index.json').read_text(encoding='utf-8'))
    states=pd.Series([r['status'] for r in summaries]).value_counts().to_dict()
    lines=['# V12 실제 미래 검증·급등 연구·EV 충방전 결과','',f'작성 {now()} · V11 기준 b41b16d · 연구 브랜치 research/v12-forward-ev-20261015','',
      '기존 V5, V11, main, 원본, 운영 모델, 불변 예측 기록을 보존했다. 별도 app_v12.py와 artifacts/v12를 사용한다. 기존 공개 웹 배포는 수행하지 않았다.',
      '', '## 공개시각 검증','',
      '최신 공식 아카이브의2026.7.25 시행 규칙 PDF를 실제 다운로드하고 SHA256 및 발췌 페이지를 로컬에 보존했다. 제2.4.2조의2는 지역별 SMP 공개의 평일 원칙을 전일17시까지로 정하고 예외를 두며(PDF71~72쪽), 제8.2.3.7조와 별표9 공개 절차는 회원 전력거래시스템 통지를 설명한다(178,677~678쪽). 휴일·연휴 및 장애 관련 예외와 재통지 가능성이 있다. 이 회원 통지 시각을 일반 홈페이지 최초 노출이나 EPSIS 갱신시각으로 단정할 수 없다.',
      '', '일반 공개 육지 홈페이지에서10월16일 가격을10월15일22:04 KST에 이미 수신한 로그가 있다. 같은 날 늦은 시각에도 EPSIS의 해당일 조회에서는 확인되지 않았다. 이는 공개 경로의 갱신 차이를 보여주며, 홈페이지의 정확한 최초 공개시각은 모른다. 로그인하지 않은 공개 페이지와 EPSIS를 분리해15분 간격으로 관측하고 source별 마지막 부재/첫 수신 구간을 기록한다. 실패 응답을 부재 증거로 사용하지 않고 전 구간0원인 미확인 응답도 공개 확정으로 사용하지 않는다.',
      '', '육지 기존 시간 SMP, EPSIS 제주 시간 자료, 제주 시범 하루전/실시간 가격은 서로 대체하지 않는다. 제주 시범사업의 별도 시장·수립·통지 규정이 있으므로 기존 제주 시간 통계만으로 시범 실시간 공표를 인증하지 않았다. 최초 일반 공개시각과 시범시장 가격의 상세 가용성은 계속 미확정이다.',
      '', 'D-1 23시 연구 모델은 유지했다. 새 모델은 D-2 23시 입력 기준의24시간 대상가격을 추정한다. 기존 V11의2일 피처 생성기를 재사용했으며 임의로 기존 모델 시각을 변경하지 않았다. 이 기준도 조기 공표·휴일 예외가 있을 수 있어 실행마다 공식가격 기확보 여부와 조회 출처를 확인한다. 이미 확보한 가격은 official_known_diagnostic으로 기록한다.',
      '', '[공식 규칙 공고](https://www.kpx.or.kr/board.es?act=view&bid=0030&list_no=77795&mid=a11002010000), [전문 PDF](https://www.kpx.or.kr/boardDownload.es?bid=0030&list_no=77795&seq=4), [제주 시범 실시간 공개 화면](https://www.kpx.or.kr/bidSmpLfdDataRt.es?device=pc&division=smpDataRt&gubun=today&mid=a10406020200).',
      '', '## 급등 실패 원인과 실험','',
      '이미 본2026-10-13 고가6시간은10·11·13·14·15·16시이며 실제값 범위는174.98~175.61원/kWh였다. V11 지연 관측 모델은110.84~118.99로 과소예측했다. 같은 시각 전일 가격은대략107.85~119.13이었고, 보존된 수요예측 값은 전일 같은 시각보다 증가했다. 일별 예비율25.7%는 피크상태 일별 수치이므로 해당 시간 희소성이나 급등 원인을 증명하지 않는다. 사후 기상·수급을 진단용으로만 결합했다.2026 시간 수요 실적, 발전기별 입찰·가용성·정비·한계발전기·연료비가 없어 인과 원인은 확정할 수 없다.',
      '', '지속성 가격 피처 중심의 평균 회귀는 드문 가격체제 변화를 평활화한다. 학습2023~2025상반기의 절대 가격 분포와 최근 분포도 달랐다. 학습 q95는241.63으로150고정 기준과 매우 다르고, 검증2025Q4의150초과 사례는7시간뿐이다. 연도·분기별 빈도 및 피처 근접 학습 사례를 로컬 CSV로 저장했다. 급등 과소예측을 전체 MAE 평균에 묻지 않고 별도 보고한다.',
      '', f'가격·경고 조건 {len(experiments)}개를 실행했다(선택 구간 가격 출력은 {len(unique)}종류). 27개 본 실험과3개 상대급등 후속이다. 위험 경고만 추가한 조건과 보정량0인 조건은 원 가격 예측을 재사용하므로 신규 가격 학습 성과로 세지 않는다. 성공/실패·가설·파라미터·씨드·피처를 V12_EXPERIMENTS.json에 기록했다.',
      '', '5개 계열 HGB/ExtraTrees/CatBoost/LightGBM/XGBoost의 기본·고가 가중·ramp 가중, 분위0.75/0.90, 확률보정 분류기+조건부 가격 전문가, 보정 잔차, 원점 변화량, 재귀 ramp, 일별 최대·시각 모델, 검증 앙상블을 비교했다. 절대 고가 기준의 가격수준 변화 문제를 확인한 뒤 학습 잔차 q95 기반 상대급등3조건을 추가했으나 EV regret가 개선되지 않아 추가 사후 평가/승격하지 않았다. 딥러닝은 제한된 희귀사례와 누락된 원인 피처를 해결한다는 근거가 없어 실행하지 않았다.',
      '', '학습2023~2025상반기(21,672행), 확률·게이트 보정2025Q3(2,208행), 선정2025Q4(2,208행).2026상반기는 이미 관측된 회고 감사이고10월1~15일은 이미 확인한 실패 진단이다.2026Q3로 재튜닝하지 않았다. 모든 역사 구간이 이전 연구에서 조회됐을 수 있으므로 새 미열람 테스트라고 주장하지 않는다. 새로운 독립 성능 근거는 이후 저장한 실제 전향 기록에서만 축적한다.',
      '', '고정 검증 조건을 통과한 상위3개: '+', '.join(selected['top3'])+'. 다음 표는 선정2025Q4의 동일 D-2 조건이다. D-1 V11 MAE6.47과 직접 비교하지 않는다.','']
    lines+=table(main_scores[main_scores.name.isin(['hgb_none']+selected['top3'])],['name','MAE','peak_MAE','peak_F1','EV_regret'])
    lines+=['', '2026상반기 회고 감사에서는 LightGBM 피크 가중이 고가 MAE를 개선했으나 전체 MAE와 F1은 악화했다. 앙상블은 전체 MAE가 소폭 낮지만 고가 MAE/F1은 일관되게 개선되지 않았다.10월 고가6시간의F1도0이었다. 기존 운영 모델을 교체하지 않는다. 모든30조건과 감사 지표는 V12_MODEL_COMPARISON.csv에 있다.',
      '', f'급등 확률의 검증 Brier={risk["Brier"]:.6f}, 상수 사전확률={risk["constant_base_rate_Brier"]:.6f}. 상수보다 개선되지 않아 확률을 유용성이 검증된 경고로 표시하지 않는다. ROC-AUC가 높아도 희귀 사건의 정확한 경고·가격 예측을 보장하지 않는다.',
      '', '## 전향 예측 및 EV 의사결정','',
      f'10월17일 대상 기본 연구 모델+상위3후보의24시간 예측을 실제 저장했다. 현재 기록 상태: {states}. 실제값이 도착하지 않은 날짜에는 MAE나 regret를 계산하지 않았다. 기존 V11 과거 기록을 새 V12 전향 성과로 재분류하지 않았다.',
      '', '실제 생성시각·D-2 입력 기준·생성 지연·모델 해시·피처별 값·원 가격의 first_seen/스냅샷·공식가격 확보 여부를 저장한다. 기상/수요 예보 사용은 false이며 미확인 예보를 입력하지 않는다. 원자적 JSON 생성 후 기존 파일 교체를 금지한다. 사후 실제값 정정은 새로운 해시별 평가 파일에만 기록한다. 부족한 입력은 skip하고 사유를 남긴다.',
      '', 'G2V는 충전가능 시간·전력·효율·SOC·목표를 적용한 선형 최적화다. V2G는 추가 방전 한계·효율·SOC 하한·출발 최소 SOC와 이진 모드로 동시 충방전을 금지하는 MILP다. 부분시간 연결, 음수 SMP, 불가능한 목표를 검증했다. 판매정산 비율·배터리 처리 에너지당 열화비를 사용자 가정으로 둔다. 실제 국내 소매요금과 실제 판매수익이 아니다.',
      '', '사용자 화면의 충방전 계획과 입력 가격도 비공개 로컬에 불변 저장하고 실제가격 수신 후 자동 regret를 계산한다. 별도 고정 실험에서는60kWh,30→80%,7kW,충방전효율90%,SOC20~90%,열화5원/배터리kWh,종일 연결을 가정해2026상반기1,448건 계획을 실행했다. 이 연결 패턴은 실제 개인 주행패턴이 아니다.','']
    lines+=table(ev,['model','mode','realized_net_cost','oracle_cost','regret'])
    lines+=['', '평균적으로 예측 기반 V2G의 실현 비용은 G2V보다 높아질 수 있었다. 완전예측에서 가능한 차익을 실제 예측의 이익으로 주장하지 않는다.',
      '', '실제 공식가격 기확보 예제: '+known['target_date']+'18~24시에서는 가격이 평탄해 즉시·균등·G2V·V2G의 열화 포함 비용이 모두 약4,059.33원이었다. 절감이 없는 경우도 그대로 표시한다.',
      '', '이미 본10월13일 종일 연결·완전가격 사후 예제는 즉시3,471.13원,균등4,581.26원,G2V3,409.67원,V2G2,734.38원이다. 가상 V2G의 G2V 대비675.29원 차이는 미래 예측으로 실제 확보한 수익이 아니다. 예측 오차를 반영한 비용/regret를 별도 저장했다.',
      '', '## 실행·자동 수집·테스트','',
      '별도 로컬 화면: http://127.0.0.1:8502 · app_v12.py. 공식가격 우선/저장 예측 연구를 구분하고 연결 시작·종료일시,배터리·SOC·전력·효율·열화·판매비율을 입력한다. 시간별 전력/SOC/가격 출처,즉시·균등 대비 비용,실제값 도착 후 regret를 표시한다.24시간 범위를 넘는 연결은 보유 공식·예측 가격이 이어지는 범위에서만 허용하며 빈 가격을 만들어 채우지 않는다.',
      '', '기존 Windows GradprojectV11Local 수집 작업 유지. 별도 GradprojectV12ForwardEV는15분 간격 공개 수신 관측·전향 기록·사후 평가를 수행한다. 노트북 중단/절전 시 실행은 보장하지 않는다.30개의 완료된 전향 날짜 이후 수동 후보 검토 준비 상태를 알리며 자동 학습·운영 승격·GitHub 업로드는 없다. 새 후보 검토는 v12.retrain --as-of로 새 폴더에만 저장한다. 이는 구현됐으나 충분한 새 전향 데이터가 없어 아직 실행하지 않았다.',
      '', 'V5 재현/보존과 V11 회귀를 포함한41개 테스트 통과. V12 불변성·미래입력 차단·미도착 미평가·오류 응답과 부재 구분·동일 예측을 개선으로 선정하지 않기·전력/SOC/효율/동시충방전/비용/불가능 목표 및 실제 Streamlit 최적화 버튼 검증을 포함한다. 별도 HTTP health 통과. V12 파이프라인5단계 모두 성공. 최초 Windows PowerShell 하위 프로세스 실행 정책 오류는 현재 PowerShell에서 직접 등록해 해결했고 시스템 실행 정책은 바꾸지 않았다.',
      '', '```powershell',r'..\.venv\Scripts\python.exe -m streamlit run app_v12.py --server.address 127.0.0.1 --server.port 8502',r'..\.venv\Scripts\python.exe -m v12.pipeline',r'..\.venv\Scripts\python.exe -m unittest tests.test_v12 -v',r'..\.venv\Scripts\python.exe -m v12.research',r'..\.venv\Scripts\python.exe -m v12.relative_followup',r'..\.venv\Scripts\python.exe -m v12.analysis',r'..\.venv\Scripts\python.exe -m v12.plots',r'..\.venv\Scripts\python.exe -m v12.report','```',
      '', '연구 재실행은 기존 설계 파일이 있으면 거부한다. 예: v12.research --output artifacts/v12/reproductions/run_new 처럼 새 디렉터리를 지정한다. 재현 실행은 전향/운영 모델을 자동 교체하지 않는다. 불변 전향/충방전 기록은 수정하지 않는다. 새 데이터 후보는 v12.retrain을 사용한다.',
      '', '## 미완료','',
      '일반 홈페이지의 정확한 최초 공개시각, 회원 정보와 일반 공개의 전송 지연, 제주 시범 가격의 별도 완전 수집, 역사 발표 기상예보 API403, 당시 수요예측 판본, 원인 발전·연료·정비 자료가 미해결이다.10월16일06:06 KST 기상예보 API를 한 번만 재확인했고403이었다. 노트북 실행 공백으로 관측 간격이 벌어진 구간은 summary.json에 간격을 기록하며 연속 관측으로 주장하지 않는다. 새 전향 대상의 실제가격이 도착하기 전에는 실제 예측 성능과 실제 의사결정 개선을 확정할 수 없다. 피크 탐지 개선은 검증되지 않아 기존 운영 모델을 유지한다.',
      '', '로컬 상세 자료: artifacts/v12/analysis/known_six_peak_hours.csv, training_price_feature_analogs.csv, seasonal_peak_incidence.csv, ev_daily_backtest.csv; 그래프 artifacts/v12/figures/*.png. 원자료·모델·개인 입력은 Git 제외하고 집계표·코드만 개인 GitHub 연구 브랜치에 저장한다.']
    (docs/'V12_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('V12 aggregate report generated:',len(experiments),'conditions',len(unique),'distinct selection price outputs')

if __name__=='__main__': main()
