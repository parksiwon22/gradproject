"""Read-only replay of event diagnostics and all seven saved research policies."""
import json
from pathlib import Path
import pandas as pd,streamlit as st
from v15.research import HOME
from v15.policies import POLICIES
LABELS={'G2V':'G2V 충전만','general':'일반 V2G','v14_conservative':'V14 보수적 V2G','profit100':'예상이익100원 기준','cvar025':'시나리오 CVaR0.25','cvar1':'시나리오 CVaR1','scenario_cap50':'시나리오 손실50원 한도'}

def csv(path):
    try: return pd.read_csv(path, float_precision='round_trip') if path.exists() else pd.DataFrame()
    except (pd.errors.EmptyDataError,pd.errors.ParserError): return pd.DataFrame()

def render(official,chart):
    st.subheader('급등 유형과 모델의 정보 한계')
    st.write('절대150원 기준과 알려진 D-2 가격 대비20원 상승 기준을 함께 집계합니다. 사건의 유형은 사후 분석용이며 미래 입력으로 사용하지 않습니다.')
    year=csv(HOME/'events/year_incidence.csv')
    if not year.empty: st.dataframe(year.rename(columns={'year':'연도','hours':'확보 시간','mean':'평균 SMP','high150_hours':'150원 이상 시간','high150_fraction':'고가 비율','relative20_hours':'상대 상승 시간','relative20_fraction':'상대 상승 비율'}),hide_index=True)
    st.caption('①2023년 고가격 체제와2025년 희소 고가 사건의 비중이 다릅니다. ②과거 가격만 비슷해도 미래 공급 제약은 다를 수 있습니다. ③고가 사례 전체가 많은 것과 현재와 비슷한 급등 사례가 많은 것은 다릅니다. ④150원 경계가 한 사건을 나눌 수 있습니다. 이는 원인 후보 분석이며 실제 한계발전기·고장 원인을 확인한 것은 아닙니다.')
    events=csv(HOME/'events/event_catalog.csv')
    if not events.empty:
        types=st.multiselect('사건 유형',events.event_type.unique(),default=list(events.event_type.unique()))
        subset=events[events.event_type.isin(types)].copy().sort_values('start',ascending=False)
        if not subset.empty:
            idx=st.selectbox('급등 사건',range(len(subset)),format_func=lambda i:str(subset.iloc[i].start)+' · '+str(subset.iloc[i].definition)+' · '+str(int(subset.iloc[i].hours))+'시간')
            event=subset.iloc[idx]; start=pd.Timestamp(event.start).normalize(); end=min(start+pd.Timedelta(days=3),pd.Timestamp(event.end).ceil('D')+pd.Timedelta(days=1))
            g=official[(official.timestamp>=start)&(official.timestamp<end)][['timestamp','smp']].rename(columns={'smp':'actual_smp'}).copy(); names=[]
            files=list((HOME/'shrinkage_followup').glob('*_predictions.csv'))+list((HOME/'diagnostics').glob('*_predictions.csv'))
            for file in files:
                pred=csv(file)
                if pred.empty: continue
                pred.timestamp=pd.to_datetime(pred.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
                cols=[c for c in pred if c.startswith('two_stage_shrink_') or c=='baseline']
                if not pred.timestamp.between(start,end).any(): continue
                g=g.merge(pred[['timestamp']+cols],on='timestamp',how='left'); names.extend(cols)
            chart(g,list(dict.fromkeys(names)))
            st.caption('2026 곡선은 이미 열람한 기간의 진단입니다. 2024/2025 곡선은 학습 이후 시간순 검증 출력입니다. 2023 학습 구간의 예측을 검증 결과로 만들지 않습니다.')
            if names:
                rows=[]
                for name in dict.fromkeys(names):
                    valid=g[name].notna(); high=valid&(g.actual_smp>=150)
                    rows.append({'모델':name,'비교 가능 시간':int(valid.sum()),'고가 Bias(원/kWh)':float((g.loc[high,name]-g.loc[high,'actual_smp']).mean()) if high.any() else None,'고가 미탐지 시간':int(((g[name]<150)&high).sum())})
                st.dataframe(pd.DataFrame(rows),hide_index=True)
            else: st.info('이 사건에는 적격한 표본 외 예측이 없습니다. 실제 가격만 표시하며 학습 구간 예측을 검증으로 만들지 않습니다.')
    st.subheader('V15 정책 비교 · 고정 연구 조건')
    st.caption('배터리60kWh · 초기30% · 종료80% · 충방전7kW · 효율90% · 열화5원/배터리 처리kWh · 24시간 연결. 사용자 EV 탭의 별도 입력과 구분합니다. 실제 소매요금·실증 V2G 수익이 아닙니다.')
    path=HOME/'policy_comparison.json'
    if not path.exists(): st.info('정책 실험 집계가 실행 중입니다.'); return
    comparison=pd.DataFrame(json.loads(path.read_text()))
    cohort=st.selectbox('정책 비교 구간',['audit','selection'],format_func=lambda v:'2026H1 후향 감사(181일)' if v=='audit' else '2025Q4 선택 검증(92일)')
    model=st.selectbox('정책 비교 가격 모델',comparison.model.unique())
    selected=comparison[(comparison.cohort==cohort)&(comparison.model==model)].copy()
    selected['policy']=selected.policy.map(LABELS)
    names={'policy':'정책','mean_export_kwh':'방전량(kWh/일)','active_days':'활동일','mean_expected_additional_profit':'점예측 추가 이익(원/일)','scenario_mean_expected_gain':'시나리오 평균 기대이익','mean_actual_additional_profit':'실제 추가 이익(원/일)','loss_days':'손실일','active_loss_rate':'활동일 중 손실 비율','mean_degradation_cost':'열화비용(원/일)','mean_V2G_regret':'V2G regret','loss_over50_days':'실제 손실50원 초과일'}
    st.dataframe(selected[[c for c in names if c in selected]].rename(columns=names),hide_index=True)
    st.warning('사전 검증에서 평균 추가 이익10원/일·방전량0.5kWh/일 등 실효성 기준을 통과한 V2G 정책이 없어 기본 선택은 G2V입니다. 시나리오 손실 한도는 입력한 시나리오 내부의 제약이며 실제 가격의 손실 보장이 아닙니다.')
    general=csv(HOME/'policy_study'/(cohort+'_'+model+'_general.csv'))
    if not general.empty:
        dates=general.sort_values('actual_gain').day.tolist(); day=st.selectbox('정책 시연 날짜(일반 V2G 손실 큰 순)',dates)
        if st.button('저장된 7개 정책 계획 비교'):
            rows=[]; power=[]; soc=[]
            for policy in POLICIES:
                file=HOME/'policy_study/plans'/cohort/model/policy/(day+'.json')
                if not file.exists(): continue
                result=json.loads(file.read_text()); frame=pd.DataFrame(result['schedule']); frame.timestamp=pd.to_datetime(frame.timestamp,utc=True).dt.tz_convert('Asia/Seoul'); label=LABELS[policy]
                rows.append(dict(정책=label,방전량=result['export_kwh'],예상추가이익=result['expected_gain'],실제추가이익=result['actual_gain'],실제순비용=result['actual_cost'],열화비용=result['degradation_cost'],종료SOC=result['ending_soc'],G2V보다유리=result['actual_gain']>1e-6))
                power.append(frame[['timestamp']].assign(정책=label,순충전전력=frame.charge_kw-frame.discharge_kw)); soc.append(frame[['timestamp']].assign(정책=label,SOC=frame.soc_end))
            st.dataframe(pd.DataFrame(rows),hide_index=True)
            st.line_chart(pd.concat(power).pivot(index='timestamp',columns='정책',values='순충전전력'),y_label='kW')
            st.line_chart(pd.concat(soc).pivot(index='timestamp',columns='정책',values='SOC'),y_label='SOC(%)')
            st.caption('실제 추가 이익이 음수이면 같은 가격 모델의 G2V보다 비용이 큽니다. 이 표는 현재 정답을 본 후향 시연이며 미래 권고가 아닙니다.')
    with st.expander('전향 평가 완료·대기·수집 누락'):
        status=HOME/'prospective_status.json'
        if status.exists(): st.json(json.loads(status.read_text()))
        st.caption('실행 기록 공백은 누락으로 표시합니다. 시스템 로그 없이 종료·절전·작업 실패의 원인을 단정하지 않습니다.')
