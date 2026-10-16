"""Korean decision explanations, event definitions and physical break-even."""
import json
import numpy as np,pandas as pd,streamlit as st
from v16.common import HOME,ROOT,csv
from v16.economics import cycle_margin

def render(official,chart):
    st.subheader('가격 수준·급격한 변화·EV 회피 시간은 다른 목표입니다')
    st.caption('150원 이상은 가격 수준, 알려진 가격 대비20원 상승은 예측하기 어려운 일간 변화, ramp는 연속 시간 변화, 일중 상위6시간은 상대적인 회피 대상입니다. 학습 구간에서 정한 ramp 임계값을 검증에 적용합니다.')
    path=ROOT/'docs/V16_DEFINITION_SCORES.csv'
    if path.exists():
        data=csv(path);fold=st.selectbox('V16 시간순 구간',data.fold.unique());model=st.selectbox('V16 정의 비교 모델',data.model.unique())
        g=data[(data.fold==fold)&(data.model==model)]
        st.dataframe(g[['definition','actual_event_hours','event_MAE','event_bias','precision','recall','F1','false_alarm_hours']].rename(columns={'definition':'급등 정의','actual_event_hours':'실제 해당 시간','event_MAE':'해당 MAE','event_bias':'Bias','false_alarm_hours':'허위 경보 시간'}),hide_index=True)
    st.subheader('새 공식 공급 정보 · 당시 발표본 여부')
    inv=ROOT/'docs/V16_SOURCE_AVAILABILITY.json'
    if inv.exists():st.dataframe(pd.DataFrame(json.loads(inv.read_text(encoding='utf-8'))),hide_index=True)
    maintenance=ROOT/'docs/V16_MAINTENANCE_ABLATION.csv'
    if maintenance.exists():
        st.info('주간 예방정비계획은 새로 확보한 공식 자료입니다. 게시시각은 있지만 현재 첨부와 당시 최초 첨부의 일치 여부는 미확인입니다. 아래는 운영 불가한 후향 실험이며 실제 미래 개선이라고 표시하지 않습니다.')
        m=csv(maintenance);st.dataframe(m[['fold','model','validation_days','MAE','peak_MAE','peak_F1','peak_hour_MAE','G2V_regret','V2G_regret']],hide_index=True)
        files=sorted((HOME/'maintenance_study').glob('*_predictions.csv'))
        if files:
            which=st.selectbox('정비계획 모델 시연 구간',files,format_func=lambda p:p.stem)
            f=csv(which);f.timestamp=pd.to_datetime(f.timestamp,utc=True).dt.tz_convert('Asia/Seoul');days=f.groupby('target_date').actual_smp.max().sort_values(ascending=False).index.tolist();day=st.selectbox('실제값 비교일(과거 시연)',days)
            chart(f[f.target_date==day],['price_only_common','price_plus_maintenance'])
    st.subheader('손익분기 조건 · 보상 충전을 포함한 방전1kWh')
    a,b,c=st.columns(3)
    buy=a.number_input('보상 충전 가격(가상 원/kWh)',value=100.0,min_value=-500.0,max_value=1000.0)
    sell=b.number_input('방전 시간 가격(가상 원/kWh)',value=150.0,min_value=-500.0,max_value=1000.0)
    eff=c.slider('각 방향 효율(%)',50,100,90)
    a,b=st.columns(2);wear=a.number_input('열화비용(원/배터리 통과kWh)',value=5.0,min_value=0.0,max_value=100.0);ratio=b.slider('가상 판매 정산비율(%)',0,100,100)
    if st.button('V16 손익분기와 권장 근거 계산'):
        r=cycle_margin(buy,sell,eff/100,eff/100,wear,ratio/100)
        st.metric('방전1kWh의 가상 추가 마진',f"{r['margin_per_export_kwh']:.2f}원")
        st.write('필요한 방전시간 가격:',f"{r['break_even_sell_price']:.2f}원/kWh" if r['break_even_sell_price'] is not None else '판매 정산이0이므로 가격 임계값으로 판단할 수 없음')
        st.warning('기본 권장: G2V. 이 계산은 한 번의 보상 충전·방전 손익조건이며 SOC·연결시간·전력 한계와 가격 예측오차를 검증한 미래 수익이 아닙니다. 양의 마진만으로 V2G를 자동 승격하지 않습니다.')
    st.latex(r'sP_{sell}>\frac{P_{buy}}{\eta_c\eta_d}+\frac{2d}{\eta_d}')
    st.caption('높고 안정적인 가격은 효율·열화 때문에 방전 이익이 없을 수 있고,150원 미만에서도 충분한 일중 가격차가 있으면 가상 이익 조건을 만족할 수 있습니다. 음수 가격과 판매비율0은 부등식 자체로 판단합니다.')
    path=ROOT/'docs/V16_SENSITIVITY.csv'
    if path.exists():
        f=csv(path);cohort=st.selectbox('민감도 구간',['selection','audit'],format_func=lambda x:'2025Q4 검증' if x=='selection' else '2026H1 이미 열람한 진단')
        g=f[f.cohort==cohort];st.dataframe(g.rename(columns={'efficiency':'각 방향 효율','wear':'열화 단가','sell_ratio':'판매 정산비율','mean_gain':'G2V 대비 실제 가상 이익/일','mean_export':'평균 방전kWh/일','loss_days':'추가 손실일'}),hide_index=True)
        st.caption('각 가정에서 동일 출발SOC·연결조건의 충전 전용과 일반V2G를 다시 최적화합니다. 가정을 바꿔 유리해진 후향 결과를 현실 요금 또는 운영 정책의 검증 성공으로 해석하지 않습니다.')
    path=ROOT/'docs/V16_INFORMATION_ORACLES.csv'
    if path.exists():
        st.subheader('어떤 예측 정보가 EV 비용에 중요한가')
        st.dataframe(csv(path),hide_index=True)
        st.warning('perfect로 시작하는 행은 실제 정답을 주입한 정보가치 상한 실험입니다. 학습 모델이나 실행 가능한 미래예측이 아닙니다. 시간대 순서를 정확히 아는 가치와 하루 평균가격을 정확히 아는 가치를 분리하기 위한 진단입니다.')
    path=ROOT/'docs/V16_HORIZON_COMPARISON.csv'
    if path.exists():
        st.subheader('같은 HGB 설정 · D-1과 D-2의 차이')
        st.dataframe(csv(path)[['cohort','days_ahead','MAE','peak_MAE','peak_F1','peak_hour_MAE','G2V_regret','V2G_regret']],hide_index=True)
        st.caption('학습 종료일2024-12-31과 하이퍼파라미터를 맞춘 기존 모델 재사용입니다. 최신 정정값의 과거 event-time 비교이며 당시 공개 전 예측 가능성을 증명하지 않습니다. D-1 가격이 이미 알려져 있었다면 공식가격 활용 모드입니다.')
    status=HOME/'prospective_status.json'
    if status.exists():
        r=json.loads(status.read_text());a,b,c=st.columns(3);a.metric('전향 독립 대상일',r['independent_target_days']);b.metric('공식가격 대조 / 날짜 경과 완료',f"{r['official_price_comparison_complete_days']} / {r['elapsed_evaluated_days']}");c.metric('전향 급등 발생일',r['spike_days'])
        st.caption('V16 신규 예측0개 · 발표가격 수신과 대상일 경과를 구분 · 수집 실패/절전 공백을 정상 실행으로 보충하지 않습니다.')
