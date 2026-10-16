"""Korean EV-first interface: official prices first, archived AI comparisons intact."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import json
from datetime import datetime,time
import numpy as np,pandas as pd,streamlit as st,altair as alt
from v12.ev import EV,baseline,score
from v15.policies import dispatch
from v17.charging import priority_plan,require_prices
from v17.forward import pack
from v18.common import HOME,ROOT,read,write_once,sha,csv,now
from v18.evaluation import reference,inventory,compare_ev

st.set_page_config(page_title='전력예보 V18 · 충전 추천',page_icon='⚡',layout='wide')
st.markdown('<style>.block-container{max-width:1250px;padding-top:1.5rem}div[data-testid="stMetric"]{background:#eef5fa;padding:12px;border-radius:10px}</style>',unsafe_allow_html=True)
st.title('⚡ 전력예보 · 충전 시간 추천')
st.caption('공개 전 저장한 AI 예측과 공개 후 공식 SMP를 비교합니다. 금액은 가상 SMP 연동 정산이며 실제 국내 충전요금·실증 V2G 판매수익이 아닙니다.')

@st.cache_data(ttl=30)
def load():return reference(),inventory()

(official,conflicts),records=load();official=official.rename(columns={'smp':'price'});page=st.sidebar.radio('화면',['충전 추천','예측 검증','연구·수집 상태'])
if st.sidebar.button('화면 자료 새로 고침'):load.clear();st.rerun()

def chart(frame,columns,title,unit):
    melted=frame.melt('timestamp',value_vars=columns,var_name='항목',value_name='값')
    c=alt.Chart(melted).mark_line(point=True).encode(x=alt.X('timestamp:T',title='한국시간'),y=alt.Y('값:Q',title=unit),color=alt.Color('항목:N',title='범례'),tooltip=['timestamp:T','항목:N',alt.Tooltip('값:Q',format='.2f')]).properties(title=title,height=265)
    st.altair_chart(c,width='stretch')

def error_message(exc):
    s=str(exc)
    return {'SOC bounds violated':'초기·목표 SOC와 최소·최대 SOC 범위를 확인하세요.','departure SOC infeasible in connection window':'연결시간과 최대전력으로 목표 SOC를 달성할 수 없습니다.','invalid capacity, power or efficiency':'배터리 용량·충방전 전력·효율을 확인하세요.','valid timezone-aware connection window required':'출발시각은 연결 시작 이후여야 합니다.'}.get(s,s)

if page=='충전 추천':
    mode=st.radio('계획 자료',['현재 자료로 충전 추천','과거 저장 예측 재생'],horizontal=True);record=None;source=None
    if mode=='과거 저장 예측 재생':
        if not records:st.info('저장된 예측이 없습니다.');st.stop()
        choice=st.selectbox('저장 예측',range(len(records)),format_func=lambda i:records[i][3]['primary_target_date']+' · '+records[i][3]['model']+' · V'+str(records[i][3]['version']))
        path,record,forecast,meta=records[choice];source=forecast[['timestamp','predicted_smp']].rename(columns={'predicted_smp':'price'});base=source.timestamp.min().normalize();key='archive_'+sha(path)[:12]
        st.warning('과거 고정 기록 재생입니다. 현재 새 미래예측이나 사전 충전 결정으로 표시하지 않습니다.')
        st.caption(f"실제 생성 {meta['issued_at']} / 입력 기준 {meta['reference_origin']} / 지연 {meta['delay_minutes']:.2f}분 / 모델 고정 적격성 {meta['model_locked_before_origin']}")
    else:
        base=pd.Timestamp.now(tz='Asia/Seoul').normalize();key='current'
        st.info('연결기간의 공식가격이 모두 확보되면 공식가격으로 최적화합니다. 미확보 시 실제 저장된 예측만 사용하며 가격을 임의 생성하지 않습니다.')
    st.subheader('차량 상태와 출발시각')
    a,b,c,d=st.columns(4);capacity=a.number_input('배터리 용량(kWh)',1.,300.,60.);initial=b.number_input('초기 SOC(%)',0.,100.,30.);target=c.number_input('목표 SOC(%)',0.,100.,80.);power=d.number_input('충전전력(kW)',.1,350.,7.)
    a,b,c,d=st.columns(4);begin_day=a.date_input('연결 시작일',base.date(),key=key+'_bd');begin_time=b.time_input('연결 시작시각',time(0) if record else time(18),key=key+'_bt');end_day=c.date_input('출발일',(base+pd.Timedelta(days=1)).date(),key=key+'_ed');end_time=d.time_input('출발시각',time(0) if record else time(8),key=key+'_et')
    with st.expander('효율·SOC·가상 V2G 설정'):
        a,b,c=st.columns(3);eff=a.slider('충방전 각 효율(%)',50,100,90);wear=b.number_input('열화비용(원/배터리 통과kWh)',0.,100.,5.);sell=c.slider('가상 판매 정산비율(%)',0,100,100)
        a,b,c=st.columns(3);minimum=a.number_input('최소 SOC(%)',0.,100.,20.);maximum=b.number_input('최대 SOC(%)',0.,100.,90.);discharge=c.number_input('방전전력(kW)',0.,350.,7.)
    st.caption('입력을 바꾸면 자동으로 계산합니다. 날짜를 넘으면 실제 다음 날짜의 공식가격 또는 같은 기록에 저장된 예측이 필요합니다.')
    try:
        start=pd.Timestamp(datetime.combine(begin_day,begin_time),tz='Asia/Seoul');end=pd.Timestamp(datetime.combine(end_day,end_time),tz='Asia/Seoul');actual=None;basis=''
        if record is None:
            try:source=require_prices(official[['timestamp','price']],start,end);basis='공식 제공가격 기반 최적화 · AI 예측 아님'
            except ValueError:
                found=False
                for path,r,f,m in sorted(records,key=lambda x:x[3]['issued_at'],reverse=True):
                    if m['official_positive_observed_by_issue'] or not m['input_receipts_eligible'] or m['model_locked_before_origin'] is not True:continue
                    try:source=require_prices(f[['timestamp','predicted_smp']].rename(columns={'predicted_smp':'price'}),start,end);record=r;meta=m;found=True;break
                    except ValueError:continue
                if not found:raise ValueError('연결기간 전체의 공식가격이나 저장된 적격 예측이 없습니다. 수신 대기 중이며 다음 날짜 가격을 복제하지 않습니다.')
                basis='기존 저장 AI 예측 기반 계획 · 현재 재계산한 시뮬레이션';st.warning('공식가격이 일부 미확보되어 저장 예측을 사용합니다. 가격오차로 비용이 증가할 수 있습니다.')
        else:source=require_prices(source,start,end);basis='과거 저장 AI 예측 기반 사후 재생 · 새 사전 계획 아님'
        times=pd.DatetimeIndex(source.timestamp);p=source.price.to_numpy();a=official.set_index('timestamp').price.reindex(times).to_numpy();ready=np.isfinite(a).all();ev=EV(capacity,initial,target,power,discharge,eff/100,eff/100,minimum,maximum,wear,sell/100)
        g=priority_plan(p,times,ev=ev,start=start,end=end);v=dispatch('general',p,times,None,ev,start,end);im=baseline(p,times,ev,start,end);un=baseline(p,times,ev,start,end,True)
        st.success(basis);graph=source.rename(columns={'price':'계획 가격'});graph['현재 확보 공식 SMP']=a;chart(graph,['계획 가격','현재 확보 공식 SMP'],'시간별 SMP','원/kWh')
        a1,a2,a3=st.columns(3);a1.metric('추천 종료 SOC',f"{g['ending_soc']:.1f}%");a2.metric('즉시 대비 예상 절감',f"{im['net_cost']-g['net_cost']:.2f}원");a3.metric('균등 대비 예상 절감',f"{un['net_cost']-g['net_cost']:.2f}원")
        costs=[]
        for label,plan in [('즉시충전',im),('균등충전',un),('추천 G2V',g),('가상 V2G',v)]:
            realized=None
            if ready:
                if label=='즉시충전':realized=baseline(a,times,ev,start,end)['net_cost']
                elif label=='균등충전':realized=baseline(a,times,ev,start,end,True)['net_cost']
                else:
                    s=plan['schedule'];realized=score(a,s.charge_kw.to_numpy(),s.discharge_kw.to_numpy(),s.connected_hours.to_numpy(),ev)['net_cost']
            costs.append({'방식':label,'계획가격 비용(원)':plan['net_cost'],'수신 공식가격 평가비용(원)':realized})
        st.dataframe(pd.DataFrame(costs),hide_index=True);s=g['schedule'];vs=v['schedule'];plot=s[['timestamp','charge_kw']].rename(columns={'charge_kw':'G2V 충전'});plot['V2G 충전']=vs.charge_kw;plot['V2G 방전']=-vs.discharge_kw;chart(plot,['G2V 충전','V2G 충전','V2G 방전'],'추천 충방전 전력','kW');soc=s[['timestamp','soc_end']].rename(columns={'soc_end':'G2V SOC'});soc['V2G SOC']=vs.soc_end;chart(soc,['G2V SOC','V2G SOC'],'출발까지 SOC','%')
        selected=s[s.charge_kw>1e-8];st.write('추천 충전시간: '+', '.join(t.strftime('%m/%d %H:%M') for t in selected.timestamp));st.caption('연결시간·전력·효율·목표 SOC를 만족하면서 낮은 계획가격 시간부터 충전합니다. 일부 시간만 연결되면 실제 연결 분수를 반영합니다.')
        st.dataframe(s[['timestamp','connected_hours','charge_kw','soc_end']].rename(columns={'timestamp':'한국시간','connected_hours':'연결시간(h)','charge_kw':'충전전력(kW)','soc_end':'종료 SOC(%)'}),hide_index=True)
        if ready:
            replay=compare_ev(p,a,times,ev,start,end);st.write(f"공식가격 최적 G2V 비용 {replay['official_G2V_optimum']:.2f}원 · 현재 계획 regret {replay['G2V_regret']:.2f}원 · 가상 V2G 추가손익 {replay['V2G_actual_gain']:.2f}원 · 방전량 {replay['V2G_export']:.2f}kWh")
        st.warning('기본 권장: G2V. V2G는 효율·열화비용·가격오차 때문에 추가 손실이 날 수 있습니다. 공식가격을 알더라도 가상 정산 결과이며 미래 수익을 보장하지 않습니다.')
        if st.button('이 시뮬레이션 결과 로컬 저장'):
            write_once(HOME/'interactive_decisions'/(pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%f')+'.json'),dict(at=now(),mode='INTERACTIVE_SIMULATION_NOT_PRIOR_FORECAST',basis=basis,start=start.isoformat(),end=end.isoformat(),EV=ev.__dict__,G2V=pack(g),V2G=pack(v),costs=costs));st.success('로컬 시뮬레이션으로 저장했습니다. 실제 사전 예측 표본에는 포함하지 않습니다.')
    except ValueError as exc:st.error(error_message(exc))

elif page=='예측 검증':
    st.subheader('저장 당시 AI 예측과 이후 공식가격 비교')
    status=read(HOME/'status.json',{});a,b,c,d=st.columns(4);a.metric('독립 기준시각',status.get('independent_reference_origins',0));b.metric('독립 기본 대상일',status.get('independent_primary_target_days',0));c.metric('공식가격 비교 완료일',status.get('official_price_comparison_complete_days',0));d.metric('대상일 경과 완료',status.get('target_day_elapsed_days',0))
    st.caption(f"정시 적격 기록 {status.get('strict_on_time_eligible_records',0)}개 · 지연 기록 {status.get('late_forecast_records',0)}개 · 급등 대상일 {status.get('spike_target_days',0)}일 · 최종 정정통계 확인 {status.get('final_corrected_statistics_verified_days',0)}일. 모델 수나 보조 D3 날짜를 독립 원점 수로 늘리지 않습니다.")
    if records:
        choice=st.selectbox('비교할 원본 예측',range(len(records)),format_func=lambda i:records[i][0].name);path,r,f,m=records[choice];graph=f[['timestamp','predicted_smp']].rename(columns={'predicted_smp':'원본 AI 예측'});graph['현재 공식 SMP']=official.set_index('timestamp').price.reindex(graph.timestamp).to_numpy();chart(graph,['원본 AI 예측','현재 공식 SMP'],'원본을 바꾸지 않은 가격 비교','원/kWh');st.json(m)
    if (HOME/'comparison_index.csv').exists():
        f=csv(HOME/'comparison_index.csv');st.write('동일 EV 조건으로 저장 가격벡터를 재생한 비교 · 원래 사전 결정과 구분');st.dataframe(f[['version','model','target_date','auxiliary_target','MAE','peak_MAE','peak_F1','cheap5_overlap','G2V_actual','official_G2V_optimum','G2V_regret','V2G_actual_gain','V2G_export']].rename(columns={'version':'버전','model':'모델','target_date':'대상일','auxiliary_target':'보조 날짜','peak_MAE':'급등 MAE','cheap5_overlap':'저가5시간 일치율','G2V_actual':'예측계획 공식가격 비용','official_G2V_optimum':'공식가격 최적비용','V2G_actual_gain':'V2G 추가손익','V2G_export':'방전kWh'}),hide_index=True)
    if (HOME/'frozen_decision_index.csv').exists():
        st.write('발급 당시 또는 이후 별도로 동결한 실제 저장 계획의 평가');st.dataframe(csv(HOME/'frozen_decision_index.csv'),hide_index=True)
    st.info('아직 독립 표본과 급등 사례가 부족합니다. 공식가격 수신은 대상일 경과나 최종 정정통계 확정을 뜻하지 않습니다. 이미 공개된 날짜에 새 사전 예측을 소급 생성하지 않습니다.')

else:
    st.subheader('공개시각 관측과 수집 상태');st.write('회원 통지시각·일반 공개시각·API 수신시각을 구분합니다. 최초 수신은 그 시각까지 확인됐다는 증거이며 정확한 최초 공개시각은 아닙니다.')
    rows=read(HOME/'publication/summary.json',[])
    if rows:st.dataframe(pd.DataFrame(rows).drop(columns=['counts'],errors='ignore'),hide_index=True)
    st.json(read(HOME/'service_status.json',{}));st.caption('노트북 종료·절전·네트워크 오류 중 원인을 확인하지 못한 공백은 누락으로 남깁니다. 빈 응답·실패·0값 자리표시자를 가격 미공개로 바꾸지 않습니다.')
    st.subheader('예측과 경제성의 한계');st.write('V17 가격·순위 혼합은 검증 기준을 통과하지 못했습니다. V18은 추가 학습이나 운영 모델 교체 없이 기존 예측의 평가와 수신 검증을 통합했습니다. 기상 관측을 미래 예보로 사용하지 않습니다.')
    st.write('가상 V2G 손익분기: 판매 정산비율×방전가격 > 충전가격/(충전효율×방전효율) + 2×열화단가/방전효율. 이 조건도 잘못된 가격예측의 손실을 없애지는 못합니다.')
    for image in ['docs/v15_figures/known_October_failure.png','docs/v17_figures/connection_regret.png']:
        if (ROOT/image).exists():st.image(str(ROOT/image),caption='이미 열람한 과거 진단 · 새 전향 성능 입증 자료가 아님')
