"""Same target D3/D2/official price and EV comparison; never invent missing curves."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
from datetime import datetime,time
import pandas as pd,numpy as np,streamlit as st,altair as alt
from v12.ev import EV,baseline,score
from v15.policies import dispatch
from v17.charging import priority_plan,require_prices
from v17.forward import pack
from v18.evaluation import reference,compare_ev
from v19.evaluation import inventory
from v19.common import HOME,ROOT,read,write_once,sha,csv,now

st.set_page_config(page_title='전력예보 V19 · D-3/D-2 검증',page_icon='⚡',layout='wide')
st.title('⚡ 전력예보 V19 · 예측과 충전비용 검증')
st.caption('육지 SMP · KST. 저장된 AI와 이후 수신한 공식가격을 비교합니다. 가상 SMP 정산이며 실제 충전요금·실증 V2G 수익이 아닙니다.')
page=st.sidebar.radio('화면',['같은 날짜 D-3·D-2 비교','차량 충전 추천','연구·발급 현황'])

@st.cache_data(ttl=20)
def load():return inventory(),reference()[0].rename(columns={'smp':'price'})
items,official=load();status=read(HOME/'status.json',{});a,b,c=st.columns(3);a.metric('독립 기준시각',status.get('independent_origins',0));b.metric('독립 기본 대상일',status.get('independent_primary_target_days',0));c.metric('V19 실제 정시 적격 기록',status.get('V19_on_time_eligible',0))
if st.sidebar.button('저장 자료 새로 고침'):load.clear();st.rerun()

def line(frame,cols,title,unit):
    f=frame.melt('timestamp',value_vars=cols,var_name='항목',value_name='값');chart=alt.Chart(f).mark_line(point=True).encode(x=alt.X('timestamp:T',title='한국시간'),y=alt.Y('값:Q',title=unit),color='항목:N',tooltip=['timestamp:T','항목:N',alt.Tooltip('값:Q',format='.2f')]).properties(title=title,height=270);st.altair_chart(chart,width='stretch')

def conditions(prefix,day,overnight=False):
    a,b,c,d=st.columns(4);cap=a.number_input('배터리 용량(kWh)',1.,300.,60.,key=prefix+'cap');initial=b.number_input('초기 SOC(%)',0.,100.,30.,key=prefix+'initial');target=c.number_input('목표 SOC(%)',0.,100.,80.,key=prefix+'target');power=d.number_input('충전전력(kW)',.1,350.,7.,key=prefix+'power')
    base=pd.Timestamp(day);a,b,c,d=st.columns(4);bd=a.date_input('연결 시작일',base.date(),key=prefix+'bd');bt=b.time_input('연결 시작시각',time(18) if overnight else time(0),key=prefix+'bt');ed=c.date_input('출발일',(base+pd.Timedelta(days=1)).date(),key=prefix+'ed');et=d.time_input('출발시각',time(8) if overnight else time(0),key=prefix+'et')
    with st.expander('효율·SOC·가상 V2G 가정'):
        a,b,c=st.columns(3);eff=a.slider('충방전 각 효율(%)',50,100,90,key=prefix+'eff');wear=b.number_input('열화비용(원/배터리 통과kWh)',0.,100.,5.,key=prefix+'wear');sell=c.slider('가상 판매 정산비율(%)',0,100,100,key=prefix+'sell')
        a,b,c=st.columns(3);minimum=a.number_input('최소 SOC(%)',0.,100.,20.,key=prefix+'min');maximum=b.number_input('최대 SOC(%)',0.,100.,90.,key=prefix+'max');discharge=c.number_input('방전전력(kW)',0.,350.,7.,key=prefix+'dis')
    return EV(cap,initial,target,power,discharge,eff/100,eff/100,minimum,maximum,wear,sell/100),pd.Timestamp(datetime.combine(bd,bt),tz='Asia/Seoul'),pd.Timestamp(datetime.combine(ed,et),tz='Asia/Seoul')

def message(e):return {'SOC bounds violated':'초기·목표 SOC와 배터리 상하한을 확인하세요.','departure SOC infeasible in connection window':'연결시간과 전력으로 목표 SOC를 달성할 수 없습니다.'}.get(str(e),str(e))

def show_plans(sources,ev,start,end):
    actual=None
    try:actual=require_prices(official[['timestamp','price']],start,end).price.to_numpy()
    except ValueError:pass
    costs=[];powers=[];soc=[];saved={}
    for label,f in sources.items():
        if f is None:st.info(label+': 미확보 · 다른 날짜나 모델 값으로 대체하지 않습니다.');continue
        try:
            selected=require_prices(f[['timestamp','price']],start,end);times=pd.DatetimeIndex(selected.timestamp);p=selected.price.to_numpy();g=priority_plan(p,times,ev=ev,start=start,end=end);v=dispatch('general',p,times,None,ev,start,end);saved[label]=dict(G2V=pack(g),V2G=pack(v))
            for kind,plan in [('G2V',g),('가상 V2G',v)]:
                actual_cost=regret=None
                if actual is not None:
                    s=plan['schedule'];actual_cost=score(actual,s.charge_kw.to_numpy(),s.discharge_kw.to_numpy(),s.connected_hours.to_numpy(),ev)['net_cost'];oracle=priority_plan(actual,times,ev=ev,start=start,end=end) if kind=='G2V' else dispatch('general',actual,times,None,ev,start,end);regret=actual_cost-oracle['net_cost']
                costs.append({'자료':label,'방식':kind,'계획가격 비용(원)':plan['net_cost'],'공식가격 평가비용(원)':actual_cost,'regret(원)':regret,'방전(kWh)':plan['grid_export_kwh'],'종료 SOC(%)':plan['ending_soc']})
                s=plan['schedule'];powers.append(pd.DataFrame(dict(timestamp=s.timestamp,계열=label+' '+kind,전력=s.charge_kw-s.discharge_kw)));soc.append(pd.DataFrame(dict(timestamp=s.timestamp,계열=label+' '+kind,SOC=s.soc_end)))
            if label=='공식 SMP':
                for kind,uniform in [('즉시충전',False),('균등충전',True)]:
                    cost=baseline(p,times,ev,start,end,uniform)['net_cost'];costs.append({'자료':label,'방식':kind,'계획가격 비용(원)':cost,'공식가격 평가비용(원)':cost,'regret(원)':cost-g['net_cost'],'방전(kWh)':0,'종료 SOC(%)':g['ending_soc']})
            st.caption(label+' G2V 추천시간: '+', '.join(t.strftime('%m/%d %H:%M') for t in g['schedule'].loc[g['schedule'].charge_kw>1e-8,'timestamp']))
        except ValueError as exc:st.warning(label+': '+message(exc))
    if costs:
        table=pd.DataFrame(costs);st.dataframe(table,hide_index=True)
        if actual is not None:
            for label in sources:
                a=table[(table['자료']==label)&(table['방식']=='G2V')];b=table[(table['자료']==label)&(table['방식']=='가상 V2G')]
                if len(a) and len(b):st.write(f"{label} 가상 V2G 추가손익: {a.iloc[0]['공식가격 평가비용(원)']-b.iloc[0]['공식가격 평가비용(원)']:.2f}원 · 방전 {b.iloc[0]['방전(kWh)']:.2f}kWh")
        for data,value,title,unit in [(powers,'전력','충방전 전력','kW'),(soc,'SOC','배터리 SOC','%')]:
            f=pd.concat(data);chart=alt.Chart(f).mark_line(point=True).encode(x=alt.X('timestamp:T',title='KST'),y=alt.Y(value+':Q',title=unit),color='계열:N',tooltip=['timestamp:T','계열:N',value+':Q']).properties(title=title,height=265);st.altair_chart(chart,width='stretch')
        st.warning('기본 권장 G2V 유지. 화면에서 다시 계산한 계획은 현재 조건의 시뮬레이션이며 원래 사전 계획을 수정하지 않습니다. 공식 제공가격과 최종 정정통계도 구분합니다.')
        if st.button('비교 시뮬레이션 로컬 저장'):
            write_once(HOME/'interactive_decisions'/(pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%f')+'.json'),dict(at=now(),mode='RETROSPECTIVE_INTERACTIVE_NOT_NEW_FORWARD',EV=ev.__dict__,start=start.isoformat(),end=end.isoformat(),plans=saved,costs=costs));st.success('로컬 시뮬레이션으로 저장했습니다. 새 사전 예측 수에는 포함하지 않습니다.')

def select_forecast(day,horizon):
    choices=[x for x in items if day in set(x[2].timestamp.dt.strftime('%Y-%m-%d')) and (pd.Timestamp(day).date()-pd.Timestamp(x[3]['reference_origin']).date()).days==horizon]
    if not choices:return None,None
    n=st.selectbox('D-'+str(horizon)+' 저장 모델',range(len(choices)),format_func=lambda i:choices[i][3]['model']+' · '+choices[i][3]['issued_at'],key='D'+str(horizon));path,r,f,m=choices[n];frame=f[['timestamp','predicted_smp']].rename(columns={'predicted_smp':'price'}).copy()
    # Cross-midnight only from the SAME model hash and SAME origin; no cloning.
    if len(frame)==24:
        for p2,r2,f2,m2 in items:
            if p2==path:continue
            if m2['reference_origin']==m['reference_origin'] and r2.get('model_sha256',r2.get('model_version'))==r.get('model_sha256',r.get('model_version')) and m2['model']==m['model']:
                addition=f2[['timestamp','predicted_smp']].rename(columns={'predicted_smp':'price'});frame=pd.concat([frame,addition]).drop_duplicates('timestamp').sort_values('timestamp')
    return frame,m

if page=='같은 날짜 D-3·D-2 비교':
    dates=sorted({d for _,_,f,_ in items for d in f.timestamp.dt.strftime('%Y-%m-%d')});default=dates.index('2026-10-18') if '2026-10-18' in dates else max(0,len(dates)-1);day=st.selectbox('동일 비교 대상일',dates,index=default);d3,m3=select_forecast(day,3);d2,m2=select_forecast(day,2)
    for h,m in [(3,m3),(2,m2)]:
        if m:st.caption(f"D-{h} 기준 {m['reference_origin']} / 실제 발급 {m['issued_at']} / {m['timing']} / 모델 고정 확인 {m['model_locked_before_origin']}")
        else:st.info(f'D-{h} 실제 저장 예측 미확보 · 과거 발급을 새로 만들지 않습니다.')
    times=pd.date_range(pd.Timestamp(day,tz='Asia/Seoul'),periods=24,freq='h');graph=pd.DataFrame(dict(timestamp=times));graph['공식 SMP']=official.set_index('timestamp').price.reindex(times).to_numpy()
    for label,frame in [('D-3 AI',d3),('D-2 AI',d2)]:graph[label]=np.nan if frame is None else frame.set_index('timestamp').price.reindex(times).to_numpy()
    st.write('공식가격 확보: '+str(int(graph['공식 SMP'].notna().sum()))+'/24시간 · 최종 정정통계 확인 대기');line(graph,['D-3 AI','D-2 AI','공식 SMP'],'같은 날짜의 두 예측시점과 공식가격','원/kWh')
    if (HOME/'comparison.csv').exists():
        f=csv(HOME/'comparison.csv');g=f[f.target_date==day];st.dataframe(g[['model','horizon_days','reference_origin','issued_at','comparison_group','MAE','RMSE','peak_MAE','peak_F1','cheap5_overlap','G2V_regret','V2G_actual_gain']].rename(columns={'model':'모델','horizon_days':'D-몇일','cheap5_overlap':'저가5시간 일치율','V2G_actual_gain':'가상 V2G 추가손익'}),hide_index=True)
    st.subheader('같은 차량 조건으로 가격별 계획 재생');ev,start,end=conditions('compare_'+day,day);show_plans({'D-3 AI':d3,'D-2 AI':d2,'공식 SMP':official},ev,start,end)
    if (HOME/'frozen_costs.csv').exists():
        with st.expander('원래 동결한 사전 계획의 평가 · 위 재생과 별도'):
            f=csv(HOME/'frozen_costs.csv');st.dataframe(f[f.target_date==day],hide_index=True)
elif page=='차량 충전 추천':
    st.info('공식가격이 모두 확보되면 공식가격 기반 G2V를 우선합니다. 과거 저장 AI를 실제 실시간 새 예측으로 표시하지 않습니다.');ev,start,end=conditions('live',pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y-%m-%d'),True);show_plans({'공식 SMP':official},ev,start,end)
else:
    st.subheader('정시 발급 준비와 실제 발급');st.json(read(HOME/'publish_status.json',{}));st.write('기준시각 전 준비 상태 · 발급 성공 기록 아님');st.json(read(HOME/'preparation_status.json',{}));st.json(read(HOME/'observer_status.json',{}))
    st.info('정시23시의 실제 기록은 그 시각이 도래한 후 확인합니다. API·이력조회·모델 추론·EV 최적화는 준비 작업에서 수행하며 발급 경로에는 없습니다. 공개상태가 불확실하면 적격 사전 예측으로 발급하지 않습니다.')
    path=ROOT/'docs/V19_D3_HISTORICAL_COMPARISON.csv'
    if path.exists():st.write('이미 열람된2025Q4의 소규모 D3 연구 · 실제 전향 아님');st.dataframe(csv(path),hide_index=True)
    st.json(status);st.caption('D-3이 모든 공식 경로보다 이르다고 보장하지 않습니다. 노트북 종료·절전은 실행 누락으로 기록합니다. 순위 점수는 가격예측과 구분하며 운영 모델은 교체하지 않았습니다.')
