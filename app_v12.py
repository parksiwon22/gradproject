"""Local-only V12 forward validation and hypothetical EV dispatch screen."""
import json
from datetime import datetime,time,timedelta
import pandas as pd
import streamlit as st
from v11.store import connect,latest,HOME as V11
from v12.research import HOME
from v12.ev import EV,optimize,baseline,regret

st.set_page_config(page_title='전력예보 V12 · EV 연구',layout='wide')
st.title('전력예보 V12 · 미래 검증 및 EV 충방전')
st.caption('로컬 실험 화면 · 가상 SMP 연동 정산. 국내 소매 충전요금·실제 V2G 판매수익이 아닙니다.')
def read(path,default):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default
db=connect(); raw=latest(db,'land'); jeju=latest(db,'jeju'); db.close()
st.write('육지 공식가격 최근 대상:',str(raw.timestamp.max()) if len(raw) else '없음')
st.caption('제주 하루전/실시간 상품은 육지 시간가격과 구분합니다. 제주 시범시장 가격으로 자동 대체하지 않습니다.')
with st.expander('공표·수신 이력 및 기존 수집 상태'):
    st.dataframe(pd.DataFrame(read(HOME/'publication/summary.json',[])),hide_index=True)
    st.write('V11 수집',read(V11/'collection_status.json',{}).get('failures',[]))
    st.write('ASOS 자동 수집',read(V11/'kma_apihub/refresh_status.json',{}))
    st.caption('first_seen은 로컬 최초 수신시각입니다. 회원 공개시각·일반 공개 최초시각·API 갱신시각을 동일시하지 않습니다.')
records=[]
for path in sorted((HOME/'forecasts').glob('*.json'),reverse=True):
    r=read(path,{})
    if r.get('status')=='created': records.append((path,r))
if not records:
    st.info('전향 예측이 없습니다. python -m v12.forward 실행 후 다시 열어주세요.'); st.stop()
choice=st.selectbox('저장된 전향 예측',range(len(records)),format_func=lambda i:records[i][1]['target_date']+' · '+records[i][1]['model']+' · '+records[i][1]['issued_at'])
file,r=records[choice]; f=pd.DataFrame(r['rows']); f.timestamp=pd.to_datetime(f.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
f=f.merge(raw[['timestamp','smp']],on='timestamp',how='left')
st.write('실제 생성:',r['issued_at'],' · 입력 기준:',r['reference_origin'],' · 모델:',r['model'])
if r['official_smp_known_at_issue']: st.info('생성 당시 공식가격 기확보: 미래 AI 성과가 아닌 진단 기록입니다.')
else: st.warning('생성 당시 확인한 출처에서 대상가격을 확보하지 못했습니다. 모든 경로에서 미공개임을 인증한 것은 아닙니다.')
st.caption('D-2 23시 입력 기준의 별도 연구 모델입니다. V11 D-1 23시 및 기존 운영 모델은 보존합니다. 기상예보·수요예측 사용: 아니요.')
st.line_chart(f.set_index('timestamp')[['predicted_smp','smp']].rename(columns={'predicted_smp':'저장된 사전 추정','smp':'확보된 공식가격'}))
st.line_chart(f.set_index('timestamp')[['spike_probability']])
st.caption('급등 위험은 별도 분류·확률 보정 연구입니다. 현재 급등 탐지와 확률의 미래 성능이 검증되지 않아 가격 예측의 정확도 보증으로 사용하지 않습니다.')
evaluations=read(HOME/'evaluation_index.json',[])
st.write('사후 평가',pd.DataFrame([{k:v for k,v in e.items() if k not in ('pairs',)} for e in evaluations]))
st.subheader('EV 충전 조건')
source=st.radio('의사결정 가격','공식가격 우선, 미확보 시간은 선택한 예측 사용|저장된 예측만 사용(연구)'.split('|'))
day=raw.timestamp.max().normalize() if source.startswith('공식') else f.timestamp.min().normalize()
startday=st.date_input('연결 시작일',day.date()); starttime=st.time_input('연결 시작시각',time(18,0))
endday=st.date_input('연결 종료일',day.date()+timedelta(days=1)); endtime=st.time_input('연결 종료시각',time(8,0))
c1,c2,c3=st.columns(3)
capacity=c1.number_input('배터리 용량(kWh)',10.,200.,60.)
initial=c2.number_input('초기 SOC(%)',0.,100.,30.)
target=c3.number_input('출발 최소/목표 SOC(%)',0.,100.,80.)
c1,c2,c3=st.columns(3)
charge=c1.number_input('최대 충전전력(kW)',.1,100.,7.)
discharge=c2.number_input('최대 방전전력(kW)',0.,100.,7.)
eff=c3.number_input('충전효율(%)',1.,100.,90.)
c1,c2,c3=st.columns(3)
deff=c1.number_input('방전효율(%)',1.,100.,90.)
minimum=c2.number_input('최소 SOC(%)',0.,100.,20.)
maximum=c3.number_input('최대 SOC(%)',0.,100.,90.)
wear=st.number_input('배터리 처리 에너지당 가상 열화비(원/kWh)',0.,1000.,5.)
ratio=st.slider('가상 판매 정산/SMP 비율',0.,1.,1.)
if st.button('G2V 및 가상 V2G 최적화'):
    try:
        begin=pd.Timestamp(datetime.combine(startday,starttime),tz='Asia/Seoul'); end=pd.Timestamp(datetime.combine(endday,endtime),tz='Asia/Seoul')
        times=pd.date_range(begin.floor('h'),end.ceil('h')-pd.Timedelta(hours=1),freq='h')
        forecastparts=[]
        for _,record in records:
            if record['model']==r['model']:
                part=pd.DataFrame(record['rows']); part.timestamp=pd.to_datetime(part.timestamp,utc=True).dt.tz_convert('Asia/Seoul'); forecastparts.append(part)
        pred=pd.concat(forecastparts).drop_duplicates('timestamp',keep='first').set_index('timestamp').predicted_smp.reindex(times)
        official=raw.set_index('timestamp').smp.reindex(times)
        price=official.combine_first(pred) if source.startswith('공식') else pred
        if price.isna().any(): raise ValueError('연결시간 전체를 덮는 공식/저장 예측가격이 없습니다. 연결시간을 줄이거나 다음 예측을 기다려주세요.')
        ev=EV(capacity,initial,target,charge,discharge,eff/100,deff/100,minimum,maximum,wear,ratio)
        g=optimize(price,times,ev,begin,end); v=optimize(price,times,ev,begin,end,True)
        immediate=baseline(price,times,ev,begin,end); uniform=baseline(price,times,ev,begin,end,True)
        st.dataframe(pd.DataFrame([dict(plan=name,purchase_cost=x['purchase_cost'],export_credit=x['export_credit'],degradation_cost=x['degradation_cost'],net_cost=x['net_cost'],saving_vs_immediate=immediate['net_cost']-x['net_cost']) for name,x in [('즉시충전',immediate),('균등충전',uniform),('G2V',g),('가상 V2G',v)]]),hide_index=True)
        st.write('G2V 대비 가상 V2G 순비용 절감',g['net_cost']-v['net_cost'])
        st.write('G2V 종료 SOC',g['ending_soc'],' · V2G 종료 SOC',v['ending_soc'])
        for title,plan in [('G2V',g),('가상 V2G',v)]:
            st.write(title); schedule=plan['schedule'].copy(); schedule['price_source']=['official' if source.startswith('공식') and pd.notna(x) else 'saved_prediction' for x in official]
            st.line_chart(schedule.set_index('timestamp')[['charge_kw','discharge_kw']]); st.line_chart(schedule.set_index('timestamp')[['soc_start','soc_end']]); st.dataframe(schedule,hide_index=True)
        if official.notna().all():
            st.write('실제 공식가격 기준 regret',{'G2V':regret(g,official,times,ev,begin,end),'V2G':regret(v,official,times,ev,begin,end,True)})
        else: st.info('아직 실제가격이 없는 시간이 있어 사후 regret 평가를 완료하지 않았습니다.')
        from v12.decisions import save_interactive
        saved=save_interactive({'G2V':g,'V2G':v},dict(source_forecast=file.name,price_policy=source,start=begin.isoformat(),end=end.isoformat(),prices=[dict(timestamp=t.isoformat(),value=float(value),source='official' if source.startswith('공식') and pd.notna(known) else 'saved_prediction') for t,value,known in zip(times,price,official)]))
        st.caption('이 충방전 의사결정은 로컬에 불변 저장했습니다: '+saved.name+' · 실제가격 수신 후 자동 사후 평가 대상입니다.')
        st.caption('비용에는 설정한 충방전 손실·열화 가정이 포함됩니다. V2G는 동시 충방전을 금지한 MILP이며 망요금·세금·실물 양방향 충전기는 포함하지 않습니다.')
    except ValueError as exc: st.error(str(exc))
with st.expander('저장된 사용자 충방전 의사결정의 사후 평가'):
    history=[e for e in read(HOME/'decision_evaluation_index.json',[]) if e.get('cohort')=='interactive_decisions']
    if history: st.dataframe(pd.DataFrame(history),hide_index=True)
    else: st.caption('평가 기록이 없거나 다음 자동 평가 실행을 기다리는 중입니다.')
