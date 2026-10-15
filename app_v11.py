"""Separate local V11 application; original app.py is preserved."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from v11.store import HOME,connect,latest

st.set_page_config(page_title='전력예보 V11',layout='wide')
st.title('전력예보 V11 · 최신 공식가격과 예측')
st.caption('육지 SMP · 개인 충전요금과 다른 가상 SMP 연동 요금 · 기존 V5 웹과 별도 실행')

def read(path,default):
    try: return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,json.JSONDecodeError): return default

db=connect(); actual=latest(db,'land'); state=dict(db.execute('SELECT key,value FROM state')); failures=pd.read_sql_query("SELECT * FROM attempts WHERE status='failure' ORDER BY at DESC LIMIT 3",db); db.close()
if actual.empty:
    st.warning('최신 공식 데이터가 없습니다. 로컬 수집 프로그램을 먼저 실행하세요.')
    st.stop()
left,right=st.columns(2)
left.metric('확보한 공식가격의 마지막 구간',actual.timestamp.max().strftime('%Y-%m-%d %H:%M KST'))
right.write('마지막 수집 성공: '+state.get('last_success_land','확인되지 않음'))
st.caption('미래 거래일의 공식가격도 사전에 공개될 수 있습니다. 거래일과 관측시각·수집시각을 구분합니다.')
status=read(HOME/'pipeline_status.json',{})
if status:
    if any(s['exit_code'] for s in status['stages']): st.warning('최근 갱신 중 실패한 단계가 있습니다. 기존 데이터를 표시합니다.')
    else: st.success('최근 수집·예측 비교 갱신 정상 · '+status['at'])
collection=read(HOME/'collection_status.json',{})
if collection.get('failures'): st.warning('공식 조회 실패가 기록돼 있습니다. 마지막 성공 자료를 보존했습니다.')

files=sorted((HOME/'forecasts').glob('*.json'),reverse=True)
if not files:
    st.warning('예측 파일이 아직 없습니다. 학습된 모델과 완전한 입력 이력 및 공개 여부 확인이 필요합니다.')
    st.line_chart(actual.tail(168).set_index('timestamp').smp)
    st.stop()
records=[(f,read(f,{})) for f in files]
records=[(f,r) for f,r in records if r.get('rows')]
labels=[r['target_date']+' · '+str(r['days_ahead'])+'일 범위 · '+('공식가격 확보' if r['mode']=='official_known' else '모델 추정 · 공개 적격성 미확인') for _,r in records]
choice=st.selectbox('조회할 가격·예측 기록',range(len(records)),format_func=lambda i:labels[i])
file,record=records[choice]
st.write('생성 시각: **'+record['issued_at']+'**')
st.write('23시 기준 피처 시각: '+record['reference_23h_origin']+' · 모델: '+record['model_family'])
if record['mode']=='official_known': st.info('이 날짜의 공식가격은 예측 실행 전에 이미 확보했습니다. EV 계획에는 공식가격을 사용합니다. 모델 추정은 진정한 미래 예측 성과로 집계하지 않습니다.')
else:
    st.warning('조회한 공식 출처에서 대상 가격을 확보하지 못했습니다. 모든 경로에서 미공개임을 입증한 것은 아닙니다.')
    if record['mode'].startswith('pre23'): st.caption('23시 이전 사전보기입니다. 실제 생성시각을 기록했으며 23시에 실행한 예측으로 표시하지 않습니다.')
f=pd.DataFrame(record['rows']); f.timestamp=pd.to_datetime(f.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
f=f.merge(actual[['timestamp','smp']],on='timestamp',how='left')
chart=f.set_index('timestamp')[['predicted_smp','smp','lower90','upper90']].rename(columns={'predicted_smp':'모델 추정','smp':'확보된 공식가격','lower90':'하한','upper90':'상한'})
st.line_chart(chart)
st.caption('단위 원/kWh. 음영 대신 하한·상한선을 표시합니다. 검증 잔차 기반 구간이며 보장된 발생확률이 아닙니다.')
st.write('고가 추정 시간: '+str(int((f.decision_price>=150).sum()))+'시간 · 불확실성 상한이 150원/kWh 이상인 시간: '+str(int((f.upper90>=150).sum()))+'시간')
with st.expander('시간별 가격 및 위험'):
    st.dataframe(f[['timestamp','predicted_smp','smp','lower90','upper90','high_price_risk']],hide_index=True)

st.subheader('EV 충전 계획 · 가상 SMP 연동 요금')
c1,c2,c3=st.columns(3)
initial=c1.slider('현재 SOC (%)',0,100,30)
target=c2.slider('목표 SOC (%)',0,100,80)
capacity=c3.number_input('배터리 용량 (kWh)',min_value=10.,max_value=200.,value=60.)
power=st.number_input('충전 출력 (kW)',min_value=1.,max_value=50.,value=7.)
eff=.9; battery=max(0.,(target-initial)*capacity/100); grid=battery/eff
duration=math.ceil(grid/power) if grid else 0
if not duration: st.info('추가 충전이 필요하지 않습니다.')
elif duration>24: st.warning('이 24시간 범위에서는 목표 SOC에 도달할 수 없습니다.')
else:
    weights=np.full(duration,power); weights[-1]=grid-power*(duration-1)
    prices=f.decision_price.to_numpy(); costs=np.array([np.dot(prices[s:s+duration],weights) for s in range(25-duration)])
    begin=int(costs.argmin()); end=begin+duration
    st.success(f'추천 시작 {f.timestamp.iloc[begin]:%m-%d %H:%M} · 연속 {duration}개 시간 구간 · 가상 구매비 {costs[begin]:,.1f}원')
    st.caption('충전효율90%, 마지막 구간 부분충전. 세금·망요금·실제 충전사업자 요금·V2G 수익은 포함하지 않습니다. 공식가격 미확보 시 모델 추정가격으로 계산합니다.')
    st.dataframe(f.iloc[begin:end][['timestamp','decision_price']],hide_index=True)

st.subheader('최근 기록의 사후 비교')
accuracy=read(HOME/'recent_accuracy.json',[])
if accuracy: st.dataframe(pd.DataFrame(accuracy)[['target_date','mode','hours','MAE']],hide_index=True)
else: st.info('평가할 예측 기록이 없습니다.')
st.caption('공식가격 확보 모드의 오차는 모델 진단이며 사전 예측 정확도가 아닙니다. 아직 실제값이 없는 미래 기록은 점수를 계산하지 않습니다.')
with st.expander('수집 상태 및 데이터 출처'):
    st.dataframe(pd.DataFrame(read(HOME/'coverage.json',[])),hide_index=True)
    st.write('출처: KPX EPSIS 시간별 SMP 및 KPX 일반 공개 육지 SMP. 제주 자료는 별도 보관하며 이 화면에서 육지와 혼합하지 않습니다.')
    st.write('기상·수요 예보의 당시 발표본이 없어 현재 모델은 SMP 이력과 달력만 사용합니다.')
    if not failures.empty: st.dataframe(failures,hide_index=True)

exo=HOME/'exogenous'
if (exo/'production_exogenous.joblib').exists():
    with st.expander('수요·기상 외생변수 실험 · 기존 EV 계산과 별도'):
        st.caption('원 V11은 SMP·달력21피처입니다. 아래 모델은 지연된 일별 수요·공급예비력과 서울 기상을 추가한43피처 실험입니다.')
        options=sorted((exo/'forecasts').glob('*.json'),reverse=True)
        if options:
            extra=read(options[0],{})
            st.write('대상 '+extra['target_date']+' · 실제 생성 '+extra['issued_at'])
            st.write('외생변수 기준일 '+extra['first_exogenous_day']+' · '+extra['mode'])
            ef=pd.DataFrame(extra['rows']); ef.timestamp=pd.to_datetime(ef.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
            ef=ef.merge(actual[['timestamp','smp']],on='timestamp',how='left')
            st.line_chart(ef.set_index('timestamp')[['predicted_smp','smp']])
            st.warning('시간별2026 수요 실적이나 당시 기상예보로 검증된 모델이 아닙니다. 사후 역사자료의 발표시각은 미확인입니다. 현재 실제 수신한 과거 일별 관측만 입력하며 미래 실제 기상·수요는 넣지 않습니다.')
            st.write('실제 입력변수',extra['features'])
        proxy=sorted((exo/'weather_proxy_forecasts').glob('*.json'),reverse=True)
        if proxy:
            st.caption('실제 발표 기상예보를 입력한 별도 연구 시나리오도 저장했습니다. 과거 학습에는 관측 대용값을 사용해 예보 입력 성능은 검증되지 않았으며 EV 추천에는 적용하지 않습니다.')
        api=read(exo/'api_connection_status.json',{})
        st.write('API 연결 상태', {k:v for k,v in api.items() if k!='at'})

if (HOME/'kma_apihub/connection_status.json').exists():
    with st.expander('기상청 API허브 · 시간 관측 수집 현황'):
        st.write(read(HOME/'kma_apihub/connection_status.json',{}))
        inventory=read(HOME/'kma_apihub/inventory.json',{})
        stations=inventory.get('stations',[])
        if stations:
            st.dataframe(pd.DataFrame([{k:v for k,v in s.items() if k!='missing_fraction'} for s in stations]),hide_index=True)
        st.caption('관측자료와 발표 예보는 별도입니다. 시간 관측자료는 수집 중이며 연간 전체 자료 확보를 의미하지 않습니다. 과거 발표 예보 API가 승인되기 전에는 이 자료로 예보 모델을 대체하거나 기본 운영 모델을 변경하지 않습니다.')
