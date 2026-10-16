"""Local V17 user-centred charging demonstration, never a public deployment."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
from datetime import time,timedelta
import json,hashlib
import joblib,numpy as np,pandas as pd,streamlit as st,altair as alt
from v11.store import connect,latest
from v12.ev import EV,score,baseline
from v12.records import write_once
from v15.policies import dispatch
from v17.common import HOME,ROOT,csv,sha
from v17.charging import PATTERNS,priority_plan,require_prices
from v17.study import query,infer,MODELS

LABEL={'joint_price':'가격 회귀 기준','cheap_classifier':'연결시간 저가 분류','window_cost_rank':'충전비용 중심 순위','rank_price_blend':'가격·순위 혼합','known_profile':'기존 가격 프로파일'}
PATTERN={'day09_18':'주간 09~18시','night18_08':'야간 18~다음날08시','short3':'짧은 충전3시간','short4':'짧은 충전4시간','short5':'짧은 충전5시간','long8':'충분한 시간8시간','long12':'충분한 시간12시간','full24':'하루24시간'}
st.set_page_config(page_title='전력예보 V17 · 충전 시간 추천',page_icon='⚡',layout='wide')
st.markdown('<style>.block-container{max-width:1300px;padding-top:1.5rem}div[data-testid="stMetric"]{background:#edf4fa;padding:12px;border-radius:12px}</style>',unsafe_allow_html=True)
st.title('⚡ 전력예보 V17 · 언제 충전하면 좋을까요?')
st.caption('공식 SMP 또는 저장된 예측으로 충전 시간을 비교합니다. 표시 비용은 가상 SMP 연동 정산이며 국내 실제 충전요금·실증 V2G 판매수익이 아닙니다.')
db=connect();official=latest(db,'land');db.close();official.timestamp=pd.to_datetime(official.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
def read(path,default=None):return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default
def lines(f,fields,title,ylabel):
    g=f.melt(id_vars='timestamp',value_vars=fields,var_name='항목',value_name=ylabel)
    chart=alt.Chart(g).mark_line(point=True).encode(x=alt.X('timestamp:T',title='한국시간'),y=alt.Y(ylabel+':Q',title=ylabel),color=alt.Color('항목:N',title='범례'),tooltip=['timestamp:T','항목:N',alt.Tooltip(ylabel+':Q',format='.2f')]).properties(title=title,height=280)
    st.altair_chart(chart,use_container_width=True)

with st.sidebar:
    st.header('비교 자료')
    mode=st.radio('자료 모드',['공식 공개가격 예제','저장된 실제 전향 예측','과거 고정 검증 예제'])
    record=None;feature_group=None;bundle=None;source_name='';selected_model='joint_price'
    if mode=='공식 공개가격 예제':
        dates=sorted(official.timestamp.dt.strftime('%Y-%m-%d').unique(),reverse=True);day=st.selectbox('공식가격 날짜',dates)
        selected=official[official.timestamp.dt.strftime('%Y-%m-%d').isin([day,(pd.Timestamp(day)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')])];source=selected[['timestamp','smp']].rename(columns={'smp':'price'});source_name='공식 공개가격 · AI 예측 아님'
    elif mode=='저장된 실제 전향 예측':
        records=[]
        for version in ['v12','v14','v15','v17']:
            for p in (ROOT/'artifacts'/version/'forecasts').glob('*.json'):
                r=read(p);rows=r.get('price_rows',r.get('rows',[]))
                if rows:records.append((p,r))
        records.sort(key=lambda item:item[1].get('issued_at',''),reverse=True)
        if not records:st.info('저장된 전향 예측이 없습니다.');st.stop()
        selected=st.selectbox('불변 예측 기록',range(len(records)),format_func=lambda i:records[i][0].name);record_path,record=records[selected];day=record.get('primary_target_date',record.get('target_date'))
        source=pd.DataFrame(record.get('price_rows',record.get('rows')))[['timestamp','predicted_smp']].rename(columns={'predicted_smp':'price'});source.timestamp=pd.to_datetime(source.timestamp,utc=True).dt.tz_convert('Asia/Seoul');source_name='저장된 실제 발급 예측 · 이후에 실제값으로 바꾸지 않음'
        if record.get('official_smp_known_at_issue') or record.get('official_known_at_issue'):source_name='공식가격 기공개 후 진단 · AI 미래예측 아님'
        st.caption('생성시각: '+record.get('issued_at','미기록'));st.caption('기준시각: '+record.get('reference_origin','미기록'))
        if record.get('actual_delay_minutes',0)>1:st.warning(f"기준시각보다 {record['actual_delay_minutes']:.1f}분 늦게 생성됐습니다. 23시 정시 예측으로 집계하지 않습니다.")
        if record.get('model_locked_after_reference_origin'):st.caption('이 기록은 모델 고정도 기준시각 이후인 지연 실험입니다. 정시 전향 검증과 구분합니다.')
    else:
        fold=st.selectbox('과거 시간순 검증 구간',['spring2025','winter2025'],format_func=lambda s:'2025년 봄' if s=='spring2025' else '2025년 겨울')
        matrix=csv(HOME/'study/matrix.csv');matrix.timestamp=pd.to_datetime(matrix.timestamp,utc=True).dt.tz_convert('Asia/Seoul');limits={'spring2025':('2025-04-01','2025-06-29'),'winter2025':('2025-10-01','2025-12-30')}[fold];dates=sorted(matrix.loc[matrix.base_day.between(*limits),'base_day'].unique());day=st.selectbox('과거 시연 날짜',dates)
        feature_group=matrix[matrix.base_day==day].copy();bundle=joblib.load(HOME/'study'/(fold+'.joblib'));point=feature_group.daily.to_numpy()+bundle['regressor'].predict(feature_group[bundle['price_features']]);source=feature_group[['timestamp']].copy();source['price']=point;source_name='과거 고정 검증 시연 · 현재 미래예측 아님'
        selected_model=st.selectbox('충전 우선순위 모델',MODELS,format_func=lambda n:LABEL[n])
    st.caption('기본 권장: G2V. 실험 모델과 V2G는 검증 중이며 운영 모델을 교체하지 않았습니다.')

tabs=st.tabs(['충전 계획 추천','모델·경제성 검증','전향·수집 현황','급등·데이터 한계'])
with tabs[0]:
    if mode=='공식 공개가격 예제':st.info(source_name)
    elif mode=='저장된 실제 전향 예측':st.success(source_name)
    else:st.warning(source_name)
    graph=source.rename(columns={'price':'선택한 계획 가격'}).copy();graph['현재 확보된 공식 SMP']=official.set_index('timestamp').smp.reindex(graph.timestamp).to_numpy();lines(graph,['선택한 계획 가격','현재 확보된 공식 SMP'],'가격과 실제값 비교','원/kWh')
    st.caption('실제값이 없는 구간은 선을 채우지 않습니다. 순위 모델의 우선점수는 원/kWh 예측값과 구분합니다.')
    with st.form('v17_conditions'):
        st.subheader('차량과 연결 조건')
        a,b,c=st.columns(3);capacity=a.number_input('배터리 용량(kWh)',min_value=1.,max_value=300.,value=60.);initial=b.number_input('초기 SOC(%)',min_value=0.,max_value=100.,value=30.);target=c.number_input('목표 SOC(%)',min_value=0.,max_value=100.,value=80.)
        a,b,c=st.columns(3);power=a.number_input('최대 충전전력(kW)',min_value=.1,max_value=350.,value=7.);eff=b.slider('충전·방전 각 효율(%)',50,100,90);pattern=c.selectbox('연결 패턴',list(PATTERN),format_func=lambda p:PATTERN[p])
        custom=st.checkbox('연결시각 직접 입력')
        a,b,c=st.columns(3);start_time=a.time_input('직접 입력: 연결 시작',value=time(9));end_time=b.time_input('직접 입력: 출발',value=time(18));next_day=c.checkbox('직접 입력: 다음 날짜 출발')
        with st.expander('가상 V2G·배터리 제한'):
            a,b,c=st.columns(3);discharge=a.number_input('최대 방전전력(kW)',min_value=0.,max_value=350.,value=7.);wear=b.number_input('열화비용(원/배터리 통과kWh)',min_value=0.,max_value=100.,value=5.);sell_ratio=c.slider('가상 판매 정산비율(%)',0,100,100)
            a,b=st.columns(2);min_soc=a.number_input('최소 SOC(%)',0.,100.,20.);max_soc=b.number_input('최대 SOC(%)',0.,100.,90.)
        calculate=st.form_submit_button('충전 계획 계산',type='primary')
    if calculate:
        try:
            base=pd.Timestamp(day,tz='Asia/Seoul')
            if custom:
                start=base+pd.Timedelta(hours=start_time.hour,minutes=start_time.minute);end=base+pd.Timedelta(days=int(next_day),hours=end_time.hour,minutes=end_time.minute)
            else:
                _,h,n=next(p for p in PATTERNS if p[0]==pattern);start=base+pd.Timedelta(hours=h);end=start+pd.Timedelta(hours=n)
            selected=require_prices(source,start,end);times=pd.DatetimeIndex(selected.timestamp);price=selected.price.to_numpy();priority=price.copy();proxy=False
            ev=EV(capacity=capacity,initial_soc=initial,target_soc=target,charge_kw=power,discharge_kw=discharge,charge_eff=eff/100,discharge_eff=eff/100,min_soc=min_soc,max_soc=max_soc,degradation=wear,sell_ratio=sell_ratio/100)
            if feature_group is not None and selected_model!='joint_price':
                q=query(feature_group,start.hour,len(times),ev);point,scores=infer(bundle,q);priority=scores[selected_model];price=priority if selected_model=='known_profile' else np.sort(point)[pd.Series(priority).rank(method='first').astype(int).to_numpy()-1];proxy=selected_model!='known_profile'
                st.caption('개별 순위점수로 충전 순서를 정했습니다. 표시 비용 계산용 가격배열은 회귀 가격을 재배열한 의사결정 대용값이며 별도 SMP 가격예측으로 주장하지 않습니다. 사용자 조건이 연구 기본값과 다르면 일반화 성능은 미확인입니다.')
            g=priority_plan(price,times,priority,ev,start,end);v=dispatch('general',price,times,None,ev,start,end);gs=g['schedule'];vs=v['schedule'];actual=official.set_index('timestamp').smp.reindex(times).to_numpy();ready=np.isfinite(actual).all()
            immediate=baseline(price,times,ev,start,end);uniform=baseline(price,times,ev,start,end,True);plans=[('즉시충전',immediate),('균등충전',uniform),('추천 G2V',g),('가상 V2G',v)];rows=[]
            for label,plan in plans:
                realized=None
                if ready:
                    if label=='즉시충전':realized=baseline(actual,times,ev,start,end)['net_cost']
                    elif label=='균등충전':realized=baseline(actual,times,ev,start,end,True)['net_cost']
                    else:
                        schedule=plan['schedule'];realized=score(actual,schedule.charge_kw.to_numpy(),schedule.discharge_kw.to_numpy(),schedule.connected_hours.to_numpy(),ev)['net_cost']
                rows.append({'방식':label,'예상 가상비용(원)':plan['net_cost'],'사후 가상비용(원)':realized})
            st.subheader('추천: G2V 충전')
            st.write(f"연결 {start:%m/%d %H:%M}~{end:%m/%d %H:%M}, 목표 충전량 {max(0,capacity*(target-initial)/100):.2f}kWh. 더 낮은 계획 가격 또는 저가 우선순위 시간부터 최대전력·효율·SOC 제약 안에서 충전합니다.")
            a,b,c=st.columns(3);a.metric('종료 SOC',f"{g['ending_soc']:.1f}%");b.metric('계통 충전량',f"{g['grid_import_kwh']:.2f}kWh");c.metric('즉시충전 대비 예상 절감',f"{immediate['net_cost']-g['net_cost']:.2f}원")
            st.dataframe(pd.DataFrame(rows),hide_index=True);st.caption('빈 사후 비용은 실제값 수신 대기입니다. 예상 절감을 실현 절감으로 표시하지 않습니다. 열화비용을 포함한 가상 비용입니다.')
            power_chart=gs[['timestamp','charge_kw']].rename(columns={'charge_kw':'추천 충전전력(kW)'});power_chart['가상 방전전력(kW)']=-vs.discharge_kw.to_numpy();power_chart['가상 충전전력(kW)']=vs.charge_kw.to_numpy();lines(power_chart,list(power_chart.columns[1:]),'시간별 충방전 계획','kW')
            soc=gs[['timestamp','soc_end']].rename(columns={'soc_end':'G2V SOC'});soc['가상 V2G SOC']=vs.soc_end.to_numpy();lines(soc,['G2V SOC','가상 V2G SOC'],'배터리 SOC 변화','SOC(%)')
            st.dataframe(gs[['timestamp','connected_hours','charge_kw','soc_start','soc_end']].rename(columns={'timestamp':'한국시간','connected_hours':'연결시간(h)','charge_kw':'충전전력(kW)','soc_start':'시작 SOC(%)','soc_end':'종료 SOC(%)'}),hide_index=True)
            expected_gain=g['net_cost']-v['net_cost'];st.write(f"가상 V2G 예상 추가이익 {expected_gain:.2f}원 · 방전량 {v['grid_export_kwh']:.2f}kWh")
            if ready:
                gc=rows[2]['사후 가상비용(원)'];vc=rows[3]['사후 가상비용(원)'];oracle=priority_plan(actual,times,ev=ev,start=start,end=end);st.write(f"사후 G2V regret {gc-oracle['net_cost']:.2f}원 · 가상 V2G 실제 추가손익 {gc-vc:.2f}원")
            st.warning('기본 G2V 권장 유지: 시간순 검증의 동시 개선 기준을 충족한 새 후보가 없고, V2G 예상이익은 가격오차로 손실로 바뀔 수 있습니다. 공식가격 모드의 가상 이익도 실제 판매수익은 아닙니다.')
            write_once(HOME/'interactive_decisions'/(pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%f')+'.json'),dict(mode='INTERACTIVE_SIMULATION_NOT_NEW_FORWARD',generated_at=pd.Timestamp.now(tz='Asia/Seoul').isoformat(),source_mode=mode,source_record_sha256=sha(record_path) if record is not None else None,EV=ev.__dict__,decision_proxy=proxy,start=start.isoformat(),end=end.isoformat(),G2V={**{k:x for k,x in g.items() if k!='schedule'},'schedule':gs.to_dict('records')},V2G={**{k:x for k,x in v.items() if k!='schedule'},'schedule':vs.to_dict('records')},costs=rows))
        except ValueError as exc:
            messages={'SOC bounds violated':'초기·목표 SOC가 최소·최대 SOC 범위를 벗어났습니다. 배터리 제한을 확인하세요.','invalid capacity, power or efficiency':'배터리 용량·전력·효율 입력을 확인하세요.','invalid settlement assumptions':'열화비용·가상 판매 정산비율을 확인하세요.','valid timezone-aware connection window required':'연결 종료시각은 시작시각 이후여야 합니다. 날짜를 넘으면 다음 날짜 출발을 선택하세요.','departure SOC infeasible in connection window':'연결시간과 최대전력으로 목표 SOC를 달성할 수 없습니다.'}
            st.error(messages.get(str(exc),str(exc)))
    if record is not None and record.get('schema')==17:
        st.subheader('실제 발급 당시 고정한 계획')
        fixed=st.selectbox('고정 연결 패턴',[p for p,e in record['decisions'].items() if e['status']=='frozen'],format_func=lambda p:PATTERN[p]);model=st.selectbox('고정 모델',list(record['decisions'][fixed]['plans']),format_func=lambda n:LABEL[n])
        if st.button('저장된 사전 충전계획 보기'):
            frozen=record['decisions'][fixed]['plans'][model];s=pd.DataFrame(frozen['G2V']['schedule']);s.timestamp=pd.to_datetime(s.timestamp,utc=True).dt.tz_convert('Asia/Seoul');lines(s.rename(columns={'charge_kw':'고정 충전전력'}),['고정 충전전력'],'원본 사전 결정 · 재최적화 아님','kW');st.dataframe(s[['timestamp','charge_kw','soc_start','soc_end']].rename(columns={'timestamp':'한국시간','charge_kw':'고정 충전전력(kW)','soc_start':'시작 SOC(%)','soc_end':'종료 SOC(%)'}),hide_index=True)

with tabs[1]:
    st.subheader('충전 시간대 모델 검증')
    path=ROOT/'docs/V17_MODEL_PATTERN_COMPARISON.csv'
    if path.exists():
        f=csv(path);role=st.selectbox('연구 구간',['validation','seen_diagnostic'],format_func=lambda r:'2025년 시간순 검증' if r=='validation' else '이미 열람한2026H1 진단');g=f[f.role==role].copy();g['model']=g.model.map(LABEL);g['pattern']=g.pattern.map(PATTERN)
        st.dataframe(g[['fold','model','pattern','days','cheap_hour_overlap','rank_spearman','G2V_actual','G2V_regret','saving_immediate','saving_uniform','V2G_actual_gain','V2G_export','V2G_loss_days']].rename(columns={'model':'모델','pattern':'연결시간','cheap_hour_overlap':'저가시간 일치율','G2V_actual':'G2V 사후 가상비용','G2V_regret':'G2V regret','saving_immediate':'즉시 대비 절감','saving_uniform':'균등 대비 절감','V2G_actual_gain':'V2G 추가손익','V2G_export':'방전kWh/일','V2G_loss_days':'추가손실일'}),hide_index=True)
    st.info('회귀 MAE와 충전 비용은 다른 지표입니다. 분류·순위 점수는 가격예측값이 아니며 비용용 재배열 가격은 대용값입니다. 3~4시간 연결은 기본 조건에서30kWh 충전이 불가능합니다. 일부 연결시간의 개선을 모든 차량의 개선으로 확대하지 않습니다.')
    old=ROOT/'docs/V17_V14_RANK_AUDIT.csv'
    if old.exists():st.write('기존 V14 순위 학습 재검토');st.dataframe(csv(old),hide_index=True)
with tabs[2]:
    st.subheader('실제 전향·자동 수집 상태')
    status=read(HOME/'prospective_status.json',{})
    if status:st.json(status)
    service=read(HOME/'service_status.json',{})
    if service:st.write('최근 복구 서비스 실행');st.json({k:v for k,v in service.items() if k!='gap'})
    st.caption('작업 스케줄러는 노트북이 종료·절전 상태이면 실행을 보장하지 않습니다. 공백은 누락으로 기록하며 정시 실행 또는 평가 완료로 채우지 않습니다. 다음 날짜 가격은 같은 기준시각의 D3 예측 또는 실제 공개가격이 있어야 사용합니다.')
with tabs[3]:
    st.subheader('급등과 입력자료의 한계')
    st.write('가격 이력만으로 한계발전기·고장·순부하 변화를 모두 알 수 없습니다. V16 정비계획198건은 당시 최초 첨부와 현재 파일의 일치가 미확인돼 후향 연구로 유지합니다. 기상 관측을 미래 예보로 바꾸지 않았습니다.')
    p=ROOT/'docs/V17_SOURCE_AVAILABILITY.json'
    if p.exists():st.dataframe(pd.DataFrame(read(p)),hide_index=True)
    for name in ['known_October_failure.png']:
        p=ROOT/'docs/v15_figures'/name
        if p.exists():st.image(str(p),caption='이미 열람한2026년10월 실패 진단 · 신규 모델 선정에 사용하지 않음')
    st.write('가상 V2G 손익분기: 판매비율×방전가격 > 충전가격/(충전효율×방전효율) + 2×열화단가/방전효율. 가격차 조건만으로 미래 수익이 보장되지 않습니다.')
