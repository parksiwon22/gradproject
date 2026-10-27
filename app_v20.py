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
from v20.core import HOME as OUTPUT,select_auto,write_audit_once
OUTPUT.mkdir(parents=True,exist_ok=True)

st.set_page_config(page_title='전력예보 V20 · D-3/D-2 검증',page_icon='⚡',layout='wide')
st.title('⚡ 전력예보 V20 · 예측과 충전비용 검증')
st.caption('육지 SMP · KST. 저장된 AI와 이후 수신한 공식가격을 비교합니다. 가상 SMP 정산이며 실제 충전요금·실증 V2G 수익이 아닙니다.')
page=st.sidebar.radio('화면',['SMP 예측·공식가격 비교','과거 예측 시연','최종 연구 fold 시연','EV 충전 추천','연구 예측 정확도','연구 결과·시스템 상태'])

@st.cache_data(ttl=20)
def load():
    actual=reference()[0].sort_values(['timestamp','source','first_seen']).drop_duplicates('timestamp',keep='last')
    return inventory(),actual.rename(columns={'smp':'price'})
items,official=load();status=read(OUTPUT/'status.json',{});publisher_status=read(HOME/'publish_status.json',{});preparation_status=read(HOME/'preparation_status.json',{});a,b,c=st.columns(3);a.metric('독립 기준시각',status.get('independent_origins',0));b.metric('독립 기본 대상일',status.get('independent_primary_target_days',0));c.metric('저장 시스템 정시 적격 기록',status.get('V19_on_time_eligible',0))
if st.sidebar.button('저장 자료 새로 고침'):load.clear();st.rerun()

def line(frame,cols,title,unit):
    f=frame.melt('timestamp',value_vars=cols,var_name='항목',value_name='값');chart=alt.Chart(f).mark_line(point=True).encode(x=alt.X('timestamp:T',title='한국시간'),y=alt.Y('값:Q',title=unit),color='항목:N',tooltip=['timestamp:T','항목:N',alt.Tooltip('값:Q',format='.2f')]).properties(title=title,height=270);st.altair_chart(chart,width='stretch')

def research_csv(path):
    if not path.exists() or path.stat().st_size<=2:return None
    try:return csv(path)
    except pd.errors.EmptyDataError:return None

def conditions(prefix,day,overnight=False):
    a,b,c,d=st.columns(4);cap=a.number_input('배터리 용량(kWh)',1.,300.,60.,key=prefix+'cap');initial=b.number_input('초기 SOC(%)',0.,100.,30.,key=prefix+'initial');target=c.number_input('목표 SOC(%)',0.,100.,80.,key=prefix+'target');power=d.number_input('충전전력(kW)',.1,350.,7.,key=prefix+'power')
    base=pd.Timestamp(day);a,b,c,d=st.columns(4);bd=a.date_input('연결 시작일',base.date(),key=prefix+'bd');bt=b.time_input('연결 시작시각',time(18) if overnight else time(0),key=prefix+'bt');ed=c.date_input('출발일',(base+pd.Timedelta(days=1)).date(),key=prefix+'ed');et=d.time_input('출발시각',time(8) if overnight else time(0),key=prefix+'et')
    with st.expander('효율·SOC·가상 V2G 가정'):
        a,b,c=st.columns(3);eff=a.slider('충방전 각 효율(%)',50,100,90,key=prefix+'eff');wear=b.number_input('열화비용(원/배터리 통과kWh)',0.,100.,5.,key=prefix+'wear');sell=c.slider('가상 판매 정산비율(%)',0,100,100,key=prefix+'sell')
        a,b,c=st.columns(3);minimum=a.number_input('최소 SOC(%)',0.,100.,20.,key=prefix+'min');maximum=b.number_input('최대 SOC(%)',0.,100.,90.,key=prefix+'max');discharge=c.number_input('방전전력(kW)',0.,350.,7.,key=prefix+'dis')
    return EV(cap,initial,target,power,discharge,eff/100,eff/100,minimum,maximum,wear,sell/100),pd.Timestamp(datetime.combine(bd,bt),tz='Asia/Seoul'),pd.Timestamp(datetime.combine(ed,et),tz='Asia/Seoul')

def message(e):return {'SOC bounds violated':'초기·목표 SOC와 배터리 상하한을 확인하세요.','departure SOC infeasible in connection window':'연결시간과 전력으로 목표 SOC를 달성할 수 없습니다.'}.get(str(e),str(e))

def show_plans(sources,ev,start,end,actual_reference=None,actual_label='공식가격'):
    actual=None
    try:actual=require_prices(official[['timestamp','price']],start,end).price.to_numpy()
    except ValueError:pass
    if actual_reference is not None:
        try:actual=require_prices(actual_reference[['timestamp','price']],start,end).price.to_numpy()
        except ValueError:actual=None
    costs=[];powers=[];soc=[];saved={}
    for label,f in sources.items():
        if f is None:st.info(label+': 미확보 · 다른 날짜나 모델 값으로 대체하지 않습니다.');continue
        try:
            selected=require_prices(f[['timestamp','price']],start,end);times=pd.DatetimeIndex(selected.timestamp);p=selected.price.to_numpy();g=priority_plan(p,times,ev=ev,start=start,end=end);v=dispatch('general',p,times,None,ev,start,end);saved[label]=dict(G2V=pack(g),V2G=pack(v))
            for kind,plan in [('G2V',g),('가상 V2G',v)]:
                actual_cost=regret=None
                if actual is not None:
                    s=plan['schedule'];actual_cost=score(actual,s.charge_kw.to_numpy(),s.discharge_kw.to_numpy(),s.connected_hours.to_numpy(),ev)['net_cost'];oracle=priority_plan(actual,times,ev=ev,start=start,end=end) if kind=='G2V' else dispatch('general',actual,times,None,ev,start,end);regret=actual_cost-oracle['net_cost']
                costs.append({'자료':label,'방식':kind,'계획가격 비용(원)':plan['net_cost'],actual_label+' 평가비용(원)':actual_cost,'regret(원)':regret,'방전(kWh)':plan['grid_export_kwh'],'종료 SOC(%)':plan['ending_soc']})
                s=plan['schedule'];powers.append(pd.DataFrame(dict(timestamp=s.timestamp,계열=label+' '+kind,전력=s.charge_kw-s.discharge_kw)));soc.append(pd.DataFrame(dict(timestamp=s.timestamp,계열=label+' '+kind,SOC=s.soc_end)))
            if True:  # Baselines use the same connection and EV even when only AI is available.
                for kind,uniform in [('즉시충전',False),('균등충전',True)]:
                    base=baseline(p,times,ev,start,end,uniform);cost=base['net_cost'];real=None if actual is None else baseline(actual,times,ev,start,end,uniform)['net_cost'];reg=None if actual is None else real-priority_plan(actual,times,ev=ev,start=start,end=end)['net_cost'];costs.append({'자료':label,'방식':kind,'계획가격 비용(원)':cost,actual_label+' 평가비용(원)':real,'regret(원)':reg,'방전(kWh)':0,'종료 SOC(%)':max(ev.initial_soc,ev.target_soc)})
            st.caption(label+' G2V 추천시간: '+', '.join(t.strftime('%m/%d %H:%M') for t in g['schedule'].loc[g['schedule'].charge_kw>1e-8,'timestamp']))
        except ValueError as exc:st.warning(label+': '+message(exc))
    if costs:
        table=pd.DataFrame(costs);st.dataframe(table,hide_index=True)
        if actual is not None:
            for label in sources:
                a=table[(table['자료']==label)&(table['방식']=='G2V')];b=table[(table['자료']==label)&(table['방식']=='가상 V2G')]
                cost_col=actual_label+' 평가비용(원)'
                if len(a) and len(b):st.write(f"{label} 가상 V2G 추가손익: {a.iloc[0][cost_col]-b.iloc[0][cost_col]:.2f}원 · 방전 {b.iloc[0]['방전(kWh)']:.2f}kWh")
        for data,value,title,unit in [(powers,'전력','충방전 전력','kW'),(soc,'SOC','배터리 SOC','%')]:
            f=pd.concat(data);chart=alt.Chart(f).mark_line(point=True).encode(x=alt.X('timestamp:T',title='KST'),y=alt.Y(value+':Q',title=unit),color='계열:N',tooltip=['timestamp:T','계열:N',value+':Q']).properties(title=title,height=265);st.altair_chart(chart,width='stretch')
        note=('fold 실제값을 이용한 사후 시뮬레이션이며 운영 결과가 아닙니다.' if actual_reference is not None
              else '공식 제공가격과 최종 정정통계는 구분합니다.')
        st.warning('기본 권장 G2V 유지. 화면에서 다시 계산한 계획은 현재 조건의 시뮬레이션이며 원래 사전 계획을 수정하지 않습니다. '+note)
        if st.button('비교 시뮬레이션 로컬 저장'):
            write_audit_once(OUTPUT/'interactive_decisions'/(pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%f')+'.json'),dict(at=now(),mode='RETROSPECTIVE_INTERACTIVE_NOT_NEW_FORWARD',EV=ev.__dict__,start=start.isoformat(),end=end.isoformat(),plans=saved,costs=costs));st.success('로컬 시뮬레이션으로 저장했습니다. 새 사전 예측 수에는 포함하지 않습니다.')

@st.cache_data(ttl=300)
def load_fold_demo():
    path=ROOT/'docs'/'smp_improvement_ml_lgbm'/'final_fold_demo.csv.gz'
    if not path.exists():return pd.DataFrame()
    return csv(path)

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

if page=='SMP 예측·공식가격 비교':
    dates=sorted({d for _,_,f,_ in items for d in f.timestamp.dt.strftime('%Y-%m-%d')} | set(official.timestamp.dt.strftime('%Y-%m-%d')));default=dates.index('2026-10-18') if '2026-10-18' in dates else max(0,len(dates)-1);day=st.selectbox('동일 비교 대상일',dates,index=default);d3,m3=select_forecast(day,3);d2,m2=select_forecast(day,2)
    for h,m in [(3,m3),(2,m2)]:
        if m:st.caption(f"D-{h} 기준 {m['reference_origin']} / 실제 발급 {m['issued_at']} / {m['timing']} / 모델 고정 확인 {m['model_locked_before_origin']}")
        else:st.info(f'D-{h} 실제 저장 예측 미확보 · 과거 발급을 새로 만들지 않습니다.')
    times=pd.date_range(pd.Timestamp(day,tz='Asia/Seoul'),periods=24,freq='h');graph=pd.DataFrame(dict(timestamp=times));graph['공식 SMP']=official.set_index('timestamp').price.reindex(times).to_numpy()
    for label,frame in [('D-3 AI',d3),('D-2 AI',d2)]:graph[label]=np.nan if frame is None else frame.set_index('timestamp').price.reindex(times).to_numpy()
    st.write('공식가격 확보: '+str(int(graph['공식 SMP'].notna().sum()))+'/24시간 · 최종 정정통계 확인 대기');line(graph,['D-3 AI','D-2 AI','공식 SMP'],'같은 날짜의 두 예측시점과 공식가격','원/kWh')
    with st.expander('시간별 예측값과 공식가격 차이'):
        details=graph.copy();details['D-3 오차(원/kWh)']=details['D-3 AI']-details['공식 SMP'];details['D-2 오차(원/kWh)']=details['D-2 AI']-details['공식 SMP'];st.dataframe(details,hide_index=True)
    if (HOME/'comparison.csv').exists():
        f=csv(HOME/'comparison.csv');g=f[f.target_date==day];st.dataframe(g[['model','horizon_days','reference_origin','issued_at','comparison_group','MAE','RMSE','peak_MAE','peak_F1','cheap5_overlap','G2V_regret','V2G_actual_gain']].rename(columns={'model':'모델','horizon_days':'D-몇일','cheap5_overlap':'저가5시간 일치율','V2G_actual_gain':'가상 V2G 추가손익'}),hide_index=True)
    st.subheader('같은 차량 조건으로 가격별 계획 재생');ev,start,end=conditions('compare_'+day,day);show_plans({'D-3 AI':d3,'D-2 AI':d2,'공식 SMP':official},ev,start,end)
    if (HOME/'frozen_costs.csv').exists():
        with st.expander('원래 동결한 사전 계획의 평가 · 위 재생과 별도'):
            f=csv(HOME/'frozen_costs.csv');st.dataframe(f[f.target_date==day],hide_index=True)
elif page=='최종 연구 fold 시연':
    demo=load_fold_demo()
    st.subheader('최종 연구 후보 · 과거 검증 fold 시연')
    st.warning('과거 검증 구간의 fold 모델 저장 예측입니다. 실제 당시 운영 발급 기록이 아니며, 원본 SMP 수신 시각과 당시 모델 파일 가용성이 확인되지 않아 검증된 운영 성능으로 볼 수 없습니다. 운영 모델과 공식 발급 gate에는 연결되지 않습니다.')
    if demo.empty:
        st.error('연구 fold 시연 자료가 없습니다.')
    else:
        horizons=sorted(demo.horizon_days.unique().tolist());h=st.selectbox('예측 기간',horizons,format_func=lambda x:f'D+{x}',key='fold_demo_horizon');part=demo[demo.horizon_days==h]
        dates=sorted(part.target_date.astype(str).unique());day=st.selectbox('검증 대상일',dates,index=max(0,len(dates)-1),key='fold_demo_day')
        day_data=part[part.target_date.astype(str)==day];models=sorted(day_data.loc[~day_data.series.isin(['실제 SMP','기준선 SMP']),'series'].unique())
        preferred={1:'V11_ExtraTrees_fold_refit',2:'ml_anchor_blend',3:'ml_anchor_blend'}.get(h)
        default=models.index(preferred) if preferred in models else 0
        model=st.selectbox('모델',models,index=default,key='fold_demo_model')
        metadata=day_data.iloc[0]
        st.caption(f"유형: 시점 재현 fold 검증 · 학습 종료 {metadata.train_end} · 보정 {metadata.calibration_start}~{metadata.calibration_end} · 평가 {metadata.validation_start}~{metadata.validation_end} · vintage 확인: 아니오")
        actual=day_data[day_data.series=='실제 SMP'].sort_values('hour');baseline_row=day_data[day_data.series=='기준선 SMP'].sort_values('hour');pred=day_data[day_data.series==model].sort_values('hour')
        chart_data=pd.DataFrame({'timestamp':pd.date_range(pd.Timestamp(day,tz='Asia/Seoul'),periods=24,freq='h'),'실제 SMP':actual.smp.to_numpy(),'기준선 SMP':baseline_row.smp.to_numpy(),'선택 모델 SMP':pred.smp.to_numpy()})
        line(chart_data,['실제 SMP','기준선 SMP','선택 모델 SMP'],f'D+{h} {day} 검증 fold 예측과 실제값','원/kWh')
        e=pred.smp.to_numpy()-actual.smp.to_numpy();mae=float(np.mean(np.abs(e)));rmse=float(np.sqrt(np.mean(e**2)));be=baseline_row.smp.to_numpy()-actual.smp.to_numpy();bmae=float(np.mean(np.abs(be)));brmse=float(np.sqrt(np.mean(be**2)))
        a,b,c,d=st.columns(4);a.metric('모델 MAE',f'{mae:.2f}',f'{mae-bmae:+.2f} vs 기준선');b.metric('모델 RMSE',f'{rmse:.2f}',f'{rmse-brmse:+.2f} vs 기준선');c.metric('표본','24시간');d.metric('운영 검증','아님')
        with st.expander('시간대별 오차 표'):
            detail=chart_data.copy();detail['모델 오차']=e;detail['기준선 오차']=be;st.dataframe(detail,hide_index=True)
        st.subheader('동일 EV 조건의 G2V 충전 추천 재생')
        ev,start,end=conditions('fold_demo_'+str(h)+'_'+day,day)
        pred_prices=chart_data[['timestamp','선택 모델 SMP']].rename(columns={'선택 모델 SMP':'price'})
        base_prices=chart_data[['timestamp','기준선 SMP']].rename(columns={'기준선 SMP':'price'})
        actual_prices=chart_data[['timestamp','실제 SMP']].rename(columns={'실제 SMP':'price'})
        show_plans({'선택 연구 모델':pred_prices,'기준선':base_prices},ev,start,end,actual_reference=actual_prices,actual_label='fold 실제 SMP')
        st.caption('regret은 같은 대상일과 화면 입력 EV 조건에서 fold 실제 SMP를 기준으로 계산합니다. 이는 retrospective simulation이며 운영 regret 표본은 아닙니다.')
elif page=='EV 충전 추천':
    st.info('전체 연결기간의 공식 SMP를 우선하며, 없으면 검증된 정시 저장 AI를 선택합니다. 현재 조건의 계획 재계산이며 원래 동결한 결정을 바꾸지 않습니다.')
    ev,start,end=conditions('live',pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y-%m-%d'),True)
    selected=select_auto(official,items,start,end)
    if start<pd.Timestamp.now(tz='Asia/Seoul'):st.caption('과거 시간이 포함된 조건입니다. 과거 시연·재계산이며 새 미래 사전 계획이 아닙니다.')
    if selected['mode']=='waiting':st.warning('수신 대기 · '+selected['reason'])
    else:
        st.success('선택 자료: '+selected['label']);st.caption(selected['reason'])
        if selected['records']:st.dataframe(pd.DataFrame(selected['records']),hide_index=True)
        st.caption('저장된 연구 예측을 활용한 충전 시뮬레이션이며 운영 모델 교체가 아닙니다. G2V 기본 권장.')
        show_plans({selected['label']:selected['prices']},ev,start,end)
elif page=='과거 예측 시연':
    research=OUTPUT/'research'
    prices=research_csv(research/'final_saved_comparison_prices.csv')
    if prices is None or prices.empty:
        st.error('보존된 과거 예측 및 비교 가격 자료가 없습니다.')
    else:
        st.warning('과거 자료 시연입니다. 선택한 예측은 저장된 발급 예측 또는 과거 재현이며, 신규 실시간 발급·운영 권장은 아닙니다.')
        cohort_labels={'saved_issue_forecasts':'실제 저장 예측 (V11–V20)','V5_saved_92day_test':'V5 보존 92일 테스트','V20_42day_historical_replay':'V20 42일 과거 재현'}
        available=prices.demo_cohort.dropna().astype(str).unique().tolist()
        options=[x for x in cohort_labels if x in available]
        a,b=st.columns(2)
        selected_cohort=a.selectbox('시연 자료 유형',options,format_func=lambda x:cohort_labels[x],key='demo_cohort')
        cohort_data=prices[prices.demo_cohort.astype(str).eq(selected_cohort)]
        horizons=sorted(int(x) for x in cohort_data.horizon_days.dropna().unique())
        selected_horizon=b.selectbox('예측 기간',horizons,format_func=lambda x:f'D+{x}',key='demo_horizon')
        cohort_data=cohort_data[cohort_data.horizon_days.eq(selected_horizon)]
        days=sorted(cohort_data.target_date.astype(str).unique())
        preferred_day='2026-10-18' if '2026-10-18' in days else ('2026-10-19' if '2026-10-19' in days else days[-1])
        day=st.selectbox('예측 대상일',days,index=days.index(preferred_day),key='demo_day')
        candidates=cohort_data[cohort_data.target_date.astype(str).eq(day)]
        ids=sorted(candidates.forecast_id.astype(str).unique())
        preferred_id=next((x for x in ids if 'lightgbm_peak' in x),next((x for x in ids if 'V17_joint' in x),ids[0]))
        forecast_id=st.selectbox('예측 모델/발급 기록',ids,index=ids.index(preferred_id),key='demo_forecast')
        selected=candidates[candidates.forecast_id.astype(str).eq(forecast_id)].copy()
        st.caption(f"{cohort_labels[selected_cohort]} · D+{selected_horizon} · {day} · 자료 유형: {selected.evaluation_type.iloc[0]}")
        chart=alt.Chart(selected).mark_line(point=True).encode(x=alt.X('hour:O',title='대상일 시간(KST)'),y=alt.Y('smp:Q',title='SMP(원/kWh)'),color=alt.Color('series:N',title='구분'),tooltip=[alt.Tooltip('hour:O',title='시각'),alt.Tooltip('smp:Q',title='SMP',format='.2f'),'series:N','evaluation_type:N']).properties(title=forecast_id,height=340)
        st.altair_chart(chart,width='stretch')
        if 'risk_level' in selected and selected.risk_level.notna().any():
            st.dataframe(selected[selected.risk_level.notna()][['hour','risk_level']].rename(columns={'hour':'시각','risk_level':'저장된 위험도'}),hide_index=True)
        elif 'spike_probability' in selected and selected.spike_probability.notna().any():
            risk=selected[selected.spike_probability.notna()][['hour','spike_probability']].rename(columns={'hour':'시각','spike_probability':'급등 확률'})
            st.dataframe(risk,hide_index=True)
        else:
            st.caption('선택 모델은 이 보존 예측에서 시간대별 가격 날씨/급등 위험도를 제공하지 않았습니다.')
        if 'weather_forecast_used' in selected and selected.weather_forecast_used.notna().any():
            st.caption('예측 기록의 기상 입력 사용 표기: '+', '.join(sorted(selected.weather_forecast_used.dropna().astype(str).unique())))
        st.subheader('입력 조건에 따른 G2V 충전 추천 및 비용 재생')
        ev,start,end=conditions('demo_'+selected_cohort+str(selected_horizon)+day,day)
        predicted=selected[selected.series.eq('예측 SMP')][['timestamp','smp']].rename(columns={'smp':'price'}).copy()
        if predicted.empty:
            predicted=selected[selected.series.eq('기준선 SMP')][['timestamp','smp']].rename(columns={'smp':'price'}).copy()
        predicted['timestamp']=pd.to_datetime(predicted.timestamp,format='mixed')
        if predicted.timestamp.dt.tz is None:predicted.timestamp=predicted.timestamp.dt.tz_localize('Asia/Seoul')
        else:predicted.timestamp=predicted.timestamp.dt.tz_convert('Asia/Seoul')
        show_plans({'선택 예측 SMP':predicted},ev,start,end)
        st.caption('표시되는 실제가격은 보존된 공식 제공 SMP입니다. 확정 정산/사후 정정 여부는 별도 확인되지 않아, 비용과 regret는 동일 EV 조건의 사후 가격 재생값입니다.')
elif page=='연구 예측 정확도':
    research=OUTPUT/'research'
    st.subheader('보존 예측 종합 비교 (V5–V20)')
    saved_summary=research_csv(research/'final_saved_comparison_model_summary.csv')
    if saved_summary is not None and len(saved_summary):
        cols=[c for c in ['cohort','version','model','horizon_days','days','MAE','RMSE','G2V_regret','V2G_regret','input_vintage_verified','model_available_at_issue','model_available_at_reference_origin','issue_delay_minutes'] if c in saved_summary]
        st.dataframe(saved_summary[cols].rename(columns={'cohort':'평가 구분','version':'버전','model':'모델','horizon_days':'예측기간(D+)','days':'평가일수','MAE':'MAE(원/kWh)','RMSE':'RMSE(원/kWh)','G2V_regret':'G2V regret(원)','V2G_regret':'가상 V2G regret(원)','input_vintage_verified':'입력 vintage 확인','model_available_at_issue':'발급 시 모델 가용','model_available_at_reference_origin':'기준 발급시 모델 가용','issue_delay_minutes':'기준시각 지연(분)'}),hide_index=True)
        cohort_options=saved_summary.cohort.dropna().astype(str).unique().tolist()
        if cohort_options:
            selected_cohort=st.selectbox('regret 비교 집단',cohort_options,key='saved_eval_regret_cohort')
            regret=saved_summary[saved_summary.cohort.astype(str)==selected_cohort]
            regret=regret.dropna(subset=['G2V_regret'])
            if len(regret):
                c=alt.Chart(regret).mark_bar().encode(x=alt.X('model:N',title='모델',sort='-y'),y=alt.Y('G2V_regret:Q',title='평균 G2V regret(원)'),color='horizon_days:N',xOffset='horizon_days:N',tooltip=['version','model','horizon_days','days','G2V_regret']).properties(title='동일 평가집단의 모델·기간별 G2V regret',height=290)
                st.altair_chart(c,width='stretch')
    else:st.info('기존 저장 예측 비교 결과가 없습니다. 별도 분석을 실행하면 표시됩니다.')
    paired=research_csv(research/'final_saved_comparison_shared_dates.csv')
    if paired is not None and len(paired):
        st.caption('아래 표는 저장 예측과 발급시점에 이용 가능했던 같은 시간대 기준선을 동일 대상일에서 비교합니다. AI 예측은 기존 저장 평가 점수를 재사용했습니다.')
        cols=[c for c in ['version','model','horizon_days','target_date','MAE','RMSE','G2V_regret','input_vintage_verified','model_available_at_reference_origin','issue_delay_minutes'] if c in paired]
        st.dataframe(paired[cols].rename(columns={'version':'버전','model':'모델','horizon_days':'예측기간(D+)','target_date':'대상일','MAE':'MAE(원/kWh)','RMSE':'RMSE(원/kWh)','G2V_regret':'G2V regret(원)','input_vintage_verified':'입력 vintage 확인','model_available_at_reference_origin':'기준 발급시 모델 가용','issue_delay_minutes':'기준시각 지연(분)'}),hide_index=True)
    prices=research_csv(research/'final_saved_comparison_prices.csv')
    if prices is not None and len(prices):
        st.subheader('SMP 실제값과 예측값')
        ids=sorted(prices.forecast_id.dropna().astype(str).unique())
        preferred=next((i for i in ids if i.startswith('V12 · lightgbm_peak · D+2 · 2026-10-18')),
                       next((i for i in ids if i.startswith('V5_profile_hgb_blend')),
                            ids[0] if ids else None))
        choice=st.selectbox('그래프로 볼 저장 예측·기간·대상일',ids,index=ids.index(preferred) if preferred in ids else 0,key='saved_eval_price_chart')
        chart_data=prices[prices.forecast_id.astype(str)==choice]
        chart=alt.Chart(chart_data).mark_line(point=True).encode(x=alt.X('hour:O',title='대상일 시간(KST)'),y=alt.Y('smp:Q',title='SMP(원/kWh)'),color=alt.Color('series:N',title='구분'),tooltip=['forecast_id','series','hour','smp','evaluation_type']).properties(title=choice,height=320)
        st.altair_chart(chart,width='stretch')
        st.caption('저장 예측·V5 테스트·V20 재현을 선택해 확인할 수 있습니다. 대상 가격은 보존된 공식 제공가격 또는 V5 저장 실적이며, 확정 정산가격/사후 정정 여부는 확인되지 않았습니다.')
    else:st.info('저장 예측 시계열 데이터가 없습니다.')
    manifest=research_csv(research/'final_saved_comparison_manifest.csv')
    if manifest is not None and len(manifest):
        excluded=int(manifest.status.eq('excluded_target_price_known_at_issue').sum())
        retry=int(manifest.status.eq('duplicate_retry_excluded').sum())
        st.caption(f'사전가격 기확보로 제외 {excluded}건 · 재시도 중복 제외 {retry}건 · 저장 이력 {len(manifest)}건. 모델/입력 vintage가 입증되지 않은 결과는 참고 비교로만 해석합니다.')
        with st.expander('저장 예측별 발급·입력 검증 내역'):
            cols=[c for c in ['version','file','model','target_date','horizon_days','issued_at','issue_delay_minutes','issue_known_target_rows','inputs_vintage_verified','model_available_at_reference_origin','status'] if c in manifest]
            st.dataframe(manifest[cols].rename(columns={'version':'버전','file':'예측 파일','model':'모델','target_date':'대상일','horizon_days':'D+','issued_at':'저장 발급시각','issue_delay_minutes':'기준 지연(분)','issue_known_target_rows':'발급시 이미 알려진 대상일 SMP 행','inputs_vintage_verified':'입력 시각 확인','model_available_at_reference_origin':'기준시 모델 가용','status':'평가 상태'}),hide_index=True)
    st.subheader('실제 운영 연구 예측')
    st.caption('정식 발급 gate와 독립적으로 저장한 예측만 집계합니다. 실제 공식 SMP 24시간 확보 전에는 점수 계산을 보류합니다.')
    op=research_csv(research/'operational_metrics.csv')
    op=op if op is not None else pd.DataFrame()
    if len(op):
        st.dataframe(op.rename(columns={'model':'모델','horizon_days':'예측기간(D+)','target_date':'대상일','status':'상태','MAE':'MAE(원/kWh)','RMSE':'RMSE(원/kWh)','G2V_regret':'G2V regret(원)','V2G_regret':'가상 V2G regret(원)','official_hours':'실제 SMP 확보시간'}),hide_index=True)
    else:st.info('아직 연구 예측 기록이 없습니다. 다음 실제 발급 시각에 gate 결과와 별도로 생성됩니다.')
    actual_hourly=research_csv(research/'operational_hourly_summary.csv')
    if actual_hourly is not None and len(actual_hourly):
        c=alt.Chart(actual_hourly).mark_line(point=True).encode(x=alt.X('hour:O',title='대상일 시간(KST)'),y=alt.Y('MAE:Q',title='시간대별 MAE(원/kWh)'),color='model:N',detail='horizon_days:N',tooltip=['model','horizon_days','hour','samples','MAE','bias']).properties(title='실제 운영 연구 예측 · 시간대별 오차',height=280)
        st.altair_chart(c,width='stretch')
    else:st.caption('실제 운영 표본이 모이면 모델·D+기간별 시간대 오차 그래프가 표시됩니다.')
    duplicate_rows=research_csv(research/'duplicate_forecasts.csv')
    duplicate_count=0 if duplicate_rows is None else len(duplicate_rows)
    st.caption(f'재시도 중복 {duplicate_count}건 제외 · 발급 시점에 대상일 SMP가 이미 확보된 예측은 사전예측 성능에서 제외합니다.')
    st.subheader('과거 시점 재현 백테스트')
    bt=research_csv(research/'backtest_42_summary.csv')
    if bt is not None and len(bt):
        st.dataframe(bt.rename(columns={'model':'모델','horizon_days':'예측기간(D+)','days':'평가일수','MAE':'일평균 MAE(원/kWh)','RMSE':'일평균 RMSE(원/kWh)','G2V_regret':'G2V regret 평균(원)','V2G_regret':'가상 V2G regret 평균(원)'}),hide_index=True)
        plot=alt.Chart(bt).mark_bar().encode(x=alt.X('horizon_days:O',title='예측기간(D+)'),y=alt.Y('MAE:Q',title='일평균 MAE(원/kWh)'),color='model:N',xOffset='model:N',tooltip=['model','horizon_days','days','MAE','RMSE','G2V_regret','V2G_regret']).properties(title='과거 재현 · 모델 및 예측기간별 가격 오차',height=300)
        st.altair_chart(plot,width='stretch')
        bh=research_csv(research/'backtest_42_hourly_errors.csv')
        if bh is not None and len(bh):
            bh=bh.groupby(['model','horizon_days','hour'],as_index=False).agg(samples=('absolute_error','count'),MAE=('absolute_error','mean'),bias=('error','mean'))
            c=alt.Chart(bh).mark_line().encode(x=alt.X('hour:O',title='대상일 시간(KST)'),y=alt.Y('MAE:Q',title='시간대별 MAE(원/kWh)'),color='model:N',detail='horizon_days:N',tooltip=['model','horizon_days','hour','samples','MAE','bias']).properties(title='과거 재현 · 시간대별 오차',height=280)
            st.altair_chart(c,width='stretch')
        st.caption('42개 공통 주간 대상일 기준입니다. 입력 자료의 당시 수신시각과 모델 체크포인트 가용성을 확인하지 못해 검증된 운영 성능이 아닌 과거 재현 결과로만 표시합니다. D+1은 기준선, D+2/D+3은 V17과 기준선을 동일 날짜·EV 조건으로 비교합니다.')
    else:st.info('백테스트 결과가 아직 생성되지 않았습니다.')
    st.subheader('보존 V5 D+1 기준 결과')
    v5=research_csv(research/'v5_d1_reference_summary.csv')
    if v5 is not None and len(v5):
        st.dataframe(v5.rename(columns={'model':'모델','horizon_days':'예측기간(D+)','days':'평가일수','MAE':'일평균 MAE(원/kWh)','RMSE':'일평균 RMSE(원/kWh)','G2V_regret':'G2V regret 평균(원)','V2G_regret':'가상 V2G regret 평균(원)'}),hide_index=True)
        st.caption('기존 V5 저장 예측과 daily profile 기준선을 동일한 V5 최종 테스트 날짜·EV 조건으로 재평가했습니다. 입력 vintage는 확인되지 않았으며 V20 결과와 기간이 달라 직접적인 우열 판단에는 사용할 수 없습니다.')
    else:st.caption('보존된 V5 최종 테스트 예측을 찾지 못했습니다.')
    status_eval=read(research/'evaluation_status.json',{})
    st.caption(f"평가 갱신 {status_eval.get('at','미확보')} · 실제 운영 평가 {status_eval.get('operational_evaluated',0)}건 · 백테스트 평가일 {status_eval.get('historical_backtest_days',0)}일")
    st.info('V5의 기존 결과와 원본 예측 파일은 보존합니다. 평가 기간과 정보 조건이 다른 결과를 직접적인 성능 개선으로 해석하지 않습니다.')
else:
    st.subheader('실제 전향 검증 · 과거 재생과 구분')
    a,b,c=st.columns(3);a.metric('공식 제공가격 비교 대상일',len(status.get('official_comparison_target_dates',[])));b.metric('대상일 경과 완료',status.get('target_day_elapsed_days',0));c.metric('급등 발생 대상일',status.get('spike_target_days',0))
    st.caption('상태 갱신: '+str(status.get('at','미확보')));st.caption('보조 대상일을 포함한 비교입니다. 공식 제공가격 수신과 날짜 경과·최종 정정통계 확인은 다릅니다. 신규 실제 발급 '+str(status.get('V19_new_actual_records',0))+'건 · 정시 적격 '+str(status.get('V19_on_time_eligible',0))+'건. 신규 학습·운영 모델 교체 없음.')
    st.subheader('공식가격 공개상태와 보류 원인')
    reasons={'official_price_already_observed':'공식가격 기확보 · 새 사전 예측 발급 금지','unverified_publication_general_public_unverified_all_zero':'홈페이지 24시간 모두 0원 · 미공개인지 확인할 수 없어 보류'}
    for r in status.get('readiness',[]):
        reason=reasons.get(r['reason'],'공개상태 확인 필요 · '+r['reason']);st.warning(f"{r['target_date']} D-{r['horizon']}: {reason}")
    st.caption('시각이 오래된 조회·실패·모두 0원 응답은 정상 미공개 근거가 아닙니다. 안전검사를 해제하지 않습니다.')
    if (OUTPUT/'publication.csv').exists():
        f=csv(OUTPUT/'publication.csv');readiness_days={r['target_date'] for r in status.get('readiness',[])}
        recent_days=f.assign(_receipt=pd.to_datetime(f.latest_check,errors='coerce')).groupby('target_date')['_receipt'].max().nlargest(5).index
        f=f[f.target_date.isin(readiness_days|set(recent_days))]
        st.dataframe(f.rename(columns={'target_date':'대상일','source':'공식 경로','market':'시장','latest_state':'최근 응답 상태','latest_check':'실제 수신시각','first_normal_receipt':'최초 정상가격 수신','last_strict_normal_nonobservation':'마지막 정상 미관측','zero_response_count':'0원 응답 수','zero_conflicts_with_already_acquired_other_source':'다른 경로 기확보와 불일치'}),hide_index=True)
    st.subheader('정시 준비·발급과 최근 수집 상태')
    prep_records=preparation_status.get('records',[]);prep_ok=sum(r.get('status')=='prepared_not_issued' for r in prep_records);prep_failed=sum(r.get('status')=='failed' for r in prep_records)
    publish_records=publisher_status.get('records',[]);publish_created=sum(r.get('status')=='created' for r in publish_records);publish_skipped=sum(r.get('status')=='skipped' for r in publish_records)
    x,y,z=st.columns(3);x.metric('최근 준비본',f"준비 {prep_ok} · 실패 {prep_failed}");y.metric('최근 발급 확인',f"발급 {publish_created} · 보류 {publish_skipped}");z.metric('발급 확인 완료시각',publisher_status.get('completed_at','미실행'))
    if preparation_status:
        st.caption(f"준비 상태: {preparation_status.get('mode','미확보')} · 기준시각 {preparation_status.get('reference_origin','미확보')} · 준비본과 실제 발급 예측은 별도입니다.")
    if publisher_status:
        st.caption(f"발급 작업: {publisher_status.get('mode','미확보')} · 기준시각 {publisher_status.get('reference_origin','미확보')} · 실제 생성 파일 {publish_created} · 네트워크 조회 {publisher_status.get('network_calls',0)}회")
        if publisher_status.get('publication_refresh'):
            refresh=publisher_status['publication_refresh'];st.write('발급 직전 공개상태 재확인:',refresh.get('status'),refresh.get('record_file','기록 경로 미확보'))
            decisions=refresh.get('decisions',[])
            if decisions:st.dataframe(pd.DataFrame([dict(대상일=d.get('target_date'),결정시각=d.get('decided_at'),발급적격=d.get('eligible'),판정근거=d.get('reason')) for d in decisions]),hide_index=True)
            checks=refresh.get('checks',[])
            if checks:st.dataframe(pd.DataFrame([dict(대상일=r.get('target_date'),경로=r.get('source'),시장=r.get('market'),요청시작=r.get('started_at'),응답수신=r.get('checked_at'),응답=r.get('status'),요약=r.get('response_summary'),오류=r.get('response_error')) for r in checks]),hide_index=True)
        elif publisher_status.get('mode')=='LIVE_LOCAL_PUBLISH_CHECK':
            st.info('이번 발급 확인은 새 공개상태 재조회 없이 기존 표본만 사용했습니다. 다음 23:02 경로에서 KPX와 필수 육지 경로를 재조회하도록 설정합니다.')
    if status.get('scheduled_slots'):
        f=pd.DataFrame(status['scheduled_slots']);f['status']=f.status.replace({'not_due':'실제 시각 도래 전','worker_observed_NOT_issuance_success':'실행 관측 · 발급 성공과 별도','awaiting_grace':'실행 확인 대기','missing_unverified_sleep_or_failure':'누락 · 종료/절전/오류 원인 미확정'});st.dataframe(f.rename(columns={'slot':'단계','due':'예정시각','status':'확인 상태','worker_observations':'실행 관측 수'}),hide_index=True)
    observer=status.get('observer',{});st.caption('최근 수집 시작: '+str(observer.get('started_at','미확보'))+' / 완료: '+str(observer.get('completed_at','미확보')))
    if observer.get('stages'):st.dataframe(pd.DataFrame(observer['stages']).rename(columns={'stage':'단계','status':'결과','seconds':'소요시간(초)','reason':'실패 사유'}),hide_index=True)
    st.warning('23시 실제 발급 여부는 원기록으로 확인하며 실행 관측과 성공을 구분합니다. 노트북 종료·절전에서는 실행을 보장하지 않습니다. 충분한 신규 독립 전향 표본이 없으므로 모델 우위를 확정하지 않습니다. G2V 기본 권장, V2G는 가상 정산 연구입니다.')
    with st.expander('준비본·발급 결과·시스템 상세 기록'):
        st.write('준비본은 정식 발급 예측이 아닙니다.');st.json(preparation_status);st.json(publisher_status);st.json(status)
