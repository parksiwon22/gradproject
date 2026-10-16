"""Independent local V15 price/peak/EV economics presentation."""
from datetime import datetime,time,timedelta
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,streamlit as st,altair as alt
from v11.store import connect,latest
from v12.research import HOME as V12
from v14.research import HOME
from v12.ev import EV,optimize,baseline,score,regret
from v12.records import write_once
from v15.research import HOME as V15

st.set_page_config(page_title='전력예보 V15 · EV 경제성',page_icon='⚡',layout='wide')
st.markdown('''<style>.block-container{max-width:1300px;padding-top:2rem}div[data-testid="stMetric"]{background:#edf4fa;border-radius:14px;padding:16px}h1{letter-spacing:-.04em}</style>''',unsafe_allow_html=True)
st.title('⚡ 전력예보 V15')
st.caption('가격 예측 → 급등 추종 → EV 의사결정 · 로컬 연구 화면 / 가상 SMP 연동 정산')

def read(path,default=None): return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default
def csv(path): return pd.read_csv(path) if path.exists() else pd.DataFrame()
db=connect(); official=latest(db,'land'); db.close()
status=read(HOME/'study/selection_before_audit.json',{})
forward=[]
for base,prefix in [(V12,'V12'),(HOME,'V14'),(V15,'V15')]:
    for path in (base/'forecasts').glob('*.json'):
        r=read(path); forward.append((prefix,path,r))
independent={r['target_date'] for _,_,r in forward}
evaluations=read(V12/'evaluation_index.json',[])+read(HOME/'evaluation_index.json',[])+read(V15/'evaluation_index.json',[])
evaluated={r['target_date'] for r in evaluations if r.get('status')=='evaluated'}
a,b,c=st.columns(3)
a.metric('최근 확보 공식가격 대상',str(official.timestamp.max().date()) if len(official) else '미확보')
b.metric('전향 대상 / 평가 가능 날짜',str(len(independent))+' / '+str(len(evaluated))+'일')
c.metric('V15 검증 통과 후보',len(list((V15/'candidates').glob('*.joblib'))))
st.caption('여러 모델의 같은 날짜 예측은 독립 표본 한 날입니다. 공식가격 수신은 거래일 경과와 별개이며 이후 정정될 수 있습니다.')
if len(official): st.caption('공식가격 최근 최초 수신시각(UTC): '+str(official.first_seen.max()))

with st.sidebar:
    st.header('비교 조건')
    mode=st.radio('자료 모드',['저장된 전향 예측','과거 검증 시연'])
    st.caption('운영 모델은 유지됩니다. 시연·후향 결과를 실시간 AI 성과로 표시하지 않습니다.')
    if mode=='저장된 전향 예측':
        dates=sorted(independent,reverse=True)
        if not dates: st.info('전향 기록이 없습니다.'); st.stop()
        day=st.selectbox('예측 대상일',dates)
        frames=[]; metadata=[]
        for prefix,path,r in forward:
            if r['target_date']!=day: continue
            part=pd.DataFrame(r['rows']); part.timestamp=pd.to_datetime(part.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
            name=prefix+':'+r['model']; frames.append(part.set_index('timestamp').predicted_smp.rename(name)); metadata.append(dict(모델=name,실제생성=r['issued_at'],입력기준=r['reference_origin'],생성당시공식기확보=r.get('official_smp_known_at_issue',r.get('official_smp_known_at_issue',False))))
        data=pd.concat(frames,axis=1).reset_index(); data=data.merge(official[['timestamp','smp']].rename(columns={'smp':'actual_smp'}),on='timestamp',how='left')
        names=[c for c in data if c not in ('timestamp','actual_smp')]
    else:
        pool=csv(HOME/'study/selection_predictions.csv'); metadata=[]
        if pool.empty: st.info('후보 실험이 실행 중입니다. 검증 예측 파일이 저장되면 시연할 수 있습니다.'); st.stop()
        pool.timestamp=pd.to_datetime(pool.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
        follow=csv(HOME/'chronological_followup/predictions.csv')
        if not follow.empty:
            follow.timestamp=pd.to_datetime(follow.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
            pool=pool.merge(follow[['timestamp','predicted_smp']].rename(columns={'predicted_smp':'gru_chronological7day_pilot'}),on='timestamp',how='left')
        candidates=csv(V15/'shrinkage_followup/winter2025_predictions.csv')
        if not candidates.empty:
            candidates.timestamp=pd.to_datetime(candidates.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
            fields=[c for c in candidates if c.startswith('two_stage_shrink_')]
            candidates=candidates[['timestamp']+fields].rename(columns={c:'V15후보:'+c for c in fields})
            pool=pool.merge(candidates,on='timestamp',how='left')
        day=st.selectbox('2025Q4 검증 날짜',sorted(pool.target_date.unique(),reverse=True))
        data=pool[pool.target_date==day].copy(); names=[c for c in data if c not in ('timestamp','actual_smp','previous_actual_smp','target_date')]
    defaults=([n for n in names if 'hgb' in n.lower()][:1]+[n for n in names if n in status.get('top3',[])][:2]) or names[:2]
    chosen=st.multiselect('그래프·EV 비교 모델',names,default=list(dict.fromkeys(defaults)))
    if not chosen: st.info('비교 모델을 하나 이상 선택하세요.'); st.stop()
    if len(chosen)>4: st.warning('계산·가독성을 위해 처음 네 모델을 사용합니다.'); chosen=chosen[:4]

def chart(frame,models):
    cols=(['actual_smp'] if frame.actual_smp.notna().any() else [])+models
    long=frame[['timestamp']+cols].melt('timestamp',var_name='계열',value_name='SMP(원/kWh)')
    long['계열']=long['계열'].replace({'actual_smp':'공식 실제가격'})
    lines=alt.Chart(long).mark_line(strokeWidth=2).encode(x=alt.X('timestamp:T',title='대상시각(KST)'),y=alt.Y('SMP(원/kWh):Q',scale=alt.Scale(zero=False)),color='계열:N',tooltip=['timestamp:T','계열:N','SMP(원/kWh):Q'])
    peaks=alt.Chart(frame[frame.actual_smp>=150]).mark_point(color='#e44c59',size=90).encode(x='timestamp:T',y='actual_smp:Q')
    st.altair_chart((lines+peaks).properties(height=320),width='stretch')

tabs=st.tabs(['가격·모델 비교','급등 사례 분석','EV 충방전 계획','기존 경제성·연구 한계','V15 사건·정책 검증'])
with tabs[0]:
    st.subheader('공식가격과 저장된 AI 추정값')
    if mode=='과거 검증 시연': st.warning('이미 열람한 2025Q4 후향 검증 시연입니다. 새 독립 테스트가 아닙니다.')
    elif data.actual_smp.notna().all(): st.info('이 대상일의 공식 SMP가 확보됐습니다. 현재 의사결정에는 공식가격을 사용할 수 있으며, 저장된 사전 예측은 별도로 대조합니다.')
    else: st.info('미확보 실제가격 구간은 평가 대기입니다. 공식 전 경로의 최초 공개시각을 인증한 것은 아닙니다.')
    chart(data,chosen)
    if metadata: st.dataframe(pd.DataFrame(metadata),hide_index=True)
    table=csv(HOME/'study/selection_metrics.csv')
    followmetrics=csv(HOME/'chronological_followup/metrics.csv')
    if not followmetrics.empty: table=pd.concat([table,followmetrics],ignore_index=True)
    if not table.empty:
        st.caption('아래 표는 동일 D-2 입력 기준·2025Q4 검증 조건입니다. V5 D-1 결과와 직접 순위를 매기지 않습니다. 외생변수 후보는 과거 수신 판본이 미확인인 후향 연구입니다.')
        st.dataframe(table[[c for c in ['name','operational_candidate','MAE','peak_MAE','peak_F1','G2V_regret','V2G_regret','reversal_count'] if c in table]].rename(columns={'name':'후보','operational_candidate':'가격 전용 입력 적격','MAE':'MAE(원/kWh)','peak_MAE':'고가 MAE(원/kWh)','peak_F1':'고가 F1','G2V_regret':'G2V regret(가상 원/일)','V2G_regret':'V2G regret(가상 원/일)','reversal_count':'반전일 수'}),hide_index=True)
    st.caption('전향 평가가 적고 급등 위험 확률은 검증되지 않았습니다. 가격범위·위험 표시는 보장이 아닙니다.')
with tabs[1]:
    st.subheader('급등 추종과 과소예측')
    chart(data,chosen)
    rows=[]
    if data.actual_smp.notna().all():
        y=data.actual_smp.to_numpy(); high=y>=150
        for name in chosen:
            p=data[name].to_numpy(); alarms=p>=150; tp=int(np.sum(high&alarms)); fp=int(np.sum(~high&alarms)); fn=int(np.sum(high&~alarms))
            rows.append(dict(모델=name,고가시간=int(high.sum()),고가MAE=float(np.mean(abs(p[high]-y[high]))) if high.any() else None,고가Bias=float(np.mean(p[high]-y[high])) if high.any() else None,F1=2*tp/max(1,2*tp+fp+fn),허위경보=fp,최고가격오차=float(abs(y.max()-p.max())),피크시간오차=abs(int(y.argmax())-int(p.argmax()))))
        st.dataframe(pd.DataFrame(rows),hide_index=True)
        if not high.any(): st.caption('150원 이상 실제 시간이 없어 이 날의 고가 MAE는 산출할 수 없습니다.')
    else: st.info('실제값 미완료: 피크 성능을 생성하지 않습니다.')
    st.caption('붉은 점: 실제 150원 이상. 희소 사례와 가격체제 변화 때문에 고정 임계값 F1만으로 모델을 선정하지 않습니다.')
with tabs[2]:
    st.subheader('EV 조건과 모델별 의사결정')
    st.caption('SMP 연동 가상 비용입니다. 국내 소매요금·실증 V2G 판매수익이 아닙니다.')
    st.caption('현재 입력으로 계획을 재계산합니다. 공식가격 공개 전에 불변 저장된 원 EV 계획과 구분하며, 이 화면의 사후 시연을 새 전향 성과로 계산하지 않습니다.')
    c1,c2,c3=st.columns(3)
    capacity=c1.number_input('배터리 용량(kWh)',10.,200.,60.); initial=c2.number_input('초기 SOC(%)',0.,100.,30.); target=c3.number_input('목표 SOC(%)',0.,100.,80.)
    with st.expander('연결시간·전력·효율·SOC 한계·열화 비용',expanded=False):
        full=st.checkbox('선택 날짜 전체 24시간 연결(연구 조건)',value=True)
        begin_date=st.date_input('연결 시작일',pd.Timestamp(day).date()); end_date=st.date_input('연결 종료일',pd.Timestamp(day).date())
        begin_time=st.time_input('연결 시작',time(18)); end_time=st.time_input('연결 종료',time(23,59))
        charge=st.number_input('충전전력(kW)',.1,100.,7.); discharge=st.number_input('방전전력(kW)',0.,100.,7.)
        eff=st.number_input('충전효율(%)',1.,100.,90.); deff=st.number_input('방전효율(%)',1.,100.,90.)
        minimum=st.number_input('SOC 하한(%)',0.,100.,20.); maximum=st.number_input('SOC 상한(%)',0.,100.,90.)
        wear=st.number_input('열화비용(원/배터리 처리kWh)',0.,1000.,5.)
        policy=st.selectbox('V2G 의사결정 정책',['일반 예측가격','방전 예상수입 15% 보수적 계획','예상 이익 100원 미만이면 G2V 유지'])
    price_mode=st.radio('계획에 쓸 가격',['저장 예측으로 모델 비교','확보된 공식가격'],horizontal=True)
    if st.button('모델별 G2V · 가상 V2G 비교',type='primary'):
        try:
            start=pd.Timestamp(data.timestamp.iloc[0]) if full else pd.Timestamp(datetime.combine(begin_date,begin_time),tz='Asia/Seoul'); end=pd.Timestamp(data.timestamp.iloc[-1])+pd.Timedelta(hours=1) if full else pd.Timestamp(datetime.combine(end_date,end_time),tz='Asia/Seoul')
            if end<=start: raise ValueError('연결 종료시각은 시작 이후여야 합니다. 밤을 넘기면 종료일도 다음 날로 설정하세요.')
            times=pd.date_range(start.floor('h'),end.ceil('h')-pd.Timedelta(hours=1),freq='h')
            actual_plan=official.set_index('timestamp').smp.reindex(times)
            ev=EV(capacity,initial,target,charge,discharge,eff/100,deff/100,minimum,maximum,wear,1)
            costs=[]; power=[]; soc=[]; saved={}; used_prices={}
            for name in chosen:
                if price_mode=='확보된 공식가격': p=actual_plan.to_numpy()
                elif mode=='과거 검증 시연': p=pool.set_index('timestamp')[name].reindex(times).to_numpy()
                else:
                    parts=[]
                    for prefix,path,record in forward:
                        if prefix+':'+record['model']==name:
                            part=pd.DataFrame(record['rows']); part.timestamp=pd.to_datetime(part.timestamp,utc=True).dt.tz_convert('Asia/Seoul'); parts.append(part[['timestamp','predicted_smp']])
                    p=pd.concat(parts).drop_duplicates('timestamp').set_index('timestamp').predicted_smp.reindex(times).to_numpy()
                if not np.isfinite(p).all(): raise ValueError('선택한 연결시간의 가격이 미확보입니다. 예측 모드나 다른 날짜를 선택하세요.')
                used_prices[name]=[dict(timestamp=t.isoformat(),value=float(value)) for t,value in zip(times,p)]
                g=optimize(p,times,ev,start,end); planner=EV(**{**ev.__dict__,'sell_ratio':.85 if policy.startswith('방전') else 1}); v=optimize(p,times,planner,start,end,True)
                if policy.startswith('예상') and g['net_cost']-v['net_cost']<100: v=g
                plans=[('즉시충전',baseline(p,times,ev,start,end)),('균등충전',baseline(p,times,ev,start,end,True)),('G2V',g),('가상 V2G',v)]
                for label,plan in plans:
                    expected=plan['net_cost']; actual=None; regret_value=None
                    if 'schedule' in plan:
                        f=plan['schedule']; expected=score(p,f.charge_kw.to_numpy(),f.discharge_kw.to_numpy(),f.connected_hours.to_numpy(),ev)['net_cost']
                        q=f.copy(); q['계열']=name+' / '+label; power.append(q[['timestamp','계열','charge_kw','discharge_kw']]); soc.append(q[['timestamp','계열','soc_end']])
                        if actual_plan.notna().all():
                            r=regret(plan,actual_plan,times,ev,start,end,label=='가상 V2G'); actual=r['realized_cost']; regret_value=r['regret']
                        saved[name+'_'+label]={**{k:value for k,value in plan.items() if k!='schedule'},'schedule':f.to_dict('records')}
                    elif actual_plan.notna().all(): actual=baseline(actual_plan,times,ev,start,end,label=='균등충전')['net_cost']
                    costs.append(dict(모델=name,방식=label,예상순비용=expected,실제순비용=actual,regret=regret_value,종료SOC=plan.get('ending_soc',max(initial,target))))
            st.caption('금액은 선택 연결기간 합계의 가상 원입니다. regret은 각 G2V/V2G 제약에 맞는 사후 최적 비용과의 차이입니다.')
            st.dataframe(pd.DataFrame(costs),hide_index=True)
            st.line_chart(pd.concat(soc).pivot(index='timestamp',columns='계열',values='soc_end'),y_label='SOC(%)')
            pp=pd.concat(power); pp['순충전전력(kW)']=pp.charge_kw-pp.discharge_kw; st.line_chart(pp.pivot(index='timestamp',columns='계열',values='순충전전력(kW)'),y_label='kW')
            folder=V15/'interactive_decisions'; file=folder/(pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%f')+'.json'); write_once(file,dict(at=pd.Timestamp.now(tz='UTC').isoformat(),mode=mode,price_mode=price_mode,target_date=day,start=start.isoformat(),end=end.isoformat(),policy=policy,settlement_assumptions=ev.__dict__,source_prices_sha256=hashlib.sha256(json.dumps(used_prices,sort_keys=True).encode()).hexdigest(),plan_price_inputs=used_prices,official_prices_complete=bool(actual_plan.notna().all()),plans=saved))
            st.caption('사용자 계획은 V14 로컬에 별도 불변 저장했습니다. 실제가격 미확보 시 regret은 평가 대기입니다. 보수적 정책은 보장을 뜻하지 않습니다.')
        except (ValueError,RuntimeError) as exc: st.error(str(exc))
with tabs[3]:
    st.subheader('예측상 이득이 실제 손실로 바뀌는 경우')
    st.caption('기존 V13: 2026H1 후향 분석, 181일 × 네 모델 = 724조합 중 212개 반전. 같은 날짜 네 모델은 독립 표본이 아닙니다.')
    old=csv(HOME.parent/'v13/economics/reversal_summary.csv')
    if not old.empty: st.dataframe(old,hide_index=True)
    reversals=csv(HOME.parent/'v13/economics/v2g_reversals.csv')
    if not reversals.empty:
        selected=st.selectbox('기존 손실 반전 사례',range(len(reversals)),format_func=lambda i:str(reversals.iloc[i]['day'])+' / '+str(reversals.iloc[i]['model']))
        case=reversals.iloc[selected]; l,r=st.columns(2); l.metric('예측상 V2G 절감(가상 원)',round(case.predicted_v2g_saving,2)); r.metric('실제 V2G 절감(음수=추가비용)',round(case.actual_v2g_saving,2))
        oldpred=csv(V12/'study/retrospective_audit_predictions.csv'); g=oldpred[oldpred.target_date==case.day].copy(); g.timestamp=pd.to_datetime(g.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
        chart(g,[case['model']])
        st.caption('예측상 고가였던 시간에 방전한 뒤 실제로 더 비싼 시간에 재충전하면 비용이 역전될 수 있습니다. 같은 예측을 사용해도 방전을 줄이는 정책과 가격 정확도 개선은 구분해야 합니다.')
    policytable=csv(HOME/'study/selection_policy_metrics.csv')
    if not policytable.empty: st.write('V14 2025Q4 동일 검증 조건 · 정책별 비용'); st.dataframe(policytable,hide_index=True)
    policy_audit=csv(HOME/'study/policy_audit_metrics.csv')
    if not policy_audit.empty:
        st.caption('2026H1 후향 정책 감사: 반전 횟수와 함께 방전량·예상이익 발생일·조건부 반전율을 확인합니다. 방전을 거의 하지 않으면 반전도 줄어듭니다.')
        st.dataframe(policy_audit,hide_index=True)
    st.caption('2025Q4와 2026H1은 기간이 달라 반전 건수를 직접 우열 비교하지 않습니다. 후보 모델은 같은 기간의 기준과 비교합니다.')
    st.warning('기상 관측·수요 실적·예비력은 충분히 지연한 과거값만 연구에 사용합니다. 당시 수신 판본이 없는 외생변수 후보는 운영에 승격하지 않습니다. 급등 탐지와 V2G 경제성은 별도 검증 대상입니다.')
    st.write('다음 검증:', '충분한 독립 전향 날짜 축적 · 당시 발표된 기상/수요·발전 가용성 확보 · 운영 모델 자동 교체 없음')

with tabs[4]:
    from v15.presentation import render
    render(official,chart)
