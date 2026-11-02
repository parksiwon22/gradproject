"""Standalone research-only Streamlit app for newly refit SMP model checkpoints."""
import pandas as pd
import numpy as np
import streamlit as st
from forecast import load_prices,forecast,g2v,evaluate
from pathlib import Path

st.set_page_config(page_title="전력예보 - D+1~D+3 연구모델",page_icon="⚡",layout="wide")
st.title("전력예보 | 현재 D+1~D+3 연구모델")
st.warning("연구용 재학습 체크포인트입니다. 기존 275일 fold 모델과 동일한 모델 파일이 아니며, 실운영 성능을 검증한 예측이 아닙니다.")
st.caption("육지 SMP(원/kWh), KST 기준. 모델 예측은 23시 이전 SMP 시간값으로만 계산하며 EV 비용은 도매가격 기반 시뮬레이션입니다.")

@st.cache_data(ttl=180)
def prices():
    return load_prices()

try:
    historic=prices()
except Exception as exc:
    st.error(f"SMP 이력 데이터 로드 실패: {exc}")
    st.stop()
observed_max=historic.timestamp.max()
st.caption(f"보유 가격 시계열: {historic.timestamp.min():%Y-%m-%d} ~ {observed_max:%Y-%m-%d %H:%M} / {len(historic):,}시간")
now=pd.Timestamp.now(tz="Asia/Seoul")
last_origin=now.normalize()+pd.Timedelta(hours=23)
if last_origin>now:last_origin-=pd.Timedelta(days=1)
h=st.selectbox("예측 기간",[1,2,3],format_func=lambda v:f"D+{v}")
default=(last_origin.normalize()+pd.Timedelta(days=h)).date()
target_date=st.date_input("예측 대상일",value=default)
strict=st.checkbox("수신 로그 first_seen이 예측 기준시각 이전인 데이터만 사용",value=True,
    help="이 검사는 현재 수집 로그를 검증할 뿐 공개된 과거 시점의 최초 가격 공표를 입증하지 않습니다.")
st.caption("단순히 시간값이 과거라는 이유만으로 그때 데이터가 공개됐다고 주장할 수 없습니다. 미검증 모드의 예측은 연구용 재현만 가능합니다.")

try:
    frame,meta=forecast(historic,str(target_date),h,strict)
except Exception as exc:
    st.warning(f"현재 조건에서는 예측할 수 없습니다: {exc}")
    st.info("기준시각 23:00의 SMP와 필요한 과거 4주 데이터가 모두 있어야 합니다. 공식 EPSIS 조회는 RUN_WINDOWS 환경에서 Python -m v11.collect --market land 명령으로 갱신할 수 있습니다.")
    st.stop()

st.success(f"D+{h} 예측 완료 (기준시각 {meta['origin']:%Y-%m-%d %H:%M})")
st.caption(f"모델: {meta['family']} / 학습 종료: {meta['trained_through']} / 과거 first_seen 원본 공표시각 검증: 불가")
actual=historic.set_index("timestamp").smp.reindex(frame.timestamp)
chart=pd.DataFrame({"예측 SMP":frame.price.to_numpy(),
     "horizon lag 기준선":frame.baseline.to_numpy()},index=frame.timestamp)
if actual.notna().all():
    chart["사후 수집 SMP"]=actual.to_numpy(float)
st.line_chart(chart)
stats=None
if actual.notna().all():
    stats=evaluate(actual.to_numpy(float),frame.price.to_numpy(float))
    a,b,c=st.columns(3)
    a.metric("MAE",f"{stats['MAE']:.3f} 원/kWh")
    b.metric("RMSE",f"{stats['RMSE']:.3f} 원/kWh")
    c.metric("G2V regret",f"{stats['G2V_regret']:.2f} 원")
else:
    st.info("대상일 24시간 실제가격이 아직 확보되지 않아 MAE/RMSE/regret은 계산하지 않습니다.")
with st.expander("예측값과 기준선 24시간 데이터"):
    st.dataframe(frame,hide_index=True)
    st.download_button("예측 CSV 받기",frame.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"d{h}_forecast_{target_date}.csv",mime="text/csv")
st.subheader("EV 충전시간 및 비용 시뮬레이션")
x,y,z,k=st.columns(4)
capacity=x.number_input("배터리(kWh)",min_value=1.,max_value=200.,value=60.)
initial=y.number_input("현재 SOC(%)",min_value=0.,max_value=100.,value=30.)
target=z.number_input("목표 SOC(%)",min_value=0.,max_value=100.,value=80.)
power=k.number_input("충전 출력(kW)",min_value=.1,max_value=350.,value=7.)
eff=st.slider("충전 효율(%)",50,100,90)/100
try:
    charge,planned_cost=g2v(frame.price.to_numpy(float),capacity,initial,target,power,eff)
    cols=st.columns(2)
    cols[0].metric("예측 SMP 기반 충전비용",f"{planned_cost:.2f} 원")
    cols[1].metric("필요 그리드 충전량",f"{sum(charge):.2f} kWh")
    schedule=pd.DataFrame(dict(timestamp=frame.timestamp,
         충전전력_kW=charge,예측SMP_원_kWh=frame.price.to_numpy(float)))
    if actual.notna().all():
        actual_cost=float(np.dot(actual.to_numpy(float),charge))
        oracle,_=g2v(actual.to_numpy(float),capacity,initial,target,power,eff)
        regret=actual_cost-float(np.dot(actual.to_numpy(float),oracle))
        st.caption(f"사후 실제 제공가격 기준 충전비용: {actual_cost:.2f}원 / regret: {regret:.2f}원")
    st.dataframe(schedule[schedule.충전전력_kW>0.00001],hide_index=True)
    st.bar_chart(schedule.set_index("timestamp")["충전전력_kW"])
except Exception as exc:
    st.error(f"EV 입력 오류: {exc}")
st.caption("실제 차량 충전요금, 세금, 송배전비용을 반영하지 않은 연구용 도매가격 기반 계산입니다.")
st.divider()
st.subheader("기존 275일 fold 연구 결과")
st.caption("아래 표는 과거 각 fold 당시 학습 조건으로 평가한 연구 후보의 결과이며 이 ZIP의 재학습 checkpoint 성능이 아닙니다.")
f=Path(__file__).parent/"docs/smp_improvement_ml_lgbm/final_recommended_models.csv"
if f.exists():
    st.dataframe(pd.read_csv(f)[["horizon_days","selected_model","days","MAE","RMSE","G2V_regret"]],hide_index=True)
