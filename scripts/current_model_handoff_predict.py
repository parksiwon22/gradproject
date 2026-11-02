"""Predict mainland D+1/D+2/D+3 prices using portable refit checkpoints."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from v11.features import build as build_d1
from v17.features import build as build_dx
from v11.store import connect, latest

ROOT=Path(__file__).resolve().parent
KST="Asia/Seoul"

def load_prices():
    db=connect(ROOT/"artifacts"/"v11")
    try: frame=latest(db,"land")
    finally: db.close()
    if frame.empty: raise ValueError("육지 SMP 이력이 없습니다")
    return frame.drop_duplicates("timestamp").sort_values("timestamp")

def forecast(prices,target_date,horizon,require_receipt=False):
    if horizon not in (1,2,3):raise ValueError("D+1~D+3만 지원합니다")
    path=ROOT/"models"/f"d{horizon}_refit.joblib"
    b=joblib.load(path)
    if int(b["horizon_days"])!=horizon:raise ValueError("잘못된 모델 기간입니다")
    target=pd.date_range(pd.Timestamp(target_date,tz=KST),periods=24,freq="h")
    origin=target[0]-pd.Timedelta(days=horizon)+pd.Timedelta(hours=23)
    if pd.Timestamp(b["trained_through"],tz=KST)+pd.Timedelta(hours=23)>=origin:
        raise ValueError("이 모델은 예측 기준시각보다 나중에 학습되었습니다. 2026-10-27 이후 기준시각만 허용됩니다")
    if origin>pd.Timestamp.now(tz=KST):
        raise ValueError("아직 도래하지 않은 23시 기준시각입니다")
    frame=prices.copy()
    frame["timestamp"]=pd.to_datetime(frame.timestamp,utc=True,format="mixed").dt.tz_convert(KST)
    frame=frame[frame.timestamp<=origin].copy()
    receipts=None
    if "first_seen" in frame:
        receipts=pd.to_datetime(frame.first_seen,utc=True,errors="coerce",format="mixed")
    if require_receipt:
        if receipts is None or receipts.isna().any():
            raise ValueError("입력 자료 수신시각을 검증할 수 없습니다")
        frame=frame[receipts<=origin.tz_convert("UTC")].copy()
    original=frame[["timestamp","smp"]].drop_duplicates("timestamp").sort_values("timestamp")
    built=build_d1(original,targets=target,days_ahead=1,include_labels=False) if horizon==1 else build_dx(original,targets=target,days_ahead=horizon,include_labels=False)
    if len(built)!=24:raise ValueError("23시 기준시각 이전 SMP가 불충분하여 24시간 예측이 불가합니다")
    X=built[b["features"]].to_numpy(float)
    if not np.isfinite(X).all():raise ValueError("입력 값에 결측이 있습니다")
    residual=b["model"].predict(X)
    pred_ml=built.daily.to_numpy(float)+residual
    anchor=np.median(np.vstack([original.set_index("timestamp").smp.reindex(target-pd.Timedelta(days=7*k)).to_numpy(float) for k in range(1,5)]),axis=0)
    if not np.isfinite(anchor).all():raise ValueError("같은 요일 anchor 4주치가 부족합니다")
    weight=b["anchor_weight"]
    pred=(1-weight)*pred_ml+weight*anchor
    original_receipt_ok=False
    if receipts is not None:
        # Price source first_seen is arrival into THIS collector, not original publication.
        original_receipt_ok=require_receipt
    result=pd.DataFrame(dict(timestamp=target,price=pred,baseline=built.daily.to_numpy(float),
                             ml_price=pred_ml,anchor_price=anchor))
    return result, dict(horizon=horizon,origin=origin,trained_through=b["trained_through"],
       family=b["family"],receipt_filter_enforced=original_receipt_ok,
       issue_origin_reached=True,mode="RESEARCH_REFIT_NO_OPERATIONAL_VALIDATION")

def g2v(prices,capacity=60.0,initial=30.0,target=80.0,power=7.0,eff=.9):
    values=np.asarray(prices,float)
    if len(values)!=24 or not np.isfinite(values).all():raise ValueError("24시간 SMP가 필요합니다")
    if not (0<=initial<=target<=100 and 0<eff<=1 and power>0 and capacity>0):
        raise ValueError("배터리 및 목표 SOC 값을 확인하세요")
    energy=capacity*(target-initial)/100/eff
    if energy>24*power+1e-9:raise ValueError("24시간 내 목표 SOC 충전이 불가능합니다")
    hours=np.zeros(24)
    for i in np.argsort(values,kind="stable"):
        hours[i]=min(power,energy)
        energy-=hours[i]
        if energy<=1e-9:break
    return hours,float(np.dot(values,hours))

def evaluate(actual,pred):
    a=np.asarray(actual,float);p=np.asarray(pred,float)
    if len(a)!=24 or len(p)!=24 or not np.isfinite(a).all():raise ValueError("실제 SMP 24시간 미확보")
    opt,oracle=g2v(a);plan,_=g2v(p)
    return dict(MAE=float(np.mean(abs(a-p))),
         RMSE=float(np.sqrt(np.mean((a-p)**2))),G2V_regret=float(np.dot(a,plan-opt)))
