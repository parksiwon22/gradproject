"""Conservative research-only daily runner; never calls or changes the V20 issue gate."""
from __future__ import annotations
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
import joblib
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(os.environ.get("SMP_SOURCE", ROOT.parent/"gradproject")).resolve()
sys.path.insert(0,str(SOURCE))
from v11.store import connect, latest
from v11.features import build as build_d1
from v17.features import build as build_dx

KST="Asia/Seoul"; MODEL_DIR=ROOT/"automation"/"models"; OUT=ROOT/"docs"/"live"/"latest.json"
STATE=ROOT/"automation"/"state"/"forecast_history.json"

def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    now=pd.Timestamp.now(tz=KST)
    origin_date=os.environ.get("SMP_ORIGIN_DATE",now.strftime("%Y-%m-%d"))
    origin=pd.Timestamp(origin_date,tz=KST)+pd.Timedelta(hours=23)
    issued=datetime.now(timezone.utc).isoformat()
    result={"schema_version":1,"market":"land","run":{"status":"withheld","reason":None,
             "scheduled_origin_kst":origin.isoformat(),"issued_at_utc":issued,"issued_at_kst":now.isoformat(),
             "collection_status":"status file unavailable","official_target_status":"unknown"},
             "last_successful_forecast":None}
    STATE.parent.mkdir(parents=True,exist_ok=True)
    try: history=json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else []
    except Exception: history=[]
    status_path=SOURCE/"artifacts"/"v11"/"collection_status.json"
    if status_path.exists():
        try:
            collected=json.loads(status_path.read_text(encoding="utf-8"))
            result["run"]["collection_status"]=("collection completed" if not collected.get("failures") else
                "collection failed: "+"; ".join(x.get("market","")+" "+x.get("error","") for x in collected["failures"]))
        except Exception: result["run"]["collection_status"]="collection status parse failed"
    try:
        if origin>now: raise RuntimeError("지정된 23:00 기준시각이 아직 도래하지 않아 예측을 보류했습니다.")
        db=connect(SOURCE/"artifacts"/"v11")
        try:
            asof_prices=latest(db,"land",as_of=origin.tz_convert("UTC").isoformat())
            observed=latest(db,"land")
        finally: db.close()
        # Attach later official observations only after the prediction object is fixed.
        # These values never enter the feature frame or predictor above.
        for old in history:
            for item in old.get("horizons",[]):
                if item.get("actual_smp") is not None: continue
                day=item.get("target_date")
                actual=observed[observed.timestamp.dt.strftime("%Y-%m-%d")==day].sort_values("timestamp")
                if len(actual)==24 and actual.timestamp.dt.hour.tolist()==list(range(24)):
                    item["actual_smp"]=[round(float(x),6) for x in actual.smp]
                    item["actual_status"]="observed_after_issue"
                    item["actual_first_seen_min_utc"]=str(actual.first_seen.min())
                    item["actual_first_seen_max_utc"]=str(actual.first_seen.max())
                    item["evaluation_type"]="post_issue_official_price_comparison"
        prices=asof_prices[asof_prices.timestamp<=origin].drop_duplicates("timestamp").sort_values("timestamp")
        if prices.empty or prices.timestamp.max()<origin:
            raise RuntimeError("23:00 기준 입력 SMP 미확보: 해당 시각 이후 수신된 SMP는 제외했으며 예측 발급을 보류했습니다.")
        recent=pd.date_range(origin-pd.Timedelta(hours=167),origin,freq="h")
        indexed=prices.set_index("timestamp").smp
        if not recent.isin(indexed.index).all(): raise RuntimeError("23:00 기준 최근 168시간 입력에 결측이 있어 예측을 보류했습니다.")
        model_meta=json.loads((MODEL_DIR/"manifest.json").read_text(encoding="utf-8"))["models"]
        horizons=[]
        for h in (1,2,3):
            meta=model_meta[str(h)]; path=MODEL_DIR/meta["file"]
            if sha256(path)!=meta["sha256"]: raise RuntimeError(f"D+{h} 체크포인트 해시 불일치")
            bundle=joblib.load(path)
            trained=pd.Timestamp(bundle["trained_through"],tz=KST)+pd.Timedelta(hours=23)
            if trained>=origin: raise RuntimeError(f"D+{h} 모델 학습 cutoff가 예측 기준시각 이후입니다.")
            target_date=origin.normalize()+pd.Timedelta(days=h)
            targets=pd.date_range(target_date,target_date+pd.Timedelta(hours=23),freq="h")
            known=asof_prices[asof_prices.timestamp.dt.strftime("%Y-%m-%d")==target_date.strftime("%Y-%m-%d")].sort_values("timestamp")
            if len(known)==24 and known.timestamp.dt.hour.tolist()==list(range(24)):
                official_status="official_smp_observed_before_origin"
                known_smp=[round(float(x),6) for x in known.smp]
            elif len(known):
                official_status="partial_smp_observed_before_origin"
                known_smp=None
            else:
                official_status="not_observed_in_eps_is_snapshot_at_origin"
                known_smp=None
            frame=(build_d1(prices,targets=targets,days_ahead=1,include_labels=False) if h==1 else
                   build_dx(prices,targets=targets,days_ahead=h,include_labels=False))
            if len(frame)!=24: raise RuntimeError(f"D+{h} 입력 피처가 24시간 완전하지 않습니다.")
            base=frame.daily.to_numpy(float)+bundle["model"].predict(frame[bundle["features"]].to_numpy(float))
            anchor=frame.weekly.to_numpy(float)
            pred=base*(1-float(bundle["anchor_weight"]))+anchor*float(bundle["anchor_weight"])
            horizons.append({"horizon_days":h,"target_date":target_date.strftime("%Y-%m-%d"),
              "model_family":bundle["family"],"model_file":path.name,"model_sha256":sha256(path),
              "trained_through":bundle["trained_through"],"predictions":[round(float(x),6) for x in pred],
              "official_smp_at_issue":official_status,"official_smp_at_issue_values":known_smp,
              "prospective_forecast_eligible":official_status=="not_observed_in_eps_is_snapshot_at_origin",
              "publication_check":"EPSIS first_seen <= scheduled origin; does not prove first-publication time",
              "actual_smp":None,
              "input_max_timestamp_kst":prices.timestamp.max().isoformat(),
              "input_max_first_seen_utc":str(prices.first_seen.max()),
              "input_vintage_verified":False,"research_metrics_applied":False})
        new_forecast={"origin_kst":origin.isoformat(),"issued_at_utc":issued,
          "issued_at_kst":now.isoformat(),"status":"issued_research_only","horizons":horizons,
          "notice":"별도 재학습 체크포인트의 연구용 예측입니다. 275일 CV 점수는 이 가중치의 운영 성능이 아니며, 과거 first_seen은 검증되지 않았습니다."}
        history.append(new_forecast)
        any_known=any(x["official_smp_at_issue"]!="not_observed_in_eps_is_snapshot_at_origin" for x in horizons)
        result["run"].update(status="issued_research_only",reason=None,
          official_target_status="target_smp_observed_for_one_or_more_horizons" if any_known else "not_observed_in_eps_is_snapshot")
    except Exception as exc:
        result["run"].update(status="withheld",reason=f"{type(exc).__name__}: {exc}")
    history=history[-90:]
    STATE.write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding="utf-8")
    result["last_successful_forecast"]=history[-1] if history else None
    result["forecast_history"]=history
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result["run"],ensure_ascii=False))

if __name__=="__main__": main()
