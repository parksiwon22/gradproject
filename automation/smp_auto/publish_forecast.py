"""Conservative research-only daily runner; never calls or changes the V20 issue gate."""
from __future__ import annotations
import hashlib, json, os, sys, urllib.request
from pathlib import Path
import joblib
import numpy as np
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

def seasonal_anchor(prices, targets):
    series=prices.set_index("timestamp").smp.sort_index()
    if series.index.tz is None or series.index.duplicated().any():
        raise ValueError("seasonal anchor 입력의 시간대 또는 시간 중복이 유효하지 않습니다.")
    weeks=np.vstack([series.reindex(targets-pd.Timedelta(days=7*k)).to_numpy(float) for k in (1,2,3,4)])
    if weeks.shape!=(4,len(targets)) or not np.isfinite(weeks).all():
        raise ValueError("7·14·21·28일 seasonal anchor 입력 중 결측 또는 비정상 값이 있습니다.")
    return np.median(weeks,axis=0)

def load_history(state_path):
    try:
        history=json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else []
        if isinstance(history,list) and history: return history,"local_state",None
    except Exception as exc:
        history_error=type(exc).__name__
    else:
        history_error=None
    url=os.environ.get("SMP_PUBLISHED_HISTORY_URL")
    if url:
        try:
            request=urllib.request.Request(url,headers={"Cache-Control":"no-cache","User-Agent":"gradproject-research-automation/1.0"})
            with urllib.request.urlopen(request,timeout=15) as response: payload=json.loads(response.read().decode("utf-8"))
            history=payload.get("forecast_history",[])
            if not history and payload.get("last_successful_forecast"):
                history=[payload["last_successful_forecast"]]
            if isinstance(history,list): return history,"published_pages",None
            history_error="invalid_history_schema"
        except Exception as exc:
            history_error=type(exc).__name__
    return [],"empty_after_recovery_attempt",history_error

def _hourly_vector(frame, day):
    rows=frame[frame.timestamp.dt.strftime("%Y-%m-%d")==str(day)].sort_values("timestamp")
    if len(rows)!=24 or rows.timestamp.dt.hour.tolist()!=list(range(24)):
        return None
    values=rows.smp.to_numpy(float)
    return values if np.isfinite(values).all() else None

def _g2v_regret(prediction, reference, capacity_kwh=60., soc_start=30., soc_goal=80.,
                charge_kw=7., efficiency=.9, start_hour=0, end_hour=24):
    """Fixed comparable 24-hour G2V regret; values are wholesale-SMP proxies."""
    need=max(0.,(soc_goal-soc_start)*capacity_kwh/100.)
    slots=list(range(int(start_hour),int(end_hour)))
    def allocate(prices):
        left=need; total=0.
        for hour in sorted(slots,key=lambda h:(float(prices[h]),h)):
            if left<=1e-9: break
            grid_kwh=min(charge_kw,left/efficiency)
            total+=grid_kwh*float(prices[hour])
            left-=grid_kwh*efficiency
        return total if left<=1e-6 else None
    selected=allocate(prediction); oracle=allocate(reference)
    if selected is None or oracle is None: return None
    return {"regret_won":float(max(0.,selected-oracle)),"selected_plan_cost_won":float(selected),
            "oracle_cost_won":float(oracle),"conditions":{"battery_kwh":capacity_kwh,
            "soc_start_pct":soc_start,"soc_goal_pct":soc_goal,"charge_kw":charge_kw,
            "efficiency":efficiency,"start_hour":start_hour,"end_hour":end_hour}}

def _reference_metrics(prediction, reference, eligible, exclusion_reason=None):
    pred=np.asarray(prediction,dtype=float); actual=np.asarray(reference,dtype=float)
    if pred.shape!=(24,) or actual.shape!=(24,) or not np.isfinite(pred).all() or not np.isfinite(actual).all():
        return None
    error=actual-pred
    return {"n_hours":24,"mae":float(np.abs(error).mean()),"rmse":float(np.sqrt(np.mean(error**2))),
        "hourly_signed_error":[round(float(x),6) for x in error],
        "g2v_regret":_g2v_regret(pred,actual),"prospective_evaluation_eligible":bool(eligible),
        "exclusion_reason":None if eligible else exclusion_reason}

def _enrich_history(history, observed, asof_prices, publication_by_day, now):
    """Attach later price observations without altering any stored forecast vector."""
    for forecast in history:
        origin_text=forecast.get("origin_kst")
        try: origin=pd.Timestamp(origin_text)
        except Exception: continue
        for item in forecast.get("horizons",[]):
            day=item.get("target_date"); predictions=item.get("predictions")
            if not day or not isinstance(predictions,list) or len(predictions)!=24: continue
            h=int(item.get("horizon_days",0) or 0)
            item.setdefault("forecast_id",f"{origin.isoformat()}|{day}|D+{h}|{item.get('model_sha256','unknown')}")
            item.setdefault("forecast_origin_kst",origin.isoformat())
            item.setdefault("forecast_origin_date",origin.strftime("%Y-%m-%d"))
            item.setdefault("original_horizon_days",h)
            item.setdefault("generated_at_kst",forecast.get("issued_at_kst"))
            item.setdefault("generated_at_utc",forecast.get("issued_at_utc"))
            item.setdefault("model_sha256",item.get("model_sha256"))
            probe=publication_by_day.get(day)
            if probe and probe.get("status")=="official_available":
                values=_hourly_vector(asof_prices,day)
                if values is not None and pd.Timestamp(probe.get("checked_at_kst"))>origin:
                    item.setdefault("official_publication",{
                        "source":"KPX inland SMP public page","status":"published_after_forecast_origin",
                        "checked_at_kst":probe.get("checked_at_kst"),"values":[round(float(x),6) for x in values],
                        "price_nature":"published official SMP; final-settlement status is not asserted"})
            eligible=bool(item.get("prospective_forecast_eligible"))
            exclusion=(None if eligible else "publication was known or not verified at this forecast origin")
            official=item.get("official_publication")
            if official:
                item.setdefault("official_metrics",_reference_metrics(predictions,official["values"],eligible,exclusion))
            target_end=pd.Timestamp(day,tz=KST)+pd.Timedelta(hours=23)
            if target_end>=now: continue
            values=_hourly_vector(observed,day)
            if values is None: continue
            rows=observed[observed.timestamp.dt.strftime("%Y-%m-%d")==str(day)]
            first_seen=pd.to_datetime(rows.first_seen,utc=True)
            last_seen=pd.to_datetime(rows.last_seen,utc=True)
            if last_seen.isna().any() or not (last_seen>target_end.tz_convert("UTC")).all(): continue
            # Keep the stored prediction immutable and attach the post-target reference separately.
            item.setdefault("post_target_observation",{
                "source":"EPSIS land SMP collection","status":"observed_after_target_end",
                "first_seen_min_utc":str(first_seen.min()),"first_seen_max_utc":str(first_seen.max()),
                "last_seen_min_utc":str(last_seen.min()),"last_seen_max_utc":str(last_seen.max()),
                "values":[round(float(x),6) for x in values],
                "price_nature":"EPSIS SMP observed again after target end; final-settlement status is not independently verified"})
            item.setdefault("actual_status","post_target_eps_is_observed_finality_unverified")
            post=item["post_target_observation"]
            item.setdefault("post_target_metrics",_reference_metrics(predictions,post["values"],eligible,exclusion))
            # Backward-compatible metric field is explicitly typed as an EPSIS post-target comparison.
            if item.get("post_target_metrics"):
                item.setdefault("metrics",item["post_target_metrics"])
                item.setdefault("evaluation_type","post_target_eps_is_observation_finality_unverified")

def generate(source=SOURCE, model_dir=MODEL_DIR, out_path=OUT, state_path=STATE,
             now=None, origin_date=None):
    source=Path(source); model_dir=Path(model_dir); out_path=Path(out_path); state_path=Path(state_path)
    now=pd.Timestamp.now(tz=KST) if now is None else pd.Timestamp(now)
    if now.tzinfo is None: now=now.tz_localize(KST)
    else: now=now.tz_convert(KST)
    origin_date=origin_date or os.environ.get("SMP_ORIGIN_DATE",now.strftime("%Y-%m-%d"))
    origin=pd.Timestamp(origin_date,tz=KST)+pd.Timedelta(hours=23)
    issued=now.tz_convert("UTC").isoformat()
    result={"schema_version":2,"market":"land","run":{"status":"withheld","reason":None,
             "scheduled_origin_kst":origin.isoformat(),"issued_at_utc":issued,"issued_at_kst":now.isoformat(),
             "schedule_delay_seconds":float((now-origin).total_seconds()) if now>=origin else None,
             "collection_completed_at_utc":None,
             "collection_status":"status file unavailable","official_target_status":"unknown"},
             "last_successful_forecast":None}
    state_path.parent.mkdir(parents=True,exist_ok=True)
    history,history_source,history_recovery_error=load_history(state_path)
    result["run"]["history_source"]=history_source
    if history_recovery_error: result["run"]["history_recovery_status"]=history_recovery_error
    status_path=source/"artifacts"/"v11"/"collection_status.json"
    if status_path.exists():
        try:
            collected=json.loads(status_path.read_text(encoding="utf-8"))
            result["run"]["collection_completed_at_utc"]=collected.get("at")
            result["run"]["collection_status"]=("collection completed" if not collected.get("failures") else
                "collection failed: "+"; ".join(x.get("market","")+" "+x.get("error","") for x in collected["failures"]))
        except Exception: result["run"]["collection_status"]="collection status parse failed"
    publication_path=source/"artifacts"/"v11"/"public_availability_live.json"
    try:
        publication=json.loads(publication_path.read_text(encoding="utf-8")) if publication_path.exists() else {}
        publication_by_day={x["target_date"]:x for x in publication.get("results",[])}
    except Exception: publication_by_day={}
    try:
        if origin>now: raise RuntimeError("지정된 23:00 기준시각이 아직 도래하지 않아 예측을 보류했습니다.")
        db=connect(source/"artifacts"/"v11")
        try:
            asof_prices=latest(db,"land",as_of=origin.tz_convert("UTC").isoformat())
            observed=latest(db,"land")
        finally: db.close()
        # Enrichment runs after the issue-time snapshot is fixed. Historical price values
        # are attached as separate references; prediction vectors are never overwritten.
        _enrich_history(history,observed,asof_prices,publication_by_day,now)
        existing=next((x for x in history if x.get("origin_kst")==origin.isoformat()
                       and x.get("status")=="issued_research_only"),None)
        if existing is not None:
            result["run"].update(status="already_issued_for_origin",reason=None,
              official_target_status="retained_original_forecast_no_duplicate")
            raise RuntimeError("ALREADY_ISSUED_FOR_ORIGIN")
        prices=asof_prices[asof_prices.timestamp<=origin].drop_duplicates("timestamp").sort_values("timestamp")
        if prices.empty or prices.timestamp.max()<origin:
            raise RuntimeError("23:00 기준 입력 SMP 미확보: 해당 시각 이후 수신된 SMP는 제외했으며 예측 발급을 보류했습니다.")
        recent=pd.date_range(origin-pd.Timedelta(hours=167),origin,freq="h")
        indexed=prices.set_index("timestamp").smp
        if not recent.isin(indexed.index).all(): raise RuntimeError("23:00 기준 최근 168시간 입력에 결측이 있어 예측을 보류했습니다.")
        if not np.isfinite(indexed.reindex(recent).to_numpy(float)).all(): raise RuntimeError("최근 168시간 입력에 NaN 또는 무한대가 있어 예측을 보류했습니다.")
        model_meta=json.loads((model_dir/"manifest.json").read_text(encoding="utf-8"))["models"]
        horizons=[]
        for h in (1,2,3):
            meta=model_meta[str(h)]; path=model_dir/meta["file"]
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
            probe=publication_by_day.get(target_date.strftime("%Y-%m-%d"),{})
            probe_time=pd.Timestamp(probe["checked_at_kst"]) if probe.get("checked_at_kst") else None
            probe_before=bool(probe_time is not None and probe_time<=origin)
            if official_status=="official_smp_observed_before_origin":
                eligibility=False
            elif official_status=="partial_smp_observed_before_origin":
                eligibility=False
            else:
                eligibility=bool(probe_before and probe.get("status")=="not_observed_on_this_page")
                if not eligibility: official_status="publication_status_unverified_at_origin"
            if h>1 and meta.get("anchor_method")!="median_same_hour_days_7_14_21_28":
                raise RuntimeError(f"D+{h} 체크포인트 anchor 방식이 최종 연구 사양과 다릅니다.")
            frame=(build_d1(prices,targets=targets,days_ahead=1,include_labels=False) if h==1 else
                   build_dx(prices,targets=targets,days_ahead=h,include_labels=False))
            if len(frame)!=24: raise RuntimeError(f"D+{h} 입력 피처가 24시간 완전하지 않습니다.")
            base=frame.daily.to_numpy(float)+bundle["model"].predict(frame[bundle["features"]].to_numpy(float))
            anchor=seasonal_anchor(prices,targets)
            pred=base*(1-float(bundle["anchor_weight"]))+anchor*float(bundle["anchor_weight"])
            if not np.isfinite(pred).all(): raise RuntimeError(f"D+{h} 예측에 NaN 또는 무한대가 있어 발급을 보류했습니다.")
            horizons.append({"forecast_id":f"{origin.isoformat()}|{target_date.strftime('%Y-%m-%d')}|D+{h}|{sha256(path)}",
              "forecast_origin_kst":origin.isoformat(),"forecast_origin_date":origin.strftime("%Y-%m-%d"),
              "horizon_days":h,"original_horizon_days":h,"target_date":target_date.strftime("%Y-%m-%d"),
              "generated_at_utc":issued,"generated_at_kst":now.isoformat(),
              "model_family":bundle["family"],"model_file":path.name,"model_sha256":sha256(path),
              "trained_through":bundle["trained_through"],"predictions":[round(float(x),6) for x in pred],
              "official_smp_at_issue":official_status,"official_smp_at_issue_values":known_smp,
              "prospective_forecast_eligible":eligibility,
              "publication_probe":probe or None,
              "publication_check":"KPX public page checked before origin plus EPSIS receipt-time snapshot; historical first-publication time is not inferred",
              "actual_smp":None,
              "input_max_timestamp_kst":prices.timestamp.max().isoformat(),
              "input_max_first_seen_utc":str(prices.first_seen.max()),
              "input_vintage_verified":False,"research_metrics_applied":False})
        new_forecast={"origin_kst":origin.isoformat(),"issued_at_utc":issued,
          "issued_at_kst":now.isoformat(),"status":"issued_research_only","horizons":horizons,
          "notice":"별도 재학습 체크포인트의 연구용 예측입니다. 275일 CV 점수는 이 가중치의 운영 성능이 아니며, 과거 first_seen은 검증되지 않았습니다."}
        history.append(new_forecast)
        any_known=any(x["official_smp_at_issue"] in ("official_smp_observed_before_origin","partial_smp_observed_before_origin") for x in horizons)
        result["run"].update(status="issued_research_only",reason=None,
          official_target_status="target_smp_observed_for_one_or_more_horizons" if any_known else "not_observed_in_eps_is_snapshot")
    except Exception as exc:
        if str(exc)=="ALREADY_ISSUED_FOR_ORIGIN":
            pass
        else:
            result["run"].update(status="withheld",reason=f"{type(exc).__name__}: {exc}")
    state_path.write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding="utf-8")
    successes=[x for x in history if x.get("status")=="issued_research_only"]
    result["last_successful_forecast"]=max(successes,key=lambda x:x.get("issued_at_utc", "")) if successes else None
    result["forecast_history"]=history
    out_path.parent.mkdir(parents=True,exist_ok=True)
    out_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result["run"],ensure_ascii=False))
    return result

def main(): generate()

if __name__=="__main__": main()
