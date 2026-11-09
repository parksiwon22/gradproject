"""Keyless EPSIS collection plus isolated 23:00-origin forecast replay."""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(os.environ.get("SMP_SOURCE",ROOT.parent/"gradproject")).resolve()
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(SOURCE))
from v11.store import connect,latest
from automation.smp_auto.probe_target_publication import probe
from automation.smp_auto.publish_forecast import generate,seasonal_anchor

def run():
    source=SOURCE
    collected_at=pd.Timestamp.now(tz="Asia/Seoul")
    day=collected_at.strftime("%Y-%m-%d")
    origin=pd.Timestamp(day,tz="Asia/Seoul")+pd.Timedelta(hours=23)
    test_dir=Path(os.environ.get("SMP_TEST_OUTPUT",ROOT/"outputs"/"integration"))
    test_dir.mkdir(parents=True,exist_ok=True)
    collect_log=test_dir/"epsis_collection.log"
    env=os.environ.copy(); env["SMP_SOURCE"]=str(source)
    for key in list(env):
        if ("EPSIS" in key.upper() or "KPX" in key.upper()) and "KEY" in key.upper():
            env.pop(key,None)
    proc=subprocess.run([sys.executable,"-m","v11.collect","--market","land"],cwd=source,
        env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    collect_log.write_text(proc.stdout,encoding="utf-8")
    home=source/"artifacts"/"v11"
    coverage_path=home/"collection_status.json"
    try: collection_status=json.loads(coverage_path.read_text(encoding="utf-8"))
    except Exception: collection_status={"failures":[{"error":"collection_status_missing"}]}
    db=connect(home)
    try:
        snapshot=latest(db,"land",as_of=origin.tz_convert("UTC").isoformat())
    finally: db.close()
    indexed=snapshot.drop_duplicates("timestamp").set_index("timestamp").smp.sort_index()
    recent=pd.date_range(origin-pd.Timedelta(hours=167),origin,freq="h")
    recent_missing=int(len(recent.difference(indexed.index)))
    anchor_coverage={}
    anchor_errors={}
    for horizon in (2,3):
        targets=pd.date_range(origin.normalize()+pd.Timedelta(days=horizon),
            origin.normalize()+pd.Timedelta(days=horizon,hours=23),freq="h")
        anchor_coverage[str(horizon)]={}
        for lag in (7,14,21,28):
            expected=targets-pd.Timedelta(days=lag)
            anchor_coverage[str(horizon)][str(lag)]={"expected_hours":24,
                "present_hours":int(expected.isin(indexed.index).sum())}
        try: seasonal_anchor(snapshot,targets)
        except ValueError as exc: anchor_errors[str(horizon)]=str(exc)
    publication=probe(source=source,origin_date=day)
    # This timestamp is a test fixture only. Inputs and KPX checks retain their
    # actual receive times; generate() still filters every input at the origin.
    simulated_issue=origin+pd.Timedelta(minutes=1)
    generated=generate(source=source,model_dir=ROOT/"automation"/"models",
        out_path=test_dir/"generator_output.json",state_path=test_dir/"forecast_history.json",
        now=simulated_issue,origin_date=day)
    forecast=generated.get("last_successful_forecast") or {}
    vectors=forecast.get("horizons",[])
    for item in vectors:
        values=np.asarray(item.get("predictions",[]),dtype=float)
        item["point_count_check"]=(len(values)==24)
        item["finite_check"]=bool(len(values)==24 and np.isfinite(values).all())
    result={"test_type":"nonproduction_integration_replay_using_live_eps_is_collection",
      "live_forecast_issued":False,"api_key_required":False,
      "collection_command":"python -m v11.collect --market land",
      "collection_exit_code":proc.returncode,"collection_status":collection_status,
      "epsis_coverage":collection_status.get("coverage",[]),
      "collection_log":"epsis_collection.log","checked_at_kst":collected_at.isoformat(),
      "scheduled_origin_kst":origin.isoformat(),"simulated_issue_at_kst":simulated_issue.isoformat(),
      "origin_already_reached_at_collection":bool(collected_at>=origin),
      "asof_snapshot_hours":len(snapshot),"recent_168h_present":len(recent)-recent_missing,
      "recent_168h_missing":recent_missing,"seasonal_anchor_coverage":anchor_coverage,
      "seasonal_anchor_errors":anchor_errors,
      "public_page_probes":publication,
      "forecast_run":generated.get("run"),
      "horizons":[{"horizon_days":x.get("horizon_days"),"target_date":x.get("target_date"),
        "official_smp_at_issue":x.get("official_smp_at_issue"),
        "prospective_forecast_eligible":x.get("prospective_forecast_eligible"),
        "point_count":len(x.get("predictions",[])),"all_finite":x.get("finite_check"),
        "input_max_timestamp_kst":x.get("input_max_timestamp_kst"),
        "input_max_first_seen_utc":x.get("input_max_first_seen_utc"),
        "input_receipts_before_origin":bool(x.get("input_max_first_seen_utc") and
          pd.Timestamp(x["input_max_first_seen_utc"])<=origin.tz_convert("UTC")),
        "input_vintage_verified":x.get("input_vintage_verified"),
        "anchor_days_coverage":anchor_coverage.get(str(x.get("horizon_days"))) if int(x.get("horizon_days",1))>1 else None}
        for x in vectors],
      "notice":"재생 테스트이며 실제 23:00 운영 발급이 아닙니다. 백필 당시의 원본 first_seen 및 과거 가격 수정 이력은 증명하지 않습니다."}
    (test_dir/"integration_report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("test_type","live_forecast_issued","api_key_required","collection_exit_code","epsis_coverage","scheduled_origin_kst","simulated_issue_at_kst","recent_168h_present","recent_168h_missing","seasonal_anchor_coverage","horizons")},ensure_ascii=False,indent=2))
    if proc.returncode or collection_status.get("failures") or recent_missing or anchor_errors or any(
        v["present_hours"]!=24 for by_lag in anchor_coverage.values() for v in by_lag.values()):
        raise SystemExit(1)
    if generated.get("run",{}).get("status")!="issued_research_only" or len(vectors)!=3 or any(not x.get("finite_check") for x in vectors):
        raise SystemExit(1)
    return result

if __name__=="__main__": run()
