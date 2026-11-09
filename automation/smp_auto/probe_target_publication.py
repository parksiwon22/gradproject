"""Read the official KPX public page before the 23:00 origin; no authentication."""
from __future__ import annotations
import json, os, sys, urllib.request
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(os.environ.get("SMP_SOURCE",ROOT.parent/"gradproject")).resolve()
sys.path.insert(0,str(SOURCE))
from v11.store import connect, ingest
from v11.public_prices import parse_public

KST="Asia/Seoul"

def probe(source=SOURCE,origin_date=None,now=None):
    source=Path(source)
    origin_date=origin_date or os.environ["SMP_ORIGIN_DATE"]
    origin=pd.Timestamp(origin_date,tz=KST)+pd.Timedelta(hours=23)
    checked=pd.Timestamp.now(tz=KST) if now is None else pd.Timestamp(now)
    if checked.tzinfo is None: checked=checked.tz_localize(KST)
    else: checked=checked.tz_convert(KST)
    results=[]
    db=connect(source/"artifacts"/"v11")
    try:
        for h in (1,2,3):
            day=(pd.Timestamp(origin_date,tz=KST)+pd.Timedelta(days=h)).strftime("%Y-%m-%d")
            url=f"https://www.kpx.or.kr/smpInland.es?mid=a10404080100&issue_date={day}"
            try:
                req=urllib.request.Request(url,headers={"User-Agent":"gradproject-research-automation/1.0"})
                with urllib.request.urlopen(req,timeout=40) as response: payload=response.read()
                frame=parse_public(payload.decode("utf-8"),day)
                status="official_available" if frame is not None else "not_observed_on_this_page"
                if frame is not None:
                    ingest(db,"land",frame,payload,url,retrieved=checked.tz_convert("UTC").isoformat())
            except Exception as exc:
                status="probe_failed"; error=type(exc).__name__
            item={"horizon_days":h,"target_date":day,"checked_at_kst":checked.isoformat(),
                  "checked_at_utc":checked.tz_convert("UTC").isoformat(),"status":status,
                  "before_origin":checked<=origin}
            if status=="probe_failed": item["error_type"]=error
            results.append(item)
    finally: db.close()
    dest=source/"artifacts"/"v11"/"public_availability_live.json"
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps({"origin_kst":origin.isoformat(),"results":results},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(results,ensure_ascii=False))
    return results

if __name__=="__main__": probe()
