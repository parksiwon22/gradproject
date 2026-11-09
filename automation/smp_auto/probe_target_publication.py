"""Read the official KPX public page before the 23:00 origin; no authentication."""
from __future__ import annotations
import json, os, re, sys, time, urllib.request
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(os.environ.get("SMP_SOURCE",ROOT.parent/"gradproject")).resolve()
sys.path.insert(0,str(SOURCE))
from v11.store import connect, ingest, validate
from v11.public_prices import parse_public

KST="Asia/Seoul"

def parse_public_status(text,day):
    frame=parse_public(text,day)
    if frame is not None:
        validate(frame)
        return frame,"official_available"
    # The public parser returns None for an all-zero placeholder, but also for
    # incomplete or malformed tables. Only the explicit 24-zero placeholder
    # proves that this page has not published nonzero prices.
    rows=[]
    for block in re.findall(r'<tr[^>]*>(.*?)</tr>',text,re.S):
        cells=re.findall(r'<td[^>]*>(.*?)</td>',block,re.S)
        if cells and re.fullmatch(r'\s*(\d{1,2})h\s*',cells[0]):
            value=re.sub(r'<[^>]*>','',cells[-1]).strip().replace(',','')
            rows.append(value)
    if f'value="{day}"' in text and len(rows)==24 and all(re.fullmatch(r'0+(?:\.0+)?',v) for v in rows):
        return None,"not_observed_on_this_page"
    raise ValueError("public page was incomplete or could not be parsed; availability is unverified")

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
            attempts=0; error=None; frame=None; received=checked
            for attempt in range(1,4):
                attempts=attempt
                try:
                    req=urllib.request.Request(url,headers={"User-Agent":"gradproject-research-automation/1.0"})
                    with urllib.request.urlopen(req,timeout=20) as response: payload=response.read()
                    received=pd.Timestamp.now(tz=KST)
                    frame,status=parse_public_status(payload.decode("utf-8"),day)
                    if frame is not None:
                        ingest(db,"land",frame,payload,url,retrieved=received.tz_convert("UTC").isoformat())
                    error=None
                    break
                except Exception as exc:
                    received=pd.Timestamp.now(tz=KST)
                    error=type(exc).__name__
                    status="probe_failed"
                    if attempt<3: time.sleep(2**(attempt-1))
            item={"horizon_days":h,"target_date":day,"checked_at_kst":received.isoformat(),
                  "checked_at_utc":received.tz_convert("UTC").isoformat(),"status":status,
                  "attempts":attempts,"before_origin":received<=origin}
            if status=="probe_failed": item["error_type"]=error
            results.append(item)
    finally: db.close()
    dest=source/"artifacts"/"v11"/"public_availability_live.json"
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps({"origin_kst":origin.isoformat(),"results":results},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(results,ensure_ascii=False))
    return results

if __name__=="__main__": probe()
