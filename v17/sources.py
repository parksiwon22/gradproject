"""Two bounded demand checks; retain receipt timing and no false past availability."""
import json,hashlib
import pandas as pd
from v11.authenticated_data import key,call,KPX
from v11.store import now
from v17.common import HOME,ROOT,write
from v17.legacy import origin

def run(publish=True):
    clock=pd.Timestamp.now(tz='Asia/Seoul');bound=origin(clock);folder=HOME/'sources';folder.mkdir(parents=True,exist_ok=True);rows=[]
    for ahead in [2,3]:
        target=(bound.normalize()+pd.Timedelta(days=ahead)).strftime('%Y%m%d');dest=folder/(target+'_demand_check.json')
        if dest.exists():rows.append(json.loads(dest.read_text()));continue
        row=dict(source=KPX,target_date=target,checked_at=now(),reference_origin=bound.isoformat(),original_issue_time=None,used_by_model=False,D2_eligible=False)
        credential=key('KPX_SERVICE_KEY')
        if not credential:row['status']='dedicated_key_missing'
        else:
            try:
                data=call(KPX,dict(pageNo=1,numOfRows=100,dataType='json',date=target),credential);payload=json.dumps(data).encode();digest=hashlib.sha256(payload).hexdigest();(folder/(digest+'.raw')).write_bytes(payload)
                body=data.get('response',{});items=body.get('body',{}).get('items',{})
                if isinstance(items,dict):items=items.get('item',[])
                row.update(status='response_received',resultCode=str(body.get('header',{}).get('resultCode')),rows=len(items) if isinstance(items,list) else 0,first_seen=now(),sha256=digest,reason='No original publication vintage; current first receipt is after reference cutoff, no as-of substitution')
            except RuntimeError as exc:row.update(status='failed',reason=str(exc))
        write(dest,row);rows.append(row)
    old=json.loads((ROOT/'docs/V16_MAINTENANCE_SOURCE.json').read_text());rows.append(dict(source=old['source'],kind='maintenance_archive',documents=old['documents'],historical_original_vintage_verified=False,used_by_V17_model=False,reason='V16 maintenance study did not consistently improve peak and EV metrics; historical replacement history unresolved'))
    rows.append(dict(source='https://apihub.kma.go.kr/apiList.do?seqApi=10&seqApiSub=286',kind='weather_forecast',status='Prior403 unresolved; not retried by V17',used_by_model=False))
    write(folder/'availability.json',dict(at=now(),rows=rows,new_eligible_historical_input_groups=0,no_additional_peak_model_fit=True))
    if publish:write(ROOT/'docs/V17_SOURCE_AVAILABILITY.json',rows)
    print('Bounded official demand capture completed; no new historical eligibility claimed')

if __name__=='__main__':run()
