"""Normalize already captured official records without another authenticated call."""
import json
from pathlib import Path
import pandas as pd
from v11.store import now
from v16.common import HOME,ROOT,write,csv

def run():
    inventory=json.loads((HOME/'sources/inventory.json').read_text(encoding='utf-8'));rows=inventory['rows']
    kpx=next((r for r in rows if r['name']=='kpx_authenticated'),{})
    if kpx.get('resultCode')=='00':
        data=json.loads((HOME/kpx['raw']).read_bytes());items=data['response']['body']['items'].get('item',[]);f=pd.DataFrame(items)
        if not f.empty:
            hours=pd.to_numeric(f.hour,errors='raise')
            if not hours.between(1,24).all():raise ValueError('KPX endpoint hours must1..24')
            f['timestamp']=pd.to_datetime(f.date,format='%Y%m%d').dt.tz_localize('Asia/Seoul')+pd.to_timedelta(hours-1,unit='h');f['first_seen']=kpx['first_seen'];f['issued_at']=None
            f.to_csv(HOME/'sources/kpx_received_native_fields.csv',index=False)
            kpx.update(rows=len(f),target_dates=sorted(f.timestamp.dt.strftime('%Y-%m-%d').unique()),reason='Forecast values received but no per-record issue timestamp or historical vintage. Native numeric fields preserved; no demand semantic inference from garbled area label',D2_eligible=False)
    outage=HOME/'outages/inventory.json'
    if outage.exists():
        archive=json.loads(outage.read_text(encoding='utf-8'));checks=json.loads((HOME/'outages/extraction_checks.json').read_text()) if (HOME/'outages/extraction_checks.json').exists() else []
        counts={}
        for r in archive['records']:counts[r['posted_date'][:4]]=counts.get(r['posted_date'][:4],0)+1
        summary=dict(at=now(),documents=len(archive['records']),year_documents=counts,files=sum(len(r['files']) for r in archive['records']),structured_rows=sum(r['parsed_rows'] for r in checks),ambiguous_rows=sum(r['ambiguous_rows'] for r in checks),original_historical_vintage_verified=False,posted_timestamp_available=sum(bool(r['posted_at']) for r in archive['records']),first_receipt_is_current=True,source='https://www.kpx.or.kr/board.es?bid=0018&mid=a10109030500',raw_redistribution='excluded; current archive license/history not established',model_use='retrospective pilot only, issue-event filter, never live-certified')
        extracted=HOME/'outages/extracted_rows.json'
        if extracted.exists():
            f=pd.DataFrame(json.loads(extracted.read_text(encoding='utf-8')))
            invalid=pd.to_datetime(f.planned_end,format='mixed')<=pd.to_datetime(f.planned_start,format='mixed')
            summary.update(invalid_interval_rows=int(invalid.sum()),invalid_interval_documents=int(f.loc[invalid,'document_id'].nunique()),invalid_interval_action='Entire document excluded; raw value preserved, no arbitrary repair')
        write(ROOT/'docs/V16_MAINTENANCE_SOURCE.json',summary)
        rows=[r for r in rows if r['name']!='weekly_maintenance_archive']+[dict(name='weekly_maintenance_archive',source=summary['source'],kind='announced_forecast_document',rows=summary['structured_rows'],issued_at='per-document posted timestamp, exact original attachment version unknown',D2_eligible=False,reason='Retrospective-only archive; current files cannot certify historical first receipt',documents=summary['documents'])]
    inventory['rows']=rows;write(HOME/'sources/inventory.json',inventory)
    write(ROOT/'docs/V16_SOURCE_AVAILABILITY.json',[{k:r.get(k) for k in ['name','source','kind','http_status','status','reason','rows','documents','issued_at','first_seen','target_dates','D2_eligible','original_vintage_verified']} for r in rows])
    print('Source audit saved; API received rows',kpx.get('rows',0),flush=True)

if __name__=='__main__':run()
