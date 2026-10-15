"""Official archived day-ahead demand values; original publication vintages unknown."""
import argparse
import html
import json
import re
import time
import pandas as pd
from v11.exogenous_collect import EXO,fetch,save
from v11.store import now

def parse_week(text,end):
    if f'value="{end}"' not in text: raise ValueError('demand forecast date mismatch')
    rows=[]
    for block in re.findall(r'<tr[^>]*>(.*?)</tr>',text,re.S):
        cells=[html.unescape(re.sub(r'<[^>]*>','',v)).strip().replace(',','') for v in re.findall(r'<td[^>]*>(.*?)</td>',block,re.S)]
        if len(cells)!=8 or not re.fullmatch(r'\d+h',cells[0]): continue
        hour=int(cells[0][:-1])-1
        for i,value in enumerate(cells[1:]):
            if re.fullmatch(r'\d+(?:\.\d+)?',value) and float(value)>0:
                timestamp=pd.Timestamp(end,tz='Asia/Seoul')-pd.Timedelta(days=6-i)+pd.Timedelta(hours=hour)
                rows.append(dict(timestamp=timestamp,day_ahead_demand_mw=float(value)))
    if not rows: return pd.DataFrame(columns=['timestamp','day_ahead_demand_mw'])
    result=pd.DataFrame(rows); counts=result.groupby(result.timestamp.dt.date).size()
    return result[result.timestamp.dt.date.isin(counts[counts==24].index)].sort_values('timestamp')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--latest',action='store_true'); args=p.parse_args()
    folder=EXO/'demand_forecast_snapshots'; folder.mkdir(exist_ok=True)
    end=pd.Timestamp.now(tz='Asia/Seoul').normalize()+pd.Timedelta(days=1)
    start=end-pd.Timedelta(days=6) if args.latest else pd.Timestamp('2023-01-01',tz='Asia/Seoul')
    ends=list(pd.date_range(start+pd.Timedelta(days=6),end,freq='7D'))
    if not ends or ends[-1]<end: ends.append(end)
    frames=[]; failed=[]
    for i,date in enumerate(ends):
        day=date.strftime('%Y-%m-%d'); path=folder/(day+'.csv')
        if path.exists():
            f=pd.read_csv(path); f.timestamp=pd.to_datetime(f.timestamp,utc=True).dt.tz_convert('Asia/Seoul'); frames.append(f); continue
        url='https://www.kpx.or.kr/pssInland.es?mid=a10606090100&issue_date='+day
        try:
            payload=fetch(url); stamp=now(); save('demand_forecast_week_'+day+'.html',payload,url,dict(resolution='hourly',unit='MW',market='land',historical_issued_at='unknown; archive values are not original vintages'))
            f=parse_week(payload.decode('utf-8'),day); f['first_seen']=stamp; f['historical_issued_at']=None
            f.to_csv(path,index=False); frames.append(f)
            print('archived demand forecast',i+1,'/',len(ends),day,len(f),flush=True)
        except Exception as exc: failed.append(dict(day=day,error=type(exc).__name__))
        time.sleep(.5)
        if frames:
            pd.concat(frames).sort_values('first_seen').drop_duplicates('timestamp',keep='last').sort_values('timestamp').to_csv(EXO/'demand_forecast_hourly.csv',index=False)
    if args.latest and (EXO/'demand_forecast_archive.csv').exists():
        old=pd.read_csv(EXO/'demand_forecast_archive.csv'); old.timestamp=pd.to_datetime(old.timestamp,utc=True).dt.tz_convert('Asia/Seoul'); frames.append(old)
    all_data=pd.concat(frames).sort_values('first_seen').drop_duplicates('timestamp',keep='last').sort_values('timestamp')
    all_data=all_data[all_data.timestamp>=pd.Timestamp('2023-01-01',tz='Asia/Seoul')]
    all_data.to_csv(EXO/'demand_forecast_hourly.csv',index=False)
    if not args.latest: all_data.to_csv(EXO/'demand_forecast_archive.csv',index=False)
    (EXO/'demand_forecast_status.json').write_text(json.dumps(dict(at=now(),failed=failed,counts=all_data.groupby(all_data.timestamp.dt.year).size().to_dict(),warning='Original historical issue/publication time unknown; use separate retrospective forecast-value research, not certified as-of replay'),indent=2),encoding='utf-8')
    print('Demand forecast collection complete',len(all_data),flush=True)

if __name__=='__main__': main()
