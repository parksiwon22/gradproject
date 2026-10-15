"""Read user-registered API keys without ever logging credentials or request URLs."""
import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
import pandas as pd
from v11.exogenous_collect import EXO,save
from v11.store import now

ASOS='https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'
KPX='https://apis.data.go.kr/B552115/SmpWithForecastDemand/getSmpWithForecastDemand'

def key(name):
    result=os.environ.get(name)
    if not result and os.name=='nt':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,'Environment') as registry: result=winreg.QueryValueEx(registry,name)[0]
        except OSError: return None
    return result.strip() if result else None

def call(endpoint,params,credential):
    # A data.go.kr account key may be shared by services, but each service must
    # still authorize it. Never emit str(HTTPError), which can contain the key.
    query=urllib.parse.urlencode({**params,'serviceKey':urllib.parse.unquote(credential)})
    try:
        with urllib.request.urlopen(endpoint+'?'+query,timeout=45) as response: payload=response.read()
    except urllib.error.HTTPError as exc: raise RuntimeError('API HTTP '+str(exc.code)) from None
    except Exception as exc: raise RuntimeError('API transport failure: '+type(exc).__name__) from None
    # Defensive redaction in case an error response echoes authentication input.
    payload=payload.replace(credential.encode(),b'[REDACTED]').replace(urllib.parse.unquote(credential).encode(),b'[REDACTED]')
    try: result=json.loads(payload)
    except (ValueError,UnicodeDecodeError): raise RuntimeError('API returned non-JSON response; content suppressed') from None
    return result

def asos(backfill=False):
    credential=key('KMA_SERVICE_KEY') or key('KPX_SERVICE_KEY')
    if not credential: return dict(connected=False,reason='no data.go.kr account key configured')
    today=pd.Timestamp.now(tz='Asia/Seoul').normalize(); date=today-pd.Timedelta(days=2)
    parameters=dict(dataType='JSON',dataCd='ASOS',dateCd='HR',startDt=date.strftime('%Y%m%d'),startHh='00',endDt=date.strftime('%Y%m%d'),endHh='23',stnIds='108',numOfRows=999,pageNo=1)
    try: result=call(ASOS,parameters,credential)
    except RuntimeError as exc: return dict(connected=False,reason=str(exc))
    header=result.get('response',{}).get('header',{})
    code=str(header.get('resultCode','missing'))
    if code!='00': return dict(connected=False,resultCode=code,reason='service authorization or API response error; credential omitted')
    items=result['response']['body'].get('items',{}).get('item',[])
    if not items: return dict(connected=False,reason='authorized response had no observation items')
    safe=json.dumps(result).encode(); save('asos_connection_response.json',safe,ASOS,dict(params=parameters,credential_logged=False))
    status=dict(connected=True,resultCode=code,example_rows=len(items),key_source='KMA_SERVICE_KEY' if key('KMA_SERVICE_KEY') else 'KPX_SERVICE_KEY shared data.go.kr account key; service accepted')
    print('ASOS API connected:',True,'sample rows:',len(items),flush=True)
    if not backfill: return status
    folder=EXO/'asos_hourly_pages'; folder.mkdir(exist_ok=True)
    frames=[]
    for year in range(2023,today.year+1):
        end=min(pd.Timestamp(f'{year}-12-31',tz='Asia/Seoul'),today-pd.Timedelta(days=1)); page=1; count=0
        while True:
            path=folder/f'{year}_{page}.json'
            params={**parameters,'startDt':f'{year}0101','endDt':end.strftime('%Y%m%d'),'pageNo':page}
            if path.exists(): data=json.loads(path.read_text(encoding='utf-8'))
            else:
                data=call(ASOS,params,credential)
                if str(data.get('response',{}).get('header',{}).get('resultCode'))!='00': raise RuntimeError('ASOS backfill response error; details suppressed')
                path.write_text(json.dumps(data),encoding='utf-8')
                save(f'asos_{year}_{page}_response.json',path.read_bytes(),ASOS,dict(params=params,credential_logged=False))
            body=data['response']['body']; rows=body.get('items',{}).get('item',[])
            frames.append(pd.DataFrame(rows)); count+=len(rows)
            print('ASOS',year,'page',page,'rows',len(rows),flush=True)
            if count>=int(body['totalCount']): break
            if not rows: raise RuntimeError('ASOS pagination stopped early')
            page+=1
    f=pd.concat(frames,ignore_index=True); f['timestamp']=pd.to_datetime(f.tm).dt.tz_localize('Asia/Seoul')
    columns={'ta':'temperature_c','hm':'humidity_percent','ws':'wind_ms','icsr':'solar_radiation_mj_m2','ss':'sunshine_hours','rn':'rainfall_mm'}
    for source,target in columns.items(): f[target]=pd.to_numeric(f[source],errors='coerce')
    f=f[['timestamp']+list(columns.values())].sort_values('timestamp')
    if f.timestamp.duplicated().any(): raise ValueError('duplicate hourly ASOS timestamps')
    f.to_csv(EXO/'weather_hourly_all.csv',index=False)
    for year,g in f.groupby(f.timestamp.dt.year): g.to_csv(EXO/f'weather_hourly_{year}.csv',index=False)
    status['year_counts']=f.groupby(f.timestamp.dt.year).size().to_dict()
    return status

def kpx():
    credential=key('KPX_SERVICE_KEY')
    if not credential: return dict(connected=False,reason='KPX_SERVICE_KEY absent')
    date=pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y%m%d')
    parameters=dict(pageNo=1,numOfRows=100,dataType='json',date=date)
    try: result=call(KPX,parameters,credential)
    except RuntimeError as exc: return dict(connected=False,reason=str(exc))
    # Persist only a key-redacted response and a credential-free request manifest.
    save('kpx_authenticated_response.json',json.dumps(result).encode(),KPX,dict(params=parameters,credential_logged=False))
    header=result.get('response',{}).get('header',{})
    code=str(header.get('resultCode','missing'))
    status=dict(connected=code in ('00','0'),resultCode=code,checked_at=now())
    items=result.get('response',{}).get('body',{}).get('items',{})
    if isinstance(items,dict): items=items.get('item',[])
    if isinstance(items,list):
        status['example_rows']=len(items)
        pd.DataFrame(items).to_csv(EXO/'kpx_api_latest_rows.csv',index=False)
    return status

def main():
    p=argparse.ArgumentParser(); p.add_argument('--backfill',action='store_true'); p.add_argument('--kpx-only',action='store_true'); a=p.parse_args()
    EXO.mkdir(exist_ok=True)
    result={'KPX':kpx()}
    if not a.kpx_only: result['ASOS']=asos(a.backfill)
    (EXO/'api_connection_status.json').write_text(json.dumps(dict(at=now(),**result),indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
