"""Official public exogenous files, preserving resolution and publication limits."""
import argparse
import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from io import StringIO,BytesIO
import numpy as np
import pandas as pd
from v11.store import HOME,KST,now

EXO=HOME/'exogenous'
ELEMENTS={'07':'temperature_mean_c','08':'temperature_max_c','10':'temperature_min_c','06':'wind_mean_ms','12':'humidity_mean_percent','35':'sunshine_hours','59':'cloud_mean_tenths'}
POWER={'c1':'installed_mw','c3':'supply_capacity_mw','c4':'daily_max_demand_mw','c5':'reserve_mw','c6':'reserve_rate_percent'}

def fetch(url,params=None):
    body=urllib.parse.urlencode(params).encode() if params else None
    last=None
    for attempt in range(3):
        try: return urllib.request.urlopen(url,body,timeout=40).read()
        except Exception as exc:
            last=type(exc).__name__
            if attempt<2: time.sleep(2**attempt)
    raise RuntimeError(last+' official retrieval failed')

def save(name,payload,source,metadata=None):
    EXO.mkdir(exist_ok=True); (EXO/name).write_bytes(payload)
    record=dict(source=source,retrieved_at=now(),sha256=hashlib.sha256(payload).hexdigest(),**(metadata or {}))
    vintages=EXO/'raw_vintages'; vintages.mkdir(exist_ok=True)
    vintage=vintages/(record['sha256']+Path(name).suffix)
    if not vintage.exists(): vintage.write_bytes(payload)
    (EXO/(name+'.manifest.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
    return record

def parse_demand(payload):
    for enc in ['utf-8-sig','cp949']:
        try: frame=pd.read_csv(BytesIO(payload),encoding=enc); break
        except UnicodeDecodeError: continue
    frame=frame[frame.iloc[:,0].notna()]
    frame=frame.rename(columns={frame.columns[0]:'date'})
    long=frame.melt(id_vars='date',var_name='ending_hour',value_name='demand_mwh')
    long['hour']=long.ending_hour.str.extract(r'(\d+)').astype(int)-1
    long['timestamp']=pd.to_datetime(long.date).dt.tz_localize('Asia/Seoul')+pd.to_timedelta(long.hour,unit='h')
    long.demand_mwh=pd.to_numeric(long.demand_mwh,errors='raise')
    if long.timestamp.duplicated().any() or not np.isfinite(long.demand_mwh).all(): raise ValueError('invalid demand data')
    return long[['timestamp','demand_mwh']].sort_values('timestamp')

def demand():
    page='https://www.data.go.kr/data/15065266/fileData.do'
    text=fetch(page).decode('utf-8'); save('demand_portal_current.html',text.encode(),page)
    uuid=re.search(r"fn_fileDataDown\('15065266',\s*'([^']+)'",text).group(1)
    history_url='https://www.data.go.kr/tcs/dss/selectHistAndCsvData.do'
    history=fetch(history_url,dict(publicDataPk='15065266',publicDataDetailPk=uuid)).decode('utf-8')
    save('demand_history_current.html',history.encode(),history_url)
    entries=[(2025,uuid)]
    for uid,year in re.findall(r'data-public-pk="([^"]+)".*?전력수요량_(202[34])1231',history,re.S): entries.append((int(year),uid))
    outputs=[]
    for year,uid in entries:
        url='https://www.data.go.kr/tcs/dss/selectFileDataDownload.do'
        meta=json.loads(fetch(url,dict(publicDataDetailPk=uid,publicDataPk='15065266',atchFileId='',fileDetailSn='1',publicDataTyCode='PR0051')))
        if not meta.get('status'): raise ValueError('official download not authorized')
        download='https://www.data.go.kr/cmm/cmm/fileDownload.do?'+urllib.parse.urlencode(dict(atchFileId=meta['atchFileId'],fileDetailSn=meta['fileDetailSn'],insertDataPrcus='N'))
        payload=fetch(download); save(f'demand_{year}_raw.csv',payload,download,dict(geography='nationwide including Jeju',unit='MWh hourly',historical_available_at='unknown; annual file released after year-end'))
        f=parse_demand(payload)
        if not f.timestamp.dt.year.eq(year).all(): raise ValueError('year mismatch in annual demand')
        f.to_csv(EXO/f'demand_{year}.csv',index=False); outputs.append(f)
        print('demand',year,len(f),flush=True)
    pd.concat(outputs).sort_values('timestamp').to_csv(EXO/'demand_hourly_all.csv',index=False)

def parse_daily(text,year,column):
    found=re.search(r'<table[^>]*id="weather_table"[^>]*>(.*?)</table>',text,re.S)
    if not found: raise ValueError('official KMA table missing')
    rows=[]
    for tr in re.findall(r'<tr[^>]*>(.*?)</tr>',found.group(1),re.S):
        cells=[html.unescape(re.sub(r'<[^>]+>','',td)).strip() for td in re.findall(r'<td[^>]*>(.*?)</td>',tr,re.S)]
        if len(cells)!=13 or not re.match(r'^\d+',cells[0]): continue
        # The public service sometimes emits replacement characters in the day
        # suffix; the literal day number and numeric observations remain intact.
        day=int(re.match(r'^\d+',cells[0]).group())
        for month,value in enumerate(cells[1:],start=1):
            try: date=pd.Timestamp(year=year,month=month,day=day,tz='Asia/Seoul')
            except ValueError: continue
            wind=re.search(r"writeWindSpeed\('(-?[\d.]+)'",value)
            number=pd.to_numeric(wind.group(1) if wind else value.replace(',',''),errors='coerce')
            if pd.notna(number): rows.append({'date':date,column:float(number)})
    if not rows: raise ValueError('no daily KMA observations')
    return pd.DataFrame(rows)

def weather():
    all_years=[]
    for year in range(2023,2027):
        combined=None
        for code,col in ELEMENTS.items():
            url='https://www.weather.go.kr/w/weather/land/past-obs/obs-by-element.do?'+urllib.parse.urlencode(dict(stn=108,yy=year,obs=code))
            payload=fetch(url)
            save(f'kma_108_{year}_{code}.html',payload,url,dict(station=108,resolution='daily',variable=col,historical_available_at='unknown; this is retrospective daily observation, never a forecast'))
            f=parse_daily(payload.decode('utf-8'),year,col)
            combined=f if combined is None else combined.merge(f,on='date',how='outer')
        combined=combined.sort_values('date'); combined.to_csv(EXO/f'weather_daily_{year}.csv',index=False); all_years.append(combined)
        print('KMA daily',year,len(combined),flush=True)
    daily=pd.concat(all_years).sort_values('date'); daily.to_csv(EXO/'weather_daily_all.csv',index=False)

def parse_power(text):
    rows=[]
    for block,year,month,day in re.findall(r'(c1\s*=.*?gridData.push\(\{"year":"(\d+)",\s*"month":"(\d+)",\s*"day":"(\d+)")',text,re.S):
        values=dict(re.findall(r'(c\d+)\s*=\s*textFormmat\("([^"]+)",count\)',block))
        row={'date':pd.Timestamp(year=int(year),month=int(month),day=int(day),tz='Asia/Seoul')}
        for key,name in POWER.items(): row[name]=pd.to_numeric(values.get(key),errors='coerce')
        rows.append(row)
    if not rows: raise ValueError('no official daily power rows')
    result=pd.DataFrame(rows).sort_values('date')
    duplicate=result[result.date.duplicated(keep=False)]
    conflict=duplicate.groupby('date').nunique().max(axis=1)
    if (conflict>1).any(): raise ValueError('conflicting duplicate power dates')
    result=result.drop_duplicates('date').reset_index(drop=True)
    return result

def power():
    url='https://epsis.kpx.or.kr/epsisnew/selectEkgeEpsMep.ajax'
    today=pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y%m%d')
    payload=fetch(url,dict(beginDate='20230101',endDate=today,selYear='',selMonth=''))
    save('daily_power_raw.js',payload,url,dict(resolution='daily peak-state',unit='MW/percent',historical_available_at='unknown; observe retrospectively'))
    result=parse_power(payload.decode('utf-8')); result.to_csv(EXO/'power_daily_all.csv',index=False)
    for year,g in result.groupby(result.date.dt.year): g.to_csv(EXO/f'power_daily_{year}.csv',index=False)
    print('official daily power',len(result),flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--only',choices=['demand','weather','power','all'],default='all'); args=p.parse_args(); EXO.mkdir(exist_ok=True)
    for name,action in [('demand',demand),('weather',weather),('power',power)]:
        if args.only in [name,'all']: action()

if __name__=='__main__': main()
