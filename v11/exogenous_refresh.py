"""Increment current-year observations and preserve receipt-time vintages."""
import json
import pandas as pd
from v11.exogenous_collect import EXO,ELEMENTS,fetch,save,parse_daily,parse_power
from v11.exogenous_live import record_observations,collect_forecast,issue,issue_weather_proxy,reconcile
from v11.exogenous_model import read_date
from v11.authenticated_data import key,kpx

def main():
    current=pd.Timestamp.now(tz='Asia/Seoul'); status=EXO/'refresh_status.json'
    old=json.loads(status.read_text(encoding='utf-8')) if status.exists() else {}
    day=current.strftime('%Y-%m-%d')
    if current.hour>=11 and old.get('daily_observations_refreshed')!=day:
        combined=None
        for code,col in ELEMENTS.items():
            url=f'https://www.weather.go.kr/w/weather/land/past-obs/obs-by-element.do?stn=108&yy={current.year}&obs={code}'
            payload=fetch(url); save(f'kma_108_{current.year}_{code}.html',payload,url,dict(resolution='daily',variable=col))
            f=parse_daily(payload.decode('utf-8'),current.year,col); combined=f if combined is None else combined.merge(f,on='date',how='outer')
        prior=read_date(EXO/'weather_daily_all.csv','date'); prior=prior[prior.date.dt.year!=current.year]
        pd.concat([prior,combined]).sort_values('date').to_csv(EXO/'weather_daily_all.csv',index=False)
        url='https://epsis.kpx.or.kr/epsisnew/selectEkgeEpsMep.ajax'; payload=fetch(url,dict(beginDate=f'{current.year}0101',endDate=current.strftime('%Y%m%d'),selYear='',selMonth=''))
        save('daily_power_latest_response.js',payload,url,dict(resolution='daily peak-state'))
        f=parse_power(payload.decode('utf-8')); prior=read_date(EXO/'power_daily_all.csv','date'); prior=prior[prior.date.dt.year!=current.year]
        pd.concat([prior,f]).sort_values('date').to_csv(EXO/'power_daily_all.csv',index=False)
        old['daily_observations_refreshed']=day
    record_observations(); collect_forecast()
    if key('KPX_SERVICE_KEY'):
        # At most two API calls per hourly run; no repeated unauthorized KMA calls.
        tomorrow=(current.normalize()+pd.Timedelta(days=1)).strftime('%Y%m%d')
        api=kpx(tomorrow)
        if api.get('connected') and api.get('example_rows',0)==0: api=kpx(current.strftime('%Y%m%d'))
        api_path=EXO/'api_connection_status.json'
        state=json.loads(api_path.read_text(encoding='utf-8')) if api_path.exists() else {}
        state['KPX']=api; api_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
    old['updated_at']=current.isoformat(); status.write_text(json.dumps(old,indent=2),encoding='utf-8')
    print(issue()); print(issue_weather_proxy()); reconcile()

if __name__=='__main__': main()
