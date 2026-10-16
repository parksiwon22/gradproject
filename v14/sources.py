"""Public source capture and optional single dedicated-key solar probe."""
import argparse,hashlib,html,json,re,urllib.request
import pandas as pd
from v11.store import now
from v11.authenticated_data import key,call
from v14.research import HOME
URLS={'generation_supply_publication':'https://www.kpx.or.kr/menu.es?mid=a10109010000','price_formation':'https://www.kpx.or.kr/menu.es?mid=a10401040000','fuel_2026q3':'https://www.kepco.co.kr/home/media/newsroom/notice/boardView.do?boardMngNo=14&boardNo=2673','solar_forecast_catalog':'https://www.data.go.kr/data/15144541/openapi.do'}

def run(probe_solar=False):
    folder=HOME/'sources'; folder.mkdir(parents=True,exist_ok=True); rows=[]
    for name,url in URLS.items():
        path=folder/(name+'.html')
        try:
            if path.exists(): payload=path.read_bytes()
            else:
                with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=25) as r: payload=r.read()
                path.write_bytes(payload)
            rows.append(dict(name=name,url=url,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),captured_at=now(),model_use=False,reason='No original historical issue-vintage hourly input series established'))
        except Exception as exc: rows.append(dict(name=name,url=url,status='failed',reason=type(exc).__name__))
    payload=(folder/'fuel_2026q3.html').read_bytes(); text=html.unescape(re.sub('<[^>]+>',' ',payload.decode('utf-8',errors='replace'))); values=[]
    for tax,m in zip(['before_tax','after_tax'],re.finditer(r'LNG\([^|]+\)\s*\|\s*([\d,.]+)\s*\|\s*([\d,.]+)\s*\|\s*([\d,.]+)',text)):
        for month,value in zip(['2026-03','2026-04','2026-05'],m.groups()): values.append(dict(month=month,fuel='LNG',tax=tax,krw_per_kg=float(value.replace(',','')),notice_date='2026-06-19',used_as_input=False))
    pd.DataFrame(values).to_csv(folder/'official_lng_2026_notice.csv',index=False)
    solar=dict(catalog=URLS['solar_forecast_catalog'],endpoint='https://apis.data.go.kr/B552070/forecastService/oamsFile31',status='not_requested',reason='Dedicated EWP_SERVICE_KEY/service approval not configured; D-1 14h forecast is not target-day information available at D-2 23h. Historical original vintages required.',model_use=False)
    if probe_solar:
        credential=key('EWP_SERVICE_KEY')
        if credential:
            try:
                response=call(solar['endpoint'],dict(pageNo=1,numOfRows=1,apiType='json'),credential)
                solar.update(status='response_received',response_type=type(response).__name__)
            except RuntimeError as exc: solar.update(status='failed',reason=str(exc))
        else: solar['status']='key_not_configured'
    (folder/'inventory.json').write_text(json.dumps(rows,indent=2),encoding='utf-8'); (folder/'solar_availability.json').write_text(json.dumps(solar,indent=2),encoding='utf-8')
    print('Official captures',sum('bytes'in r for r in rows),'LNG monthly values',len(values),'solar',solar['status'])

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--probe-solar',action='store_true'); run(p.parse_args().probe_solar)
