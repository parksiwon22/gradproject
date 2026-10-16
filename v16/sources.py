"""Bounded official captures; no authenticated URL or error body is logged."""
import hashlib,json,os,urllib.request
import pandas as pd
from v11.store import now
from v11.authenticated_data import key,call,KPX
from v11.kma_apihub import request,BASE
from v11.demand_forecasts import parse_week
from v11.kma_hourly import parse
from v16.common import HOME,write,sha,eligible

PUBLIC={
 'demand_week':'https://www.kpx.or.kr/pssInland.es?mid=a10606090100&issue_date=',
 'supply_publication':'https://www.kpx.or.kr/menu.es?mid=a10109010000',
 'generator_operations':'https://www.kpx.or.kr/menu.es?mid=a10402010000',
 'generator_update_catalog':'https://epsis.kpx.or.kr/epsisnew/selectEkccRecentDbList.do?menuId=090203',
 'solar_catalog':'https://www.data.go.kr/data/15144541/openapi.do',
 'fuel_notice':'https://www.kepco.co.kr/home/media/newsroom/notice/boardView.do?boardMngNo=14&boardNo=2673'}

def capture(name,payload,metadata):
    digest=hashlib.sha256(payload).hexdigest(); folder=HOME/'sources/raw'; folder.mkdir(parents=True,exist_ok=True)
    dest=folder/(name+'_'+digest+'.raw'); marker=dest.with_suffix('.json')
    if not dest.exists(): dest.write_bytes(payload)
    prior=json.loads(marker.read_text(encoding='utf-8')) if marker.exists() else {}
    m=dict(name=name,sha256=digest,first_seen=prior.get('first_seen',now()),last_seen=now(),raw=str(dest.relative_to(HOME)),**metadata)
    m['revision_status']='first_observed_version_not_original_release' if not prior else 'same_hash_seen_again'
    write(marker,m); return m

def run():
    (HOME/'sources').mkdir(parents=True,exist_ok=True); clock=pd.Timestamp.now(tz='Asia/Seoul'); day=clock.normalize(); rows=[]
    for name,base in PUBLIC.items():
        url=base+((day+pd.Timedelta(days=1)).strftime('%Y-%m-%d') if name=='demand_week' else '')
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=25) as r: payload=r.read()
            m=capture(name,payload,dict(source=url,kind='forecast' if name=='demand_week' else 'documentation_or_retrospective',issued_at=None,original_vintage_verified=False,D2_eligible=False))
            if name=='demand_week':
                f=parse_week(payload.decode('utf-8'),(day+pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
                f['first_seen']=m['first_seen']; f['issued_at']=None; f.to_csv(HOME/'sources/demand_captured.csv',index=False)
                m.update(rows=len(f),target_dates=sorted(f.timestamp.dt.strftime('%Y-%m-%d').unique()),reason='Archive forecast values, issue/version timing unknown; first receipt now cannot replay past D2')
            rows.append(m)
        except Exception as exc: rows.append(dict(name=name,status='failed',reason=type(exc).__name__,source=url))
    credential=key('KMA_APIHUB_AUTH_KEY')
    tests={
      'asos_recent':('typ01/url/kma_sfctm2.php',dict(tm=(day-pd.Timedelta(days=1)).strftime('%Y%m%d')+'1200',stn='108:133:143:156:159:105:184',help=0)),
      'grid_recent':('typ01/cgi-bin/url/nph-dfs_shrt_grd',dict(tmfc=day.strftime('%Y%m%d')+'17',tmef=(day+pd.Timedelta(days=2)).strftime('%Y%m%d')+'12',vars='TMP')),
      'grid_historical':('typ01/cgi-bin/url/nph-dfs_shrt_grd',dict(tmfc='2025010120',tmef='2025010312',vars='TMP')),
      'forecast_json_recent':('typ02/openApi/VilageFcstInfoService_2.0/getVilageFcst',dict(pageNo=1,numOfRows=1000,dataType='JSON',base_date=day.strftime('%Y%m%d'),base_time='1700',nx=60,ny=127))}
    for name,(path,params) in tests.items():
        payload,status=request(path,params,credential); result=dict(name=name,source=BASE+path,params=params,checked_at=now(),process_key_present=bool(os.environ.get('KMA_APIHUB_AUTH_KEY')),user_or_process_key_readable=bool(credential),**status)
        if payload is not None:
            m=capture(name,payload,dict(source=BASE+path,params=params,kind='observation' if name=='asos_recent' else 'forecast_unvalidated',issued_at=None,original_vintage_verified=False,D2_eligible=False)); result.update(m)
            if name=='asos_recent':
                f=parse(payload); f.to_csv(HOME/'sources/asos_sample.csv',index=False); result['rows']=len(f)
            elif name=='forecast_json_recent':
                try:
                    data=json.loads(payload); code=str(data.get('response',{}).get('header',{}).get('resultCode'))
                    items=data.get('response',{}).get('body',{}).get('items',{}).get('item',[])
                    result.update(resultCode=code,rows=len(items))
                    if code=='00' and items:
                        f=pd.DataFrame(items); f['issued_at']=pd.to_datetime(f.baseDate+f.baseTime,format='%Y%m%d%H%M').dt.tz_localize('Asia/Seoul'); f['target_timestamp']=pd.to_datetime(f.fcstDate+f.fcstTime,format='%Y%m%d%H%M').dt.tz_localize('Asia/Seoul'); f['first_seen']=m['first_seen']
                        f.to_csv(HOME/'sources/issued_weather_recent.csv',index=False)
                        result.update(kind='forecast',issued_at=str(f.issued_at.min()),target_dates=sorted(f.target_timestamp.dt.strftime('%Y-%m-%d').unique()),original_vintage_verified=True)
                        target=day+pd.Timedelta(days=2); origin=target-pd.Timedelta(days=2)+pd.Timedelta(hours=23)
                        result['D2_eligible']=eligible(result,origin)
                except (ValueError,KeyError,TypeError): result['reason']='forecast_schema_or_error_response_unvalidated'
            else: result['reason']='Grid bytes received; archived issue metadata and grid decoding not yet validated'
        rows.append(result)
        print(name,result.get('http_status'),result.get('reason',result.get('rows','received')),flush=True)
    # Correct dedicated credential, single attempt; no assumption of shared service approval.
    credential=key('KPX_SERVICE_KEY'); result=dict(name='kpx_authenticated',source=KPX,status='key_missing',D2_eligible=False)
    if credential:
        try:
            data=call(KPX,dict(pageNo=1,numOfRows=100,dataType='json',date=(day+pd.Timedelta(days=1)).strftime('%Y%m%d')),credential)
            m=capture('kpx_authenticated',json.dumps(data).encode(),dict(source=KPX,kind='forecast_unvalidated',issued_at=None,original_vintage_verified=False,D2_eligible=False))
            result.update(m,status='response_received',resultCode=str(data.get('response',{}).get('header',{}).get('resultCode')))
        except RuntimeError as exc: result.update(status='failed',reason=str(exc))
    rows.append(result)
    solar_key=bool(key('EWP_SERVICE_KEY')); rows.append(dict(name='solar_dedicated_key',key_present=solar_key,status='not_requested',D2_eligible=False,reason='D-1 product and no verified D2 target original vintage; dedicated subscription required'))
    write(HOME/'sources/inventory.json',dict(at=now(),attempts_per_endpoint=1,rows=rows,historical_D2_training_dataset_ready=False,new_model_training_authorized_by_data_gate=False))
    public=[{k:r.get(k) for k in ['name','source','kind','http_status','status','reason','rows','issued_at','first_seen','target_dates','D2_eligible','original_vintage_verified']} for r in rows]
    write(HOME.parent.parent/'docs/V16_SOURCE_AVAILABILITY.json',public)
    print('Official captures complete; no historical eligible training dataset asserted',flush=True)

if __name__=='__main__': run()
