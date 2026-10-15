"""Credential-safe APIhub probes; never persist authenticated request URLs."""
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
import pandas as pd
from v11.authenticated_data import key
from v11.store import HOME, now

FOLDER = HOME / 'kma_apihub'
BASE = 'https://apihub.kma.go.kr/api/'

def request(path, params, credential):
    if not credential: return None, {'reason': 'environment_key_unavailable'}
    url = BASE + path + '?' + urllib.parse.urlencode({**params, 'authKey': credential})
    try:
        with urllib.request.urlopen(url, timeout=45) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode('utf-8', errors='replace').lower()
        category = 'permission_or_service_restriction_unresolved'
        if exc.code>=500: category='upstream_service_failure'
        elif exc.code==429: category='service_limit'
        elif exc.code==400: category='request_parameter_rejected'
        elif exc.code==401: category='authentication_rejected'
        elif 'api' in body and ('신청' in body or '승인' in body):
            category = 'service_application_or_approval'
        elif 'key' in body and ('invalid' in body or '인증' in body):
            category = 'authentication_rejected'
        elif 'limit' in body or '초과' in body:
            category = 'service_limit'
        return None, {'http_status': exc.code, 'reason': category, 'body_suppressed': True}
    except Exception as exc:
        return None, {'reason': 'transport_' + type(exc).__name__}
    # Reject echoed secrets entirely, preserving hashes only for safe bodies.
    if any(s.encode() in payload for s in (credential, urllib.parse.quote(credential))):
        return None, {'reason': 'credential_echo_body_suppressed'}
    return payload, {'http_status': 200, 'bytes': len(payload)}

def probe():
    import os
    credential = key('KMA_APIHUB_AUTH_KEY')
    status = dict(at=now(), process_present=bool(os.environ.get('KMA_APIHUB_AUTH_KEY')),
                  user_or_process_readable=bool(credential), tests={})
    if not credential:
        return status
    today = pd.Timestamp.now(tz='Asia/Seoul').normalize()
    day = today-pd.Timedelta(days=2)
    tests = {
        'asos_instant': ('typ01/url/kma_sfctm2.php', dict(tm=day.strftime('%Y%m%d')+'1200', stn=108, help=1)),
        'asos_period': ('typ01/url/kma_sfctm3.php', dict(tm1=day.strftime('%Y%m%d')+'0000', tm2=day.strftime('%Y%m%d')+'2300', stn=108, help=1)),
        'forecast_recent': ('typ02/openApi/VilageFcstInfoService_2.0/getVilageFcst', dict(pageNo=1,numOfRows=1000,dataType='JSON',base_date=(today-pd.Timedelta(days=1)).strftime('%Y%m%d'),base_time='2000',nx=60,ny=127)),
        'forecast_historical_grid': ('typ01/cgi-bin/url/nph-dfs_shrt_grd', dict(tmfc='2023010120',tmef='2023010212',vars='TMP')),
        'forecast_historical_land': ('typ01/url/fct_afs_dl.php', dict(reg='11B10101',tmfc1='2023010120',tmfc2='2023010120',disp=1,help=1)),
        'forecast_historical_land2': ('typ01/url/fct_afs_dl2.php', dict(reg='11B10101',tmfc1='2023010117',tmfc2='2023010117',disp=1,help=1)),
    }
    FOLDER.mkdir(parents=True,exist_ok=True)
    for name,(path,params) in tests.items():
        payload, result = request(path,params,credential)
        if payload is not None:
            digest=hashlib.sha256(payload).hexdigest()
            (FOLDER/(name+'_'+digest+'.raw')).write_bytes(payload)
            result.update(sha256=digest,source=BASE+path,params=params,
                          first_seen=now(),data_rows=sum(bool(re.match(rb'^\d{12}\s',line)) for line in payload.splitlines()))
        status['tests'][name]=result
        print(name,json.dumps(result),flush=True)
    (FOLDER/'connection_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    history=FOLDER/'connection_history'; history.mkdir(exist_ok=True)
    (history/(pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S')+'.json')).write_text(json.dumps(status,indent=2),encoding='utf-8')
    return status

if __name__ == '__main__':
    probe()
