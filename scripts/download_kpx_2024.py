"""Download public mainland hourly SMP; no API credentials needed."""
import hashlib
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

def main():
    target=Path(__file__).resolve().parents[1]/'artifacts/research_local'
    target.mkdir(parents=True,exist_ok=True)
    path=target/'kpx_2024_raw.js'
    if path.exists():
        print('Existing snapshot preserved:',path)
        return
    url='https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpShd.ajax'
    params=dict(beginDate='20231201',endDate='20241231',selYear='',selMonth='',selKind='land',locale='')
    payload=urllib.request.urlopen(url,urllib.parse.urlencode(params).encode(),timeout=60).read()
    # Parse text as data; never execute returned JavaScript.
    from scripts.research_2024 import parse_kpx
    frame=parse_kpx(payload.decode('utf-8'))
    if len(frame)!=397*24: raise ValueError('Unexpected coverage')
    path.write_bytes(payload)
    (target/'download_manifest.json').write_text(json.dumps(dict(url=url,params=params,sha256=hashlib.sha256(payload).hexdigest(),retrieved_at_utc=datetime.now(timezone.utc).isoformat(),market='mainland',unit='KRW/kWh',publication_time='unknown; historical final snapshot'),indent=2),encoding='utf-8')
    print('Saved official snapshot:',path)

if __name__=='__main__': main()
