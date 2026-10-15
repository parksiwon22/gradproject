"""Optional official annual demand file; not a historical forecast vintage."""
import hashlib
import json
import re
import urllib.request
from v11.store import HOME,now

def main():
    HOME.mkdir(exist_ok=True,parents=True)
    source='https://www.data.go.kr/data/15065266/fileData.do'
    text=urllib.request.urlopen(source,timeout=40).read().decode('utf-8')
    links=re.findall(r'"contentUrl"\s*:\s*"(https://www.data.go.kr/cmm/cmm/fileDownload.do[^"<>]+)"',text)
    if len(links)!=1: raise ValueError('official CSV link not unique; review changed portal schema')
    payload=urllib.request.urlopen(links[0].replace('&amp;','&'),timeout=50).read()
    digest=hashlib.sha256(payload).hexdigest(); folder=HOME/'optional_raw'; folder.mkdir(exist_ok=True)
    path=folder/(digest+'.csv')
    if not path.exists(): path.write_bytes(payload)
    (HOME/'demand_official.csv').write_bytes(payload)
    (HOME/'optional_sources.json').write_text(json.dumps(dict(source=source,url=links[0],retrieved_at=now(),sha256=digest,unit='MWh over hourly intervals',region='national including Jeju',available_at='unknown historical release',use_for_training=False,KPX_SERVICE_KEY='optional; configure locally only after official API application',KMA_SERVICE_KEY='optional; never enter key in chat; retrospective forecast archive still required'),indent=2),encoding='utf-8')
    print('Official optional demand CSV stored locally:',digest)

if __name__=='__main__': main()
