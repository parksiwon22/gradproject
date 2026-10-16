"""Bounded, resumable public weekly maintenance-plan archive, not fault telemetry."""
import argparse,html,json,re,time,urllib.request
from v16.common import HOME,write,sha
from v16.sources import capture
from v11.store import now

BASE='https://www.kpx.or.kr'
LIST=BASE+'/board.es?bid=0018&mid=a10109030500'

def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0','Accept-Language':'ko-KR,ko;q=0.9'}),timeout=25) as response:return response.read()

def decode(payload):
    try:return payload.decode('utf-8')
    except UnicodeDecodeError:return payload.decode('cp949',errors='replace')

def collect(max_pages=23):
    folder=HOME/'outages';folder.mkdir(parents=True,exist_ok=True);inventory=[];failure=[]
    for page in range(1,max_pages+1):
        cache=folder/f'list_{page}.html'
        if not cache.exists():cache.write_bytes(fetch(LIST+f'&nPage={page}'))
        found=[]
        for block in re.findall(r'<tr[^>]*>(.*?)</tr>',decode(cache.read_bytes()),re.S):
            m=re.search(r'href="([^"]*act=view[^"]*list_no=([^&"\s]+)[^"]*)"',block)
            date=re.search(r'(20\d{2})/(\d{2})/(\d{2})',block)
            if m and date:found.append((html.unescape(m[1]),m[2],'-'.join(date.groups())))
        if not found:failure.append(dict(page=page,reason='no_archive_items'));break
        for link,ident,date in found:
            if not '2023-01-01'<=date<='2026-12-31':continue
            marker=folder/(ident+'.json')
            if marker.exists():inventory.append(json.loads(marker.read_text(encoding='utf-8')));continue
            try:
                url=BASE+link;payload=fetch(url);text=decode(payload)
                page_m=capture('outage_page_'+ident,payload,dict(source=url,kind='forecast_document',issued_at=None,original_vintage_verified=False,D2_eligible=False))
                issued=re.search(r'(20\d{2}/\d{2}/\d{2})\s+(\d{2}:\d{2})',text)
                posted=(issued[1].replace('/','-')+'T'+issued[2]+':00+09:00') if issued else None
                links=list(dict.fromkeys(html.unescape(x) for x in re.findall(r'href="([^"]*boardDownload[^\"]+)"',text)))
                files=[]
                for i,attach in enumerate(links):
                    blob=fetch(BASE+attach);digest=__import__('hashlib').sha256(blob).hexdigest()
                    ext='.pdf' if blob.startswith(b'%PDF') else '.bin';dest=folder/(ident+'_'+str(i)+'_'+digest+ext)
                    if not dest.exists():dest.write_bytes(blob)
                    files.append(dict(file=dest.name,sha256=digest,bytes=len(blob),source=BASE+attach,pdf=ext=='.pdf'))
                title=re.search(r'<title>(.*?)</title>',text,re.S)
                r=dict(id=ident,board_url=url,posted_date=date,posted_at=posted,first_seen=now(),kind='announced_maintenance_plan_not_actual_outage',original_vintage_verified=False,revision='title_mentions_revision' if title and '수정' in title[1] else 'unknown_attachment_replacement_history',D2_historical_certified=False,board_sha256=page_m['sha256'],files=files)
                write(marker,r);inventory.append(r)
            except Exception as exc:failure.append(dict(id=ident,reason=type(exc).__name__))
            time.sleep(.12)
        print('maintenance page',page,'captured',len(inventory),'failures',len(failure),flush=True)
        write(folder/'inventory.json',dict(at=now(),max_pages=max_pages,records=inventory,failures=failure,warning='Posted time is not proof current attachment bytes equal original issue. First local receipt is current. Do not certify historical as-of availability.'))
        if min(date for _,_,date in found)<'2023-01-01':break
    return inventory

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--max-pages',type=int,default=23);a=p.parse_args();collect(a.max_pages)
