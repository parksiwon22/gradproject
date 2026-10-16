"""Read-only PDF table extraction. Run with a Python containing pypdf."""
import argparse,json,re,datetime
from pathlib import Path
from pypdf import PdfReader

def extract(folder):
    folder=Path(folder);inventory=json.loads((folder/'inventory.json').read_text(encoding='utf-8'));rows=[];checks=[]
    for record in inventory['records']:
        for attach in record['files']:
            if not attach['pdf']:continue
            path=folder/attach['file'];local=[];bad=[];text=[]
            cache=folder/(path.stem+'.parsed_v2.json')
            if cache.exists():
                item=json.loads(cache.read_text(encoding='utf-8'));rows.extend(item['rows']);checks.append(item['check']);continue
            for page in PdfReader(path).pages:
                body=page.extract_text() or '';text.append(body)
                for line in body.splitlines():
                    date=r'20\d{2}[-.]\d{1,2}[-.]\d{1,2}\s+\d{1,2}:\d{2}'
                    match=re.match(r'^(.*?)\s+([\d,]+(?:\.\d+)?)\s+('+date+r')\s+('+date+r')',line.strip())
                    if match:
                        prefix,capacity,start,end=match.groups()
                        local.append(dict(document_id=record['id'],fuel='LNG' if re.search(r'\bLNG\b',prefix) else 'OTHER_UNCLASSIFIED',unit=prefix,capacity_mw=float(capacity.replace(',','')),planned_start=start.replace('.','-'),planned_end=end.replace('.','-'),posted_at=record['posted_at'],first_seen=record['first_seen'],source_sha256=attach['sha256'],historical_vintage_verified=False))
                    elif len(re.findall(date,line))>=2:bad.append(line)
            full='\n'.join(text)
            period=re.search(r'대상기간\s*:\s*(\d{1,2})\.(\d{1,2}).{0,15}~\s*(\d{1,2})\.(\d{1,2})',full)
            start=end=None
            if period:
                year=int(record['posted_date'][:4]);m1,d1,m2,d2=map(int,period.groups())
                start=datetime.date(year,m1,d1);end=datetime.date(year+(m2<m1),m2,d2)
                # A January plan posted in December refers to next year.
                if m1==1 and record['posted_date'][5:7]=='12':start=start.replace(year=year+1);end=end.replace(year=year+1)
            for row in local:row.update(plan_period_start=str(start) if start else None,plan_period_end=str(end) if end else None)
            (folder/(path.stem+'.txt')).write_text(full,encoding='utf-8');rows.extend(local)
            check=dict(document_id=record['id'],pdf=attach['file'],parsed_rows=len(local),ambiguous_rows=len(bad),period_start=str(start) if start else None,period_end=str(end) if end else None)
            checks.append(check);cache.write_text(json.dumps(dict(rows=local,check=check),ensure_ascii=False),encoding='utf-8')
            if len(checks)%20==0:print('PDFs parsed',len(checks),flush=True)
    (folder/'extracted_rows.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    (folder/'extraction_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print('PDFs',len(checks),'structured rows',len(rows),'ambiguous rows',sum(r['ambiguous_rows'] for r in checks))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();extract(a.folder)
