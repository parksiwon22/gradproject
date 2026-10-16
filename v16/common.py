import hashlib,json,os,uuid
from pathlib import Path
import pandas as pd
from v11.store import ROOT,now

HOME=ROOT/'artifacts/v16'

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str),encoding='utf-8'); os.replace(temp,path)

def csv(path): return pd.read_csv(path,float_precision='round_trip')

def sha(path):
    with Path(path).resolve().open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def eligible(row,origin):
    """Unknown issue or receipt timing cannot certify a historical input."""
    if row.get('kind')!='forecast' or not row.get('issued_at') or not row.get('first_seen'): return False
    try:
        issue,receipt,bound=map(pd.Timestamp,[row['issued_at'],row['first_seen'],origin])
        if any(t.tzinfo is None for t in [issue,receipt,bound]): return False
        return bool(issue<=bound and receipt<=bound and row.get('original_vintage_verified') is True)
    except (ValueError,TypeError): return False

def preservation(start=False):
    path=HOME/'preservation_before.json'
    if start:
        if path.exists(): return
        files=set()
        for p in (ROOT/'artifacts').rglob('*.joblib'):
            if 'v16' not in p.parts: files.add(p)
        for version in ['v11','v12','v13','v14','v15']:
            for p in (ROOT/'artifacts'/version).rglob('*.json'):
                if any('forecast' in part or 'decision' in part for part in p.relative_to(ROOT).parts[:-1]): files.add(p)
        for name in ['data/aligned_hourly_all_features_2023.csv']:
            files.add(ROOT/name)
        previous=json.loads((ROOT/'artifacts/v15/preservation_before.json').read_text())
        files.update(Path(p) for p in previous['packages'])
        write(path,dict(at=now(),files={str(p):sha(p) for p in sorted(files)})); return
    before=json.loads(path.read_text(encoding='utf-8'))['files']; bad=[p for p,digest in before.items() if not Path(p).exists() or sha(p)!=digest]
    result=dict(at=now(),checked=len(before),changed=len(bad),unchanged=not bad)
    write(HOME/'preservation_after.json',result)
    if bad: raise RuntimeError('Original asset changed; see private baseline paths')
    return result
