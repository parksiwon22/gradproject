import hashlib,json,os,uuid
from pathlib import Path
import pandas as pd
from v11.store import ROOT,now
HOME=ROOT/'artifacts/v17'
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8');os.replace(tmp,path)
def csv(path):return pd.read_csv(path,float_precision='round_trip')
def sha(path):
    with Path(path).resolve().open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def preserve(start=False):
    file=HOME/'preservation_before.json'
    if start:
        if file.exists():return
        files={p for p in (ROOT/'artifacts').rglob('*.joblib') if 'v17' not in p.parts}
        for version in ['v11','v12','v13','v14','v15','v16']:
            for p in (ROOT/'artifacts'/version).rglob('*.json'):
                if any('forecast' in part or 'decision' in part for part in p.relative_to(ROOT).parts[:-1]):files.add(p)
        baseline=json.loads((ROOT/'artifacts/v15/preservation_before.json').read_text());files.update(Path(p) for p in baseline['packages']);files.add(ROOT/'data/aligned_hourly_all_features_2023.csv')
        write(file,dict(at=now(),files={str(p):sha(p) for p in sorted(files)}));return
    before=json.loads(file.read_text(encoding='utf-8'))['files'];bad=[p for p,h in before.items() if not Path(p).exists() or sha(p)!=h]
    result=dict(at=now(),checked=len(before),changed=len(bad),unchanged=not bad);write(HOME/'preservation_after.json',result)
    if bad:raise ValueError('Original artifact changed; private baseline retained')
    return result
