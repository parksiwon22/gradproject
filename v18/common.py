import json
from pathlib import Path
from v11.store import ROOT,now
from v17.common import write,sha,csv
from v12.records import write_once
HOME=ROOT/'artifacts/v18'

def read(path,default=None):
    return json.loads(Path(path).read_text(encoding='utf-8-sig')) if Path(path).exists() else default

def preserve(start=False):
    manifest=HOME/'preservation_before.json'
    if start:
        if manifest.exists():return
        files=set()
        for version in range(11,18):
            base=ROOT/'artifacts'/('v'+str(version))
            for p in base.rglob('*'):
                if p.is_file() and (p.suffix=='.joblib' or (p.suffix=='.json' and any('forecast' in s or 'decision' in s for s in p.relative_to(base).parts[:-1]))):files.add(p)
        old=read(ROOT/'artifacts/v17/preservation_before.json')['files'];files.update(Path(p) for p in old)
        write_once(manifest,dict(at=now(),files={str(p):sha(p) for p in sorted(files)}));return
    before=read(manifest)['files'];bad=[p for p,h in before.items() if not Path(p).exists() or sha(p)!=h];r=dict(at=now(),checked=len(before),changed=len(bad),unchanged=not bad);write(HOME/'preservation_after.json',r)
    if bad:raise ValueError('Preserved original hash mismatch')
    return r
