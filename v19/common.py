from contextlib import contextmanager
import os,pandas as pd
from v18.common import ROOT,read,write,write_once,write_csv,sha,csv,now
HOME=ROOT/'artifacts/v19'

def next_origin(clock):
    clock=pd.Timestamp(clock);bound=clock.normalize()+pd.Timedelta(hours=23)
    return bound if clock<bound else bound+pd.Timedelta(days=1)

@contextmanager
def lock(name):
    import msvcrt
    HOME.mkdir(exist_ok=True,parents=True);f=(HOME/(name+'.lock')).open('a+b')
    try:
        if os.fstat(f.fileno()).st_size==0:f.write(b'0');f.flush()
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:f.close();raise RuntimeError('Concurrent '+name+' rejected') from None
    try:yield
    finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1);f.close()

def preserve(start=False):
    path=HOME/'preservation_before.json'
    if start:
        if path.exists():return
        files=set(read(ROOT/'artifacts/v18/preservation_before.json')['files'])
        for version in ['v17','v18']:
            for p in (ROOT/'artifacts'/version).rglob('*'):
                if p.is_file() and (p.suffix=='.joblib' or (p.suffix=='.json' and any('forecast' in x or 'decision' in x for x in p.parts[:-1]))):files.add(str(p))
        write_once(path,dict(at=now(),files={p:sha(p) for p in sorted(files)}));return
    before=read(path)['files'];changed=[p for p,h in before.items() if not __import__('pathlib').Path(p).exists() or sha(p)!=h];result=dict(at=now(),checked=len(before),changed=len(changed),unchanged=not changed);write(HOME/'preservation_after.json',result)
    if changed:raise ValueError('Protected V5-V18 assets changed')
    return result
