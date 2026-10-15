"""Private V12 checkpoint archive with CRC and SHA verification."""
import hashlib
import json
import subprocess
import zipfile
import pandas as pd
from v11.store import ROOT,now
from v12.research import HOME

def main():
    folder=ROOT/'artifacts/v12_backups'; folder.mkdir(exist_ok=True)
    stamp=pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y%m%dT%H%M%S')
    bundle=folder/(stamp+'.bundle')
    subprocess.run(['git','bundle','create',str(bundle),'HEAD'],cwd=ROOT,check=True)
    path=folder/(stamp+'.zip')
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for file in HOME.rglob('*'):
            if file.is_file() and file.suffix!='.tmp': archive.write(file,'v12/'+file.relative_to(HOME).as_posix())
    with zipfile.ZipFile(path) as archive:
        bad=archive.testzip()
        if bad: raise ValueError('backup CRC failed: '+bad)
    (folder/(stamp+'.json')).write_text(json.dumps(dict(at=now(),archive=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_bundle=str(bundle),source_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),v11_dependency='Preserved separately in artifacts/v11 and artifacts/v11_backups; no original V11 files rewritten'),indent=2),encoding='utf-8')
    print('Verified private V12 backup:',path)

if __name__=='__main__': main()
