"""Private local archive with an online SQLite backup; never commits data."""
import hashlib
import json
import sqlite3
import zipfile
from datetime import datetime
from v11.store import HOME,KST,connect,ROOT

def main():
    destination=ROOT/'artifacts/v11_backups'; destination.mkdir(exist_ok=True)
    stamp=datetime.now(KST).strftime('%Y%m%dT%H%M%S')
    temporary=destination/(stamp+'.sqlite')
    db=connect(); target=sqlite3.connect(temporary); db.backup(target); target.close(); db.close()
    extra_temporary=None
    observations=HOME/'exogenous/observations.sqlite'
    if observations.exists():
        extra_temporary=destination/(stamp+'_observations.sqlite')
        source=sqlite3.connect(observations); target=sqlite3.connect(extra_temporary); source.backup(target); target.close(); source.close()
    path=destination/(stamp+'.zip')
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        archive.write(temporary,'prices.sqlite')
        if extra_temporary: archive.write(extra_temporary,'v11/exogenous/observations.sqlite')
        for item in HOME.rglob('*'):
            if item.is_file() and (item.suffix in ['.json','.js','.html','.csv'] or item.name in ['production.joblib','production_2day.joblib','production_exogenous.joblib','production_exogenous_2day.joblib','weather_forecast_proxy_1day.joblib','weather_forecast_proxy_2day.joblib']): archive.write(item,'v11/'+item.relative_to(HOME).as_posix())
    # Verify after closing the writer, so all compression buffers are flushed.
    with zipfile.ZipFile(path,'r') as verification:
        result=verification.testzip()
        if result: raise ValueError('backup checksum failed: '+result)
    temporary.unlink()
    if extra_temporary: extra_temporary.unlink()
    (path.with_suffix('.sha256')).write_text(hashlib.sha256(path.read_bytes()).hexdigest(),encoding='ascii')
    print(path)

if __name__=='__main__': main()
