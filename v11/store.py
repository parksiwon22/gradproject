"""Append-only official price vintages and collection state."""
import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
HOME=ROOT/'artifacts/v11'
KST=ZoneInfo('Asia/Seoul')

def now(): return datetime.now(timezone.utc).isoformat()

def connect(home=HOME):
    home=Path(home); home.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(home/'prices.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''CREATE TABLE IF NOT EXISTS snapshots (hash TEXT PRIMARY KEY, source TEXT, retrieved_at TEXT, path TEXT);
    CREATE TABLE IF NOT EXISTS prices (market TEXT, interval_start TEXT, price REAL, first_seen TEXT, last_seen TEXT, revision INTEGER, snapshot TEXT, PRIMARY KEY(market,interval_start,revision));
    CREATE TABLE IF NOT EXISTS attempts (at TEXT,market TEXT,begin_date TEXT,end_date TEXT,status TEXT,detail TEXT);
    CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY,value TEXT);''')
    return db

def parse(text):
    # Extract literal numbers only, never eval site JavaScript.
    rows=[]
    pattern=r'((?:\s*c\d+\s*=\s*textFormmat\("[^"]*",count\);\s*)+)gridData.push\(\{"Date":"(\d{4}/\d{2}/\d{2})"'
    for block,day in re.findall(pattern,text):
        fields=dict(re.findall(r'c(\d+)\s*=\s*textFormmat\("([^"]*)",count\)',block))
        if not all(str(h) in fields for h in range(1,25)): raise ValueError('incomplete 24-hour official day')
        for h in range(1,25): rows.append((pd.Timestamp(day,tz='Asia/Seoul')+pd.Timedelta(hours=h-1),float(fields[str(h)].replace(',',''))))
    if not rows: raise ValueError('empty/unrecognized official response')
    result=pd.DataFrame(rows,columns=['timestamp','smp']).sort_values('timestamp').reset_index(drop=True)
    validate(result)
    return result

def validate(frame):
    if frame.empty or frame.timestamp.duplicated().any() or not np.isfinite(frame.smp).all(): raise ValueError('empty, duplicate, or non-finite prices')
    # Negative official prices are preserved, particularly in the separate Jeju market.
    if frame.timestamp.dt.tz is None: raise ValueError('timezone required')
    counts=frame.groupby(frame.timestamp.dt.date).size()
    if not counts.eq(24).all(): raise ValueError('partial day')
    for _,g in frame.groupby(frame.timestamp.dt.date):
        if g.timestamp.dt.hour.tolist()!=list(range(24)): raise ValueError('invalid hour definition')

def ingest(db,market,frame,payload,source,home=HOME,retrieved=None):
    if market not in ('land','jeju'): raise ValueError('invalid market')
    validate(frame)
    stamp=retrieved or now(); digest=hashlib.sha256(payload).hexdigest()
    raw=Path(home)/'raw'; raw.mkdir(parents=True,exist_ok=True)
    path=raw/(digest+'.js')
    if not path.exists(): path.write_bytes(payload)
    new=corrected=unchanged=0
    with db:
        db.execute('INSERT OR IGNORE INTO snapshots VALUES (?,?,?,?)',(digest,source,stamp,str(path)))
        for row in frame.itertuples():
            ts=row.timestamp.isoformat()
            old=db.execute('SELECT price,revision FROM prices WHERE market=? AND interval_start=? ORDER BY revision DESC LIMIT 1',(market,ts)).fetchone()
            if old and old[0]==row.smp:
                db.execute('UPDATE prices SET last_seen=? WHERE market=? AND interval_start=? AND revision=?',(stamp,market,ts,old[1])); unchanged+=1
            else:
                rev=old[1]+1 if old else 1
                db.execute('INSERT INTO prices VALUES (?,?,?,?,?,?,?)',(market,ts,float(row.smp),stamp,stamp,rev,digest))
                corrected+=int(old is not None); new+=int(old is None)
    return dict(new=new,corrected=corrected,unchanged=unchanged,sha256=digest)

def latest(db,market='land',as_of=None):
    clause=' AND first_seen <= ?' if as_of else ''
    args=[market]+([as_of] if as_of else [])
    data=pd.read_sql_query('SELECT * FROM (SELECT *,ROW_NUMBER() OVER (PARTITION BY market,interval_start ORDER BY revision DESC) rn FROM prices WHERE market=?'+clause+') WHERE rn=1 ORDER BY interval_start',db,params=args)
    if data.empty: return pd.DataFrame(columns=['timestamp','smp','first_seen','snapshot'])
    return data.rename(columns={'interval_start':'timestamp','price':'smp'}).assign(timestamp=lambda x:pd.to_datetime(x.timestamp,utc=True).dt.tz_convert('Asia/Seoul'))

def export(db,home=HOME):
    summary=[]
    for market in ('land','jeju'):
        data=latest(db,market)
        if data.empty: continue
        for year,g in data.groupby(data.timestamp.dt.year):
            g.to_csv(Path(home)/f'{market}_{year}.csv',index=False)
            expected=len(pd.date_range(f'{year}-01-01',f'{year}-12-31 23:00',freq='h'))
            coverage=pd.date_range(g.timestamp.min(),g.timestamp.max(),freq='h')
            summary.append(dict(market=market,year=int(year),hours=len(g),days=int(g.timestamp.dt.date.nunique()),expected_full_year=expected,missing_between_endpoints=len(coverage.difference(pd.DatetimeIndex(g.timestamp))),latest=g.timestamp.max().isoformat(),exogenous_status='SMP-only',publication_time='unknown; first_seen is current collection time'))
    (Path(home)/'coverage.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary
