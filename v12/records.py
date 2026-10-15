"""Atomically publish a complete JSON record without replacing any existing file."""
import json
import os
from pathlib import Path
import uuid

def write_once(path,record):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(record,ensure_ascii=False,indent=2,default=str).encode('utf-8')
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        # Same-volume hard link is create-only: an existing destination makes
        # this fail atomically. Interrupted writes never publish partial JSON.
        os.link(temporary,path)
    finally:
        if temporary.exists(): temporary.unlink()
