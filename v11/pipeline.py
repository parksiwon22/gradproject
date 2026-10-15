"""Scheduled collector and forecast evaluation; no automatic model retraining."""
import argparse
import json
import subprocess
import sys
from v11.store import HOME,ROOT,now

def main():
    p=argparse.ArgumentParser(); p.add_argument('--preview',action='store_true'); args=p.parse_args()
    HOME.mkdir(exist_ok=True,parents=True)
    stages=[]
    commands=[['v11.collect'],['v11.public_prices'],['v11.audit'],['v11.forecast']+(['--preview'] if args.preview else [])]
    for command in commands:
        with (HOME/'pipeline.log').open('a',encoding='utf-8') as log:
            log.write('\n'+now()+' '+command[0]+'\n'); log.flush()
            result=subprocess.run([sys.executable,'-m']+command,cwd=ROOT,stdout=log,stderr=log)
        stages.append(dict(stage=command[0],exit_code=result.returncode))
    (HOME/'pipeline_status.json').write_text(json.dumps(dict(at=now(),stages=stages),indent=2),encoding='utf-8')
    if any(s['exit_code'] for s in stages): raise SystemExit(1)

if __name__=='__main__': main()
