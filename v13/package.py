"""Build separate external-share and credential-free private restore copies."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import urllib.parse
import zipfile
import pandas as pd
from v11.store import ROOT,now
from v11.authenticated_data import key

HOME=ROOT/'artifacts/v13'
FORECAST_DIRS=['artifacts/v11/forecasts','artifacts/v11/exogenous/forecasts','artifacts/v11/exogenous/weather_proxy_forecasts','artifacts/v12/forecasts','artifacts/v12/decisions','artifacts/v12/interactive_decisions']
SOURCES=[
 dict(data='KPX hourly SMP',url='https://www.data.go.kr/data/15086088/fileData.do',license='No restriction stated on official dataset page, linked to EPSIS',external='research charts and aggregate errors only; no raw table export'),
 dict(data='KPX SMP and day-ahead demand forecast',url='https://www.data.go.kr/data/15131225/openapi.do',license='No restriction stated on official service page',external='derived aggregates only; issue vintages unknown'),
 dict(data='KPX actual national demand',url='https://www.data.go.kr/data/15065266/fileData.do',license='No restriction stated on official catalog',external='inventory and source link only'),
 dict(data='KMA ASOS and issued weather forecast',url='https://apihub.kma.go.kr/policy.do',license='Service terms include third-party rights; no blanket redistribution inferred',external='no raw ASOS/forecast payload; collection method and counts only'),
 dict(data='Original V5 aligned data/model package',url='https://github.com/parksiwon22/gradproject',license='Original package redistribution scope not established',external='summary metrics only; raw data and model binaries excluded')]

def fs(path):
    path=Path(path).absolute()
    return Path('\\\\?\\'+str(path)) if os.name=='nt' and not str(path).startswith('\\\\?\\') else path

def sha(path):
    h=hashlib.sha256()
    with fs(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def forecast_hashes():
    return {p.relative_to(ROOT).as_posix():sha(p) for d in FORECAST_DIRS for p in (ROOT/d).glob('*.json')}

def secret_values():
    values=[]
    for name in ('KMA_APIHUB_AUTH_KEY','KMA_SERVICE_KEY','KPX_SERVICE_KEY','GITHUB_TOKEN','GH_TOKEN','OPENAI_API_KEY'):
        value=key(name)
        if value:
            for s in (value,urllib.parse.quote(value),urllib.parse.quote_plus(value)):
                values.extend([s.encode(),s.encode('utf-16le')])
    return list(dict.fromkeys(values))

def scan(data,secrets):
    if any(value in data for value in secrets): raise ValueError('Credential detected; content suppressed, package not released')
    # Do not archive a credential file even if its credential is not one of the
    # currently registered keys. Obvious bearer/private-key payloads are barred.
    if re.search(rb'(?m)^-----BEGIN (?:[A-Z ]*PRIVATE KEY)-----\r?$',data):
        raise ValueError('Private key detected; package not released')

def sanitized(text):
    for path in (str(ROOT),ROOT.as_posix(),str(ROOT.parent),ROOT.parent.as_posix()): text=text.replace(path,'<PROJECT>')
    text=re.sub(r'[A-Za-z]:[\\/]+Users[\\/]+[^\s`<>\"]+','<USER_PATH>',text)
    return text

def text(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(value,encoding='utf-8')

def copy(source,destination,secrets):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if source.suffix=='.sqlite':
        src=sqlite3.connect('file:'+source.as_posix()+'?mode=ro',uri=True); dst=sqlite3.connect(destination)
        src.backup(dst); assert dst.execute('PRAGMA quick_check').fetchone()[0]=='ok'; dst.close(); src.close()
        scan(destination.read_bytes(),secrets)
    else:
        data=source.read_bytes(); scan(data,secrets); destination.write_bytes(data)

def pack_verify(staging,path,kind,secrets):
    files=sorted(p for p in staging.rglob('*') if p.is_file())
    records=[]
    for p in files:
        scan(p.read_bytes(),secrets)
        records.append(dict(path=p.relative_to(staging).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    manifest=dict(kind=kind,created_at=now(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files=records,
        excluded=['API credentials and environment values','.git credentials and account files','nested backup archives','Python environments and caches'],
        forecast_preservation='original JSON bytes copied unchanged; source hashes checked before and after packaging')
    text(staging/'MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for p in sorted(staging.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(staging).as_posix())
    return verify_archive(path,manifest,secrets)

def verify_archive(path,manifest,secrets):
    records=manifest['files']; kind=manifest['kind']
    extracted=HOME/'verify'/(('s_' if kind=='EXTERNAL_SHARE' else 'p_')+pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%f'))
    extracted.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None: raise ValueError('ZIP CRC failed')
        for name in z.namelist():
            resolved=(extracted/name).resolve()
            if not resolved.is_relative_to(extracted.resolve()): raise ValueError('unsafe archive member')
        z.extractall(fs(extracted))
    actual={p.relative_to(fs(extracted)).as_posix() for p in fs(extracted).rglob('*') if p.is_file()}
    expected={r['path'] for r in records}|{'MANIFEST.json'}
    if actual!=expected: raise ValueError('extracted file inventory mismatch')
    for r in records:
        p=fs(extracted/r['path'])
        if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']: raise ValueError('extracted hash mismatch')
        scan(p.read_bytes(),secrets)
    if json.loads((extracted/'MANIFEST.json').read_text(encoding='utf-8'))!=manifest: raise ValueError('manifest mismatch')
    digest=sha(path); path.with_suffix('.zip.sha256').write_text(digest+'  '+path.name+'\n',encoding='ascii')
    return dict(kind=kind,path=str(path),bytes=path.stat().st_size,files=len(actual),data_files=len(records),sha256=digest,full_extraction_verified=True,all_file_hashes_verified=True,extracted_copy=str(extracted))

def finish_existing(suffix):
    day=pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y%m%d'); tag=day+('_'+suffix if suffix else '')
    folder=HOME/'packages'; secrets=secret_values(); results=[]
    for name in ('share','private_backup'):
        path=folder/f'gradproject_V12_{name}_{tag}.zip'
        with zipfile.ZipFile(path) as z: manifest=json.loads(z.read('MANIFEST.json'))
        results.append(verify_archive(path,manifest,secrets)); print(name,'fully extracted and verified',flush=True)
    with zipfile.ZipFile(Path(results[1]['path'])) as z: before=json.loads(z.read('FORECAST_ORIGINAL_HASHES.json'))
    after=forecast_hashes()
    if any(after.get(p)!=h for p,h in before.items()): raise ValueError('Original record hashes changed; no preservation claim')
    result=dict(at=now(),share=results[0],private=results[1],preserved_original_record_count=len(before),original_oct17_forecasts=sum(p.startswith('artifacts/v12/forecasts/2026-10-17_') for p in before),all_preexisting_record_hashes_unchanged=True,new_records_during_packaging=len(set(after)-set(before)))
    text(folder/'package_results.json',json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

def main(suffix=''):
    day=pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y%m%d'); stamp=pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y%m%dT%H%M%S')
    folder=HOME/'packages'; folder.mkdir(parents=True,exist_ok=True)
    if suffix and not re.fullmatch(r'[A-Za-z0-9_-]+',suffix): raise ValueError('invalid output suffix')
    tag=day+('_'+suffix if suffix else '')
    share=folder/f'gradproject_V12_share_{tag}.zip'; private=folder/f'gradproject_V12_private_backup_{tag}.zip'
    if share.exists() or private.exists(): raise FileExistsError('Existing packages preserved; choose a separate output run before rebuilding')
    before=forecast_hashes(); v12={p:h for p,h in before.items() if p.startswith('artifacts/v12/forecasts/2026-10-17_')}
    if len(v12)!=4: raise ValueError('Expected four original Oct17 V12 forecasts; do not proceed silently')
    audit=HOME/'preservation'; audit.mkdir(parents=True,exist_ok=True)
    text(audit/(stamp+'_before.json'),json.dumps(before,indent=2))
    secrets=secret_values(); stage=folder/('staging_'+stamp); publicstage=stage/'external_share'; privatestage=stage/'private_restore'
    publicstage.mkdir(parents=True); privatestage.mkdir(parents=True)
    summaries=['V5_REPRODUCTION.md','LOCAL_V6_V10_RESULTS.md','V11_RESULTS.md','V11_FEATURE_AUDIT.md','V11_EXOGENOUS_RESULTS.md','V11_KMA_HOURLY_RESULTS.md','V12_RESULTS.md']
    for name in summaries:
        if (ROOT/'docs'/name).exists(): text(publicstage/'reports'/name,sanitized((ROOT/'docs'/name).read_text(encoding='utf-8')))
    for pattern in ('V11*METRICS.csv','V12*COMPARISON.csv','V11*INVENTORY.csv'):
        for p in (ROOT/'docs').glob(pattern): copy(p,publicstage/'tables'/p.name,secrets)
    # SMP-only derived figures: source licensing checked separately above.
    for base in (ROOT/'artifacts/v12/figures',ROOT/'artifacts/v11/kma_apihub/hourly_study'):
        for p in base.glob('*.png'): copy(p,publicstage/'figures'/(base.name+'_'+p.name),secrets)
    for name in ('example_G2V.csv','example_V2G.csv','ev_summary.csv'):
        p=ROOT/'artifacts/v12/analysis'/name
        if p.exists(): copy(p,publicstage/'ev'/name,secrets)
    text(publicstage/'SOURCES_AND_RIGHTS.json',json.dumps(dict(checked_date=day,sources=SOURCES,note='Provider raw source files, KMA data, trained models, personal interactive dispatches and raw forecast records are excluded. Author-generated research graphics attribute KPX hourly SMP; no redistribution permission inferred for unrelated datasets.'),ensure_ascii=False,indent=2))
    text(publicstage/'README.md','''# 전력예보 V5~V12 외부 공유용 연구 산출물

이 ZIP은 발표·연구검토용이다. 비공개 복구 ZIP과 혼동하지 않는다.

- V5 재현: 2023Q4 2,208행 MAE11.093786, R²0.642919.
- V6~V10: 별도 로컬 실험이며 과거 클라우드 실험의 동일 재현으로 주장하지 않는다.
- V11: 다년 SMP·수요예측·ASOS 수집. 역사 예보 판본/발표시각 한계가 있다.
- V12: 30개 조건 평가, 서로 다른 가격 출력28종. 급등 F1 개선이 일관되지 않아 운영 모델 유지.
- 2026-10-17 사전 예측4개는 별도 비공개 복구본에 원본 그대로 보존했다. 미래 평가 표본이 부족하며 성능 확정 전이다.
- G2V 및 가상 V2G는 SMP 연동 연구 시나리오다. 실제 소매충전료·실증 판매수익이 아니다.

reports/: 연구 요약 및 한계. tables/: 집계 성능표. figures/: 실제/예측·급등·SOC 그래프. ev/: 개인 정보가 없는 고정 연구 조건의 결과.
실제 SMP 출처: 한국전력거래소 EPSIS 및 공공데이터포털15086088. 기상 원자료 및 판본 불명확 자료는 외부 ZIP에서 제외했다. 상세 권리 확인은 SOURCES_AND_RIGHTS.json 참조.

코드: https://github.com/parksiwon22/gradproject/tree/research/v12-forward-ev-20261015 (기준b65baa3).
V13: https://github.com/parksiwon22/gradproject/tree/research/v13-integration-20261016

실행: 저장소 루트에서 python -m streamlit run app_v12.py --server.address=127.0.0.1 --server.port=8502
이 공유본에는 학습 모델/원자료가 없으므로 라이브 예측 재현은 별도 비공개 복구본 또는 본인 권한으로 자료 재수집이 필요하다. 그래프·성능표는 ZIP만으로 검토할 수 있다.

원자료 CSV, API 키, 인증정보, 환경변수 값, 개인 충전조건, 모델 바이너리, 개인 시스템 경로는 포함하지 않았다.
''')
    share_result=pack_verify(publicstage,share,'EXTERNAL_SHARE',secrets); print('External share ZIP verified',share_result['files'],flush=True)
    # Current source snapshot, not .git history or credentials. Include new
    # packager source so the recovery copy explains its own construction.
    tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    sourcepaths={ROOT/p for p in tracked if (ROOT/p).is_file()}
    sourcepaths.update((ROOT/'v13').glob('*.py'))
    excluded=[]
    for p in sorted(sourcepaths):
        if p.name.startswith('.env') or any(x in p.parts for x in ('.git','.venv','.aws','.ssh')): excluded.append(p.relative_to(ROOT).as_posix()); continue
        copy(p,privatestage/'RESTORE'/p.relative_to(ROOT),secrets)
    for root in ('data','local_original','outputs','artifacts/models','artifacts/results','artifacts/research_local','artifacts/v11','artifacts/v12'):
        for p in sorted((ROOT/root).rglob('*')):
            if not p.is_file(): continue
            if any(x in p.parts for x in ('__pycache__','matplotlib-cache','.matplotlib-cache','test_tmp')) or p.suffix in ('.pyc','.log','.tmp') or p.name.endswith(('-wal','-shm')):
                excluded.append(p.relative_to(ROOT).as_posix()); continue
            if p.name.startswith('.env') or p.name.lower() in ('credentials','token.json','tokens.json'): excluded.append(p.relative_to(ROOT).as_posix()); continue
            destination=privatestage/'RESTORE'/p.relative_to(ROOT)
            if not destination.exists(): copy(p,destination,secrets)
    text(privatestage/'FORECAST_ORIGINAL_HASHES.json',json.dumps(before,indent=2))
    text(privatestage/'RESTORE_GUIDE.md','''# 비공개 연구 복구본 — 외부 공유 금지

RESTORE/를 새 작업 디렉터리로 복사한다. 원래 작업 디렉터리에 덮어쓰지 말고 먼저 별도 위치에서 검증한다.
MANIFEST.json의 SHA256 및 파일 수를 확인한다. SQLite는 온라인 백업으로 일관된 별도 복사본을 만들었다. WAL/SHM은 복구할 필요가 없다.
기존 사전 예측 JSON은 원본 바이트를 그대로 포함했다. FORECAST_ORIGINAL_HASHES.json으로 대조한다. 복구 중 기존 예측 파일을 수정하지 않는다.
환경은 RESTORE/environment/installed_packages.json 및 requirements*.txt를 참조해 별도 가상환경에 설치한다.
API 키와 GitHub 인증은 포함하지 않았다. 본인의 승인된 KMA_APIHUB_AUTH_KEY, KMA_SERVICE_KEY, KPX_SERVICE_KEY를 환경변수로 다시 등록한다. 값은 README나 로그에 적지 않는다. 개인 GitHub parksiwon22 인증은 별도로 설정한다.
처음에는 python -m unittest tests.test_v12 -v 로 검증하고 python -m streamlit run app_v12.py 로 로컬 실행한다. 유효한 과거 판본의 재현 한계와 공식가격 모드를 V12_RESULTS.md에서 확인한다.
이 백업은 원본 데이터/모델/개인 연구 의사결정이 포함되므로 공개 GitHub나 외부 공유용으로 전송하지 않는다.
''')
    text(privatestage/'RESTORE/environment/installed_packages.json',json.dumps(dict(python=sys.version,packages=sorted([dict(name=d.metadata['Name'],version=d.version) for d in importlib.metadata.distributions()],key=lambda d:d['name'].lower())),indent=2))
    text(privatestage/'EXCLUDED.json',json.dumps(dict(excluded=excluded,notes=['Existing nested backup ZIPs omitted; underlying assets included instead','Original source files were not moved or deleted','Service keys and authentication deliberately absent']),indent=2))
    private_result=pack_verify(privatestage,private,'PRIVATE_RECOVERY_DO_NOT_SHARE',secrets)
    after=forecast_hashes()
    if any(after.get(p)!=h for p,h in before.items()): raise ValueError('Original forecast/decision record changed during packaging')
    text(audit/(stamp+'_after.json'),json.dumps(after,indent=2))
    result=dict(at=now(),share=share_result,private=private_result,preserved_original_record_count=len(before),original_oct17_forecasts=len(v12),all_preexisting_record_hashes_unchanged=True,new_records_during_packaging=len(set(after)-set(before)))
    text(folder/'package_results.json',json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--suffix',default=''); parser.add_argument('--verify-existing',action='store_true'); args=parser.parse_args()
    finish_existing(args.suffix) if args.verify_existing else main(args.suffix)
