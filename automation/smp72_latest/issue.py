"""One dated issuance from sealed numeric inputs, with no API, credentials or fitting."""
from pathlib import Path
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
import argparse,hashlib,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from estimator import Estimator
TZ=ZoneInfo('Asia/Seoul')

def verify_inputs():
    seal=json.loads((HERE/'manifest.json').read_text())
    for name,digest in seal['sha256'].items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=digest:raise ValueError('Sealed artifact changed: '+name)
    a=np.load(HERE/'inputs.npz',allow_pickle=False);origin=pd.Timestamp(str(a['origin']));cutoff=pd.Timestamp(str(a['cutoff']))
    if origin!=pd.Timestamp('2026-11-19T23:00:00+09:00') or cutoff!=origin-pd.Timedelta(hours=72):raise ValueError('Wrong dated origin/cutoff')
    if pd.Timestamp(str(a['state_last']))+pd.Timedelta(hours=1)!=cutoff:raise ValueError('State exceeds actual cutoff')
    if len(a['features'])!=83 or a['future_x'].shape!=(145,83) or a['state_x'].shape[1]!=83:raise ValueError('Wrong feature dimensions')
    if pd.Timestamp(seal['input_snapshot_at'])>origin:raise ValueError('Input snapshot made after planned origin')
    for key in ['state_x','state_y','future_x','weekly']:
        if not np.isfinite(a[key]).all():raise ValueError('Nonfinite sealed inputs')
    return a,origin,cutoff

def predict(a,origin,cutoff):
    model=Estimator.load(HERE/'model.npz');model.reset(a['state_y'],a['state_x'],str(a['state_last']));p,_,bridge=model.predict(a['future_x'],origin,cutoff)
    if bridge!=73:raise ValueError('Wrong forecast bridge')
    return p,.4*p+.6*a['weekly']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verify-only',action='store_true');args=ap.parse_args();a,origin,cutoff=verify_inputs()
    path=ROOT/'docs/demo/forecast-20261120.json';base=json.loads(path.read_text());raw,pred=predict(a,origin,cutoff)
    np.testing.assert_allclose(pred,[r['predicted_smp'] for r in base['rows']],rtol=1e-8,atol=1e-6)
    if args.verify_only:print('PASS: sealed model and inputs reproduce all 72 prices; no fitting/API');return
    now=pd.Timestamp.now(tz='Asia/Seoul')
    if now<origin:raise ValueError('Cannot issue before 2026-11-19 23:00 KST')
    if now>origin+pd.Timedelta(hours=15):raise ValueError('Dated issuance window expired; no backdated publication')
    if base['status']=='issued':print('Already issued; keeping immutable published forecast');return
    base['status']='issued';base['calculated_at']=now.isoformat();base['issued_at']=now.isoformat();base['issuance_method']='frozen pre-origin snapshot; actual SARIMAX refilter and inference after planned origin; no refit'
    base['warnings']=[w for w in base['warnings'] if 'Calculated before 23:00' not in w]
    for i,row in enumerate(base['rows']):row.update(calculated_at=now.isoformat(),predicted_smp=float(pred[i]),sarimax_prediction=float(raw[i]),prediction_status='issued')
    path.write_text(json.dumps(base,ensure_ascii=False,indent=2)+'\n')
    status_path=ROOT/'docs/demo/live-status.json';status=json.loads(status_path.read_text());status.update(status='issued',calculated_at=now.isoformat(),issued_at=now.isoformat(),reason='고정한 사전 입력으로 예정 기준시각 이후 실제 재추론·발행 완료. 입력 대체 조건 확인 필요.');status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status='issued',forecast_origin=origin.isoformat(),calculated_at=now.isoformat(),input_snapshot_at=base['input_snapshot_at'],targets=72,training_performed=False),ensure_ascii=False))
if __name__=='__main__':main()
