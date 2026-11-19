"""Export real saved E3 forecasts for a static/offline demonstration; no inference."""
from pathlib import Path
import argparse,csv,gzip,hashlib,json,math,statistics
from datetime import datetime,timezone,timedelta

def score(actual,pred):
    errors=[p-y for p,y in zip(pred,actual)];mean=statistics.mean(actual);den=sum((y-mean)**2 for y in actual)
    return dict(mae=statistics.mean(abs(e) for e in errors),rmse=math.sqrt(statistics.mean(e*e for e in errors)),r2=1-sum(e*e for e in errors)/den)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models-root',type=Path,required=True);parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'docs/demo/smp72-2025.json');args=parser.parse_args()
    registry=[];canonical={};provenance=[]
    for name,label in [('sarimax','SARIMAX'),('lstm','LSTM'),('lightgbm','LightGBM'),('xgboost','XGBoost')]:
        root=args.models_root/name/'upgrades/upgrade_72h_20261119';source=root/'outputs/predictions/final_test_predictions.csv.gz';policy=json.loads((root/'config/selection_before_test.json').read_text())
        alpha=policy['blending']['alpha'];assert policy['blending']['baseline']=='baseline_weekly'
        with gzip.open(source,'rt',encoding='utf-8') as file:rows=[r for r in csv.DictReader(file) if r['experiment_id']=='E3']
        assert len(rows)==13104
        rows.sort(key=lambda r:(r['forecast_origin'],int(r['horizon'])))
        actual=[float(r['actual_smp']) for r in rows];pred=[float(r['predicted_smp']) for r in rows];metrics=score(actual,pred)
        registry.append(dict(id=name,label=label,model_weight=alpha,weekly_weight=1-alpha,metrics=metrics,policy='Train+Validation refit + fixed weekly blend',report_url='https://github.com/gemmanzu-a11y/smp-graduation-2026/blob/'+('codex/lstm-sarimax-72h-upgrade-20261119' if name in ['lstm','sarimax'] else 'codex/xgboost-lightgbm-72h-upgrade-20261119')+'/1120gradpj/models/'+name+'/upgrades/upgrade_72h_20261119/reports/final_upgrade_report.md'))
        provenance.append(dict(model=name,source='outputs/predictions/final_test_predictions.csv.gz',sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=len(rows)))
        grouped={}
        for r in rows:grouped.setdefault(r['forecast_origin'],[]).append(r)
        for origin,part in grouped.items():
            assert len(part)==72 and [int(r['horizon']) for r in part]==list(range(1,73))
            o=datetime.fromisoformat(origin);assert o.hour==23
            targets=[datetime.fromisoformat(r['target_timestamp']) for r in part]
            assert all(t==o+timedelta(hours=k) for k,t in enumerate(targets,1))
            base=dict(forecast_origin=o.isoformat(),observed_cutoff=(o-timedelta(hours=72)).isoformat(),targets=[t.isoformat() for t in targets],actual=[float(r['actual_smp']) for r in part],weekly=[float(r['baseline_prediction']) for r in part])
            if origin not in canonical:canonical[origin]=dict(**base,predictions={})
            else:assert all(canonical[origin][k]==v for k,v in base.items())
            canonical[origin]['predictions'][name]=[round(float(r['predicted_smp']),6) for r in part]
        exported=[v for origin in sorted(canonical) for v in canonical[origin]['predictions'][name]]
        assert abs(score(actual,exported)['mae']-metrics['mae'])<1e-6
    bundle=dict(schema_version=1,mode='historical_replay',dataset_version='v3_a90141b81ecd_lag72',generated_at_utc=datetime.now(timezone.utc).isoformat(),presentation_at_kst='2026-11-20T15:00:00+09:00',default_origin='2025-07-01T23:00:00+09:00',timezone='Asia/Seoul',study=dict(train='2023-01-01~2024-12-31',validation='2025-01-01~2025-06-30',refit='2023-01-01~2025-06-30',evaluation='2025-07-01~2025-12-31',evaluation_role='이미 확인한 하반기 구간의 후속 재평가',actual_delay_hours=72,domestic_lng='동월 단가 제공 조건부 실험',generation_2025_missing_pct=99.72602739726028),models=registry,origins=[canonical[k] for k in sorted(canonical)],provenance=provenance)
    assert len(bundle['origins'])==182 and all(len(r['predictions'])==4 for r in bundle['origins'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(bundle,ensure_ascii=False,separators=(',',':'))+'\n')
    print('EXPORTED',len(bundle['origins']),'origins;',args.output.stat().st_size,'bytes')
    print([(r['label'],r['metrics']['mae']) for r in registry])

if __name__=='__main__':main()
