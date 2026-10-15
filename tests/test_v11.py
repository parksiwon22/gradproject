import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from v11.store import connect,ingest,latest,parse,validate,HOME
from v11.collect import collect_range
from v11.features import build,FEATURES
from v11.forecast import write_once
from v11.public_prices import parse_public
from v11.evaluate import metrics
TEMP=HOME/'test_tmp'
TEMP.mkdir(parents=True,exist_ok=True)

def frame():
    return pd.DataFrame({'timestamp':pd.date_range('2026-01-01',periods=24,freq='h',tz='Asia/Seoul'),'smp':np.arange(24,dtype=float)})

class V11Tests(unittest.TestCase):
    def test_vintages_markets_and_first_seen(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as tmp:
            db=connect(tmp); f=frame()
            r=ingest(db,'land',f,b'v1','official',tmp,'2026-01-02T00:00:00+00:00'); self.assertEqual(r['new'],24)
            self.assertEqual(ingest(db,'land',f,b'v1','official',tmp,'2026-01-03T00:00:00+00:00')['unchanged'],24)
            changed=f.copy(); changed.loc[0,'smp']=-5
            self.assertEqual(ingest(db,'land',changed,b'v2','official',tmp,'2026-01-04T00:00:00+00:00')['corrected'],1)
            self.assertEqual(latest(db,'land',as_of='2026-01-03T12:00:00+00:00').smp.iloc[0],0)
            self.assertEqual(latest(db,'land').smp.iloc[0],-5)
            self.assertTrue(latest(db,'jeju').empty)
            self.assertEqual(db.execute('SELECT count(*) FROM prices').fetchone()[0],25)
            db.close()

    def test_api_error_retry_preserves_data(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as tmp:
            db=connect(tmp); ingest(db,'land',frame(),b'v1','official',tmp)
            calls=[]
            def failure(*args): calls.append(args); raise TimeoutError('offline')
            with self.assertRaises(RuntimeError): collect_range(db,'land','2026-01-01','2026-01-02',tmp,fetch=failure,sleep=lambda _:None)
            self.assertEqual(len(calls),3); self.assertEqual(len(latest(db)),24)
            self.assertEqual(db.execute("SELECT count(*) FROM attempts WHERE status='failure'").fetchone()[0],3)
            db.close()

    def test_integrity_and_time_ending(self):
        block=''.join(f'c{i} = textFormmat("{-i}",count);' for i in range(1,28))+'gridData.push({"Date":"2024/02/29"'
        f=parse(block); self.assertEqual(len(f),24); self.assertEqual(f.timestamp.iloc[-1].hour,23); self.assertEqual(f.smp.iloc[0],-1)
        for bad in ['empty',block+block]:
            with self.assertRaises(ValueError): parse(bad)
        with self.assertRaises(ValueError): validate(frame().iloc[:-1])
        bad=frame(); bad.loc[0,'smp']=np.nan
        with self.assertRaises(ValueError): validate(bad)

    def test_future_mutation_cannot_change_features(self):
        raw=pd.DataFrame({'timestamp':pd.date_range('2026-01-01',periods=15*24,freq='h',tz='Asia/Seoul'),'smp':np.arange(15*24,dtype=float)})
        for ahead in [1,2]:
            a=build(raw,days_ahead=ahead)
            cutoff=pd.Timestamp('2026-01-11 23:00',tz='Asia/Seoul')
            changed=raw.copy(); changed.loc[changed.timestamp>cutoff,'smp']+=100000
            b=build(changed,days_ahead=ahead)
            indexes=a.origin_timestamp==cutoff
            np.testing.assert_array_equal(a.loc[indexes,FEATURES],b.loc[indexes,FEATURES])
            self.assertEqual(int(indexes.sum()),24)
            self.assertTrue((a.origin_timestamp<a.timestamp).all())

    def test_public_page_absence_is_not_known(self):
        empty='<input value="2026-10-17">'
        self.assertIsNone(parse_public(empty,'2026-10-17'))
        with self.assertRaises(ValueError): parse_public(empty,'2026-10-16')
        body=empty+''.join(f'<tr><td>{i}h</td><td>999</td><td>{i}</td></tr>' for i in range(1,25))
        parsed=parse_public(body,'2026-10-17'); self.assertEqual(parsed.smp.iloc[0],1)

    def test_forecast_files_are_immutable(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as tmp:
            path=Path(tmp)/'record.json'; write_once(path,{'prediction':1})
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(FileExistsError): write_once(path,{'prediction':2})
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)

    def test_oracle_and_peak_hour(self):
        y=np.arange(24)+140
        f=frame().assign(actual_smp=y,target_date='2026-01-01',previous_actual_smp=np.r_[139,y[:-1]])
        result=metrics(f,y)
        self.assertEqual(result['EV_regret'],0); self.assertEqual(result['peak_hour_MAE'],0); self.assertEqual(result['peak_F1'],1)

    def test_executed_split_manifest(self):
        path=HOME/'training/selection_before_test.json'
        if not path.exists(): self.skipTest('local training assets absent')
        record=json.loads(path.read_text(encoding='utf-8')); self.assertFalse(record['test_used_for_selection'])
        config=json.loads((HOME/'training/config_before_evaluation.json').read_text(encoding='utf-8'))
        self.assertLess(config['validation'][1],config['test'][0])
        experiments=json.loads((HOME/'training/experiments.json').read_text(encoding='utf-8'))
        self.assertEqual(len(experiments),24)
        for e in experiments:
            self.assertEqual(e['status'],'success'); self.assertLess(e['train_end'],config['validation'][0])

class V11WebTests(unittest.TestCase):
    def test_real_local_app(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file('app_v11.py').run(timeout=45)
        self.assertEqual(len(app.exception),0)
        if list((HOME/'forecasts').glob('*.json')):
            self.assertGreater(len(app.selectbox),0)
            app.selectbox[0].select(1).run(timeout=45)
            self.assertEqual(len(app.exception),0)
            app.slider[0].set_value(90).run(timeout=45)
            self.assertEqual(len(app.exception),0)

if __name__=='__main__': unittest.main()
