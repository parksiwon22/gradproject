import copy
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from v13.service import audit_record,decision_result,single_run,read,HOME,V12
from v12.ev import EV,optimize,regret

class V13Tests(unittest.TestCase):
    def test_saved_receipts_replay_and_late_input_rejected(self):
        paths=list((V12/'forecasts').glob('*.json'))
        if not paths: self.skipTest('local immutable artifacts unavailable')
        record=read(paths[0]); self.assertTrue(audit_record(record)['valid'])
        tampered=copy.deepcopy(record); tampered['source_prices'][0]['first_seen']='2099-01-01T00:00:00Z'
        with self.assertRaisesRegex(ValueError,'cutoff'): audit_record(tampered)
        tampered=copy.deepcopy(record); key=next(iter(tampered['rows'][0]['inputs'])); tampered['rows'][0]['inputs'][key]+=100
        with self.assertRaisesRegex(ValueError,'replay'): audit_record(tampered)

    def test_partial_window_oracle_uses_saved_connection(self):
        t=pd.date_range('2026-10-16',periods=3,freq='h',tz='Asia/Seoul')
        start=t[0]+pd.Timedelta(minutes=30); end=t[-1]+pd.Timedelta(minutes=30)
        ev=EV(capacity=10,initial_soc=20,target_soc=70,charge_kw=5,charge_eff=1,min_soc=0,max_soc=100)
        p=[1,100,200]; plan=optimize(p,t,ev,start,end)
        record={'context':{'start':start.isoformat(),'end':end.isoformat()},'plans':{'G2V':{**plan,'schedule':plan['schedule'].to_dict('records')}}}
        result=decision_result(record,pd.Series(p,index=t))[0]
        self.assertEqual(result['status'],'evaluated'); self.assertAlmostEqual(result['regret'],0,places=6)
        self.assertGreater(regret(plan,p,t,ev)['regret'],0)
        del record['context']; self.assertEqual(decision_result(record,pd.Series(p,index=t))[0]['status'],'ineligible')

    def test_actuals_missing_never_evaluated(self):
        t=pd.date_range('2026-10-16',periods=2,freq='h',tz='Asia/Seoul'); plan=optimize([1,2],t,EV(capacity=10,initial_soc=30,target_soc=40,min_soc=0))
        record={'plans':{'G2V':{**plan,'schedule':plan['schedule'].to_dict('records')}}}
        self.assertEqual(decision_result(record,pd.Series([1,np.nan],index=t))[0]['status'],'awaiting_actual')

    def test_overlapping_run_rejected(self):
        with single_run():
            with self.assertRaisesRegex(RuntimeError,'overlapping'):
                with single_run(): pass

    def test_intervals_not_backdated_or_additional_samples(self):
        paths=list((HOME/'interval_records').glob('*.json'))
        if not paths: self.skipTest('local annotations unavailable')
        for path in paths:
            r=read(path); self.assertFalse(r['independent_new_point_forecast'])
            self.assertGreaterEqual(pd.Timestamp(r['generated_at']),pd.Timestamp(r['source_issued_at']))
            self.assertLess(pd.Timestamp(r['calibration']['label_latest_timestamp']),pd.Timestamp(r['reference_origin']))
            self.assertEqual(len(r['rows']),24)

    def test_streamlit_button(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app_v13.py')).run(timeout=30)
        self.assertEqual(len(app.exception),0)
        app.button[0].click().run(timeout=30)
        self.assertEqual(len(app.exception),0)
        self.assertEqual(len(app.error),0)

if __name__=='__main__': unittest.main()
