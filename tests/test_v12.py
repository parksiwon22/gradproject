import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from v12.ev import EV,optimize,baseline,regret
from v12.publication import classify,parse_eps_response
from v12.records import write_once
from v11.features import build,FEATURES

class EVTests(unittest.TestCase):
    def times(self,n): return pd.date_range('2026-10-16',periods=n,freq='h',tz='Asia/Seoul')
    def test_g2v_known_optimum_and_baselines(self):
        t=self.times(3); ev=EV(capacity=10,initial_soc=20,target_soc=80,charge_kw=4,charge_eff=1,min_soc=0,max_soc=100)
        plan=optimize([100,10,50],t,ev)
        self.assertAlmostEqual(plan['net_cost'],140)
        self.assertAlmostEqual(plan['ending_soc'],80)
        self.assertAlmostEqual(baseline([100,10,50],t,ev)['net_cost'],420)
        self.assertAlmostEqual(baseline([100,10,50],t,ev,uniform=True)['net_cost'],320)
    def test_v2g_arbitrage_and_wear(self):
        t=self.times(2); ev=EV(capacity=10,initial_soc=50,target_soc=50,charge_kw=5,discharge_kw=5,charge_eff=1,discharge_eff=1,min_soc=0,max_soc=100)
        plan=optimize([10,100],t,ev,v2g=True)
        self.assertAlmostEqual(plan['net_cost'],-450)
        self.assertAlmostEqual(plan['ending_soc'],50)
        ev.degradation=100
        self.assertAlmostEqual(optimize([10,100],t,ev,v2g=True)['net_cost'],0)
    def test_energy_efficiency_soc_power_and_no_simultaneous(self):
        t=self.times(8); ev=EV(capacity=20,initial_soc=30,target_soc=70,charge_kw=5,discharge_kw=3,charge_eff=.8,discharge_eff=.9,min_soc=20,max_soc=90)
        p=optimize([-10,10,200,20,30,150,10,50],t,ev,start=t[0]+pd.Timedelta(minutes=30),end=t[-1]+pd.Timedelta(minutes=15),v2g=True)
        f=p['schedule']; self.assertLessEqual(f.charge_kw.max(),5+1e-6); self.assertLessEqual(f.discharge_kw.max(),3+1e-6)
        self.assertTrue((np.minimum(f.charge_kw,f.discharge_kw)<1e-6).all())
        self.assertTrue((f.soc_end>=20-1e-6).all() and (f.soc_end<=90+1e-6).all())
        self.assertGreaterEqual(p['ending_soc'],70-1e-6)
        delta=((f.charge_kw*.8-f.discharge_kw/.9)*f.connected_hours).sum()
        self.assertAlmostEqual(delta,20*(p['ending_soc']-30)/100)
    def test_regret_uses_actual_price_oracle(self):
        t=self.times(2); ev=EV(capacity=10,initial_soc=0,target_soc=50,charge_kw=5,charge_eff=1,min_soc=0,max_soc=100)
        p=optimize([100,10],t,ev); r=regret(p,[10,100],t,ev)
        self.assertAlmostEqual(r['regret'],450)
    def test_disconnected_hours_and_infeasible_target(self):
        t=self.times(4); ev=EV(capacity=10,initial_soc=20,target_soc=50,charge_kw=3,charge_eff=1,min_soc=0,max_soc=100)
        p=optimize([1,10,20,1],t,ev,start=t[1],end=t[2]); f=p['schedule']
        self.assertEqual(f.charge_kw.iloc[0],0); self.assertEqual(f.charge_kw.iloc[-1],0)
        with self.assertRaises(ValueError): optimize([1,10,20,1],t,ev,start=t[1],end=t[1]+pd.Timedelta(minutes=10))
    def test_bad_inputs_and_missing_prices(self):
        with self.assertRaises(ValueError): optimize([1,np.nan],self.times(2))
        with self.assertRaises(ValueError): optimize([1,2],self.times(2),EV(charge_eff=0))

class ForwardTests(unittest.TestCase):
    def test_unchanged_predictions_are_not_an_improved_candidate(self):
        from v12.research import validation_improved
        baseline=dict(MAE=10.,peak_MAE=30.,EV_regret=5.,peak_F1=.2)
        self.assertFalse(validation_improved(baseline.copy(),baseline))
        self.assertTrue(validation_improved({**baseline,'MAE':9.},baseline))
        self.assertFalse(validation_improved({**baseline,'MAE':12.,'peak_F1':.4},baseline))
    def test_error_html_is_not_successful_absence(self):
        with self.assertRaises(ValueError): parse_eps_response(b'<html>Service unavailable</html>')
        with self.assertRaises(ValueError): parse_eps_response(b'var c1; gridData=[]; gridData.push(broken);')
        self.assertIsNone(parse_eps_response(b'var c1,c2; gridData = [];'))
    def test_not_observed_is_not_universal_nonpublication(self):
        self.assertEqual(classify(None,'2026-10-16'),'not_observed')
        f=pd.DataFrame({'timestamp':pd.date_range('2026-10-16',periods=24,freq='h',tz='Asia/Seoul'),'smp':100})
        self.assertEqual(classify(f,'2026-10-16'),'official_acquired')
        self.assertEqual(classify(f.head(23),'2026-10-16'),'not_observed')
        f.smp=0; self.assertEqual(classify(f,'2026-10-16'),'unverified_all_zero')
    def test_forecast_write_once(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'issued.json'; write_once(p,{'price':100}); original=p.read_bytes()
            with self.assertRaises(FileExistsError): write_once(p,{'price':200})
            self.assertEqual(original,p.read_bytes())
    def test_two_day_feature_no_future_values(self):
        t=pd.date_range('2023-01-01',periods=24*15,freq='h',tz='Asia/Seoul'); data=pd.DataFrame({'timestamp':t,'smp':100+np.sin(np.arange(len(t)))})
        target=pd.date_range('2023-01-13',periods=24,freq='h',tz='Asia/Seoul')
        a=build(data,target,days_ahead=2,include_labels=False)
        data.loc[data.timestamp>pd.Timestamp('2023-01-11 23:00',tz='Asia/Seoul'),'smp']=99999
        b=build(data,target,days_ahead=2,include_labels=False)
        assert_frame_equal(a[FEATURES],b[FEATURES])

class IntegrationTests(unittest.TestCase):
    def test_live_records_remain_immutable_and_pending_is_not_scored(self):
        from v12.research import HOME
        from v12.forward import issue,reconcile
        if not (HOME/'forward_model.joblib').exists(): self.skipTest('private research model not installed')
        originals={p:p.read_bytes() for p in (HOME/'forecasts').glob('*.json')}
        issue(); results=reconcile()
        self.assertTrue(all(p.read_bytes()==body for p,body in originals.items()))
        for row in results:
            if row['status']=='awaiting_actual': self.assertNotIn('MAE',row)
    def test_streamlit_dispatch_flow(self):
        from v12.research import HOME
        if not list((HOME/'forecasts').glob('*.json')): self.skipTest('private forward records not installed')
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file('app_v12.py').run(timeout=30)
        self.assertEqual(len(app.exception),0)
        app.button[0].click().run(timeout=30)
        self.assertEqual(len(app.exception),0); self.assertEqual(len(app.error),0)
        self.assertGreaterEqual(len(app.dataframe),4)

if __name__=='__main__': unittest.main()
