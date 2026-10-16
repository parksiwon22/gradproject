import unittest,json,hashlib
from unittest.mock import patch
import numpy as np,pandas as pd
from v11.store import connect,latest
from v14.features import augment,safe_event_lags
from v14.research import HOME,specifications,ev_metrics,predict
from v12.ev import EV,optimize,regret

class V14Tests(unittest.TestCase):
    def test_registered_budget_and_new_names(self):
        specs=specifications(); self.assertEqual(len({s['name'] for s in specs}),len(specs)); self.assertLessEqual(len(specs)+2,40); self.assertGreaterEqual(len(specs),20)
    def test_live_issue_cannot_be_backdated(self):
        from v14.forward import issue
        with self.assertRaisesRegex(ValueError,'Cannot fabricate'): issue(pd.Timestamp('2025-10-01T23:00:00+09:00'))
    def test_future_observations_do_not_change_price_features(self):
        db=connect(); raw=latest(db,'land'); db.close()
        targets=pd.date_range('2025-10-01',periods=24,freq='h',tz='Asia/Seoul')
        a,groups,_=augment(raw,external=False,targets=targets,include_labels=False)
        altered=raw.copy(); altered.loc[altered.timestamp.isin(targets),'smp']=9999
        b,_,_=augment(altered,external=False,targets=targets,include_labels=False)
        self.assertEqual(len(a),24); self.assertTrue(np.allclose(a[groups['price']],b[groups['price']]))
        for t in targets: self.assertTrue(all(safe_event_lags(t,pd.Timestamp('2025-09-29T23:00:00+09:00')).values()))
    def test_external_models_marked_research_only(self):
        import joblib
        for name in ['lgb_lag_weather','lgb_lag_demand','lgb_lag_supply','lgb_lag_combined']:
            path=HOME/'study'/(name+'.joblib')
            if not path.exists(): self.skipTest('research checkpoints not ready')
            self.assertFalse(joblib.load(path)['operational_candidate'])
    def test_price_and_ev_metrics_are_different_objectives(self):
        t=pd.date_range('2025-10-01',periods=24,freq='h',tz='Asia/Seoul'); y=np.r_[np.ones(12)*50,np.ones(12)*100]
        f=pd.DataFrame(dict(timestamp=t,target_date='2025-10-01',actual_smp=y))
        a,_=ev_metrics(f,y); b,_=ev_metrics(f,y[::-1]); self.assertAlmostEqual(a['G2V_regret'],0,places=5); self.assertGreater(b['G2V_regret'],0)
    def test_same_window_and_negative_price_constraints(self):
        t=pd.date_range('2026-10-16',periods=4,freq='h',tz='Asia/Seoul'); ev=EV(capacity=10,initial_soc=30,target_soc=60,charge_kw=4,discharge_kw=3,min_soc=20,max_soc=90,degradation=5)
        start=t[0]+pd.Timedelta(minutes=15); end=t[-1]+pd.Timedelta(minutes=40)
        p=np.array([-20,30,120,10]); plan=optimize(p,t,ev,start,end,True); s=plan['schedule']
        self.assertTrue((np.minimum(s.charge_kw,s.discharge_kw)<1e-6).all()); self.assertGreaterEqual(plan['ending_soc'],60-1e-6)
        self.assertAlmostEqual(regret(plan,p,t,ev,start,end,True)['regret'],0,places=4)
    def test_model_selection_frozen_before_audit(self):
        path=HOME/'study/selection_before_audit.json'
        if not path.exists(): self.skipTest('selection running')
        selection=json.loads(path.read_text()); self.assertEqual(len(selection['top3']),3); self.assertTrue(selection['not_promoted'])
        design=json.loads((HOME/'study/design_before_scores.json').read_text()); self.assertLess(design['train'][1],design['calibration'][0]); self.assertLess(design['calibration'][1],design['selection'][0]); self.assertLess(design['selection'][1],design['audit'][0])
    def test_streamlit_tabs_and_ev_button(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file('app_v14.py').run(timeout=45); self.assertFalse(app.exception); self.assertEqual(len(app.tabs),4)
        app.button[0].click().run(timeout=60); self.assertFalse(app.exception); self.assertFalse(app.error)
        next(r for r in app.radio if r.label=='자료 모드').set_value('과거 검증 시연').run(timeout=45)
        self.assertFalse(app.exception)
        app.button[0].click().run(timeout=60); self.assertFalse(app.exception); self.assertFalse(app.error)

if __name__=='__main__': unittest.main()
