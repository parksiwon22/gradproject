import unittest,json
from pathlib import Path
import numpy as np,pandas as pd
from v12.ev import EV,score
from v15.policies import scenario_optimize,residual_templates,dispatch
from v15.research import HOME,FOLDS

class V15Tests(unittest.TestCase):
    def test_identical_scenarios_allow_real_arbitrage(self):
        t=pd.date_range('2026-10-16',periods=2,freq='h',tz='Asia/Seoul'); ev=EV(capacity=10,initial_soc=50,target_soc=50,min_soc=0,max_soc=100,charge_kw=5,discharge_kw=5,charge_eff=1,discharge_eff=1)
        plan=scenario_optimize([20,100],t,[[20,100],[20,100]],ev,risk_weight=1)
        self.assertAlmostEqual(plan['net_cost'],-400,places=4); self.assertGreater(plan['grid_export_kwh'],4.99)
    def test_finite_scenario_cap_not_future_guarantee(self):
        t=pd.date_range('2026-10-16',periods=2,freq='h',tz='Asia/Seoul'); ev=EV(capacity=10,initial_soc=50,target_soc=50,min_soc=0,max_soc=100,charge_kw=5,discharge_kw=5,charge_eff=1,discharge_eff=1)
        plan=scenario_optimize([20,100],t,[[20,100],[100,20]],ev,risk_weight=0,maxloss=50)
        self.assertLessEqual(max(plan['scenario_excess_losses']),50+1e-4)
    def test_negative_price_energy_partial_window_and_terminal(self):
        t=pd.date_range('2026-10-16',periods=4,freq='h',tz='Asia/Seoul'); ev=EV(capacity=10,initial_soc=30,target_soc=60,min_soc=20,max_soc=90,charge_kw=4,discharge_kw=3,degradation=5)
        p=np.array([-30,100,20,-10]); plan=scenario_optimize(p,t,np.vstack([p,p+10]),ev,t[0]+pd.Timedelta(minutes=15),t[-1]+pd.Timedelta(minutes=45),.25)
        s=plan['schedule']; self.assertAlmostEqual(plan['ending_soc'],60,places=5)
        self.assertTrue(np.allclose(s.soc_end-s.soc_start,(s.charge_kw*.9-s.discharge_kw/.9)*s.connected_hours/10*100,atol=1e-6)); self.assertTrue((np.minimum(s.charge_kw,s.discharge_kw)<1e-6).all())
        self.assertTrue((s.charge_kw<=4+1e-6).all()); self.assertTrue((s.discharge_kw<=3+1e-6).all())
    def test_complete_prior_residual_days_required(self):
        with self.assertRaises(ValueError): residual_templates(np.zeros((3,24)))
        templates=residual_templates(np.random.default_rng(42).normal(size=(40,24))); self.assertEqual(templates.shape,(9,24))
    def test_event_features_no_target_leakage(self):
        from v11.store import connect,latest
        from v15.features import build
        db=connect(); prices=latest(db,'land'); db.close(); targets=pd.date_range('2025-10-01',periods=24,freq='h',tz='Asia/Seoul')
        a,cols=build(prices,targets,False); altered=prices.copy(); altered.loc[altered.timestamp.isin(targets),'smp']=9999
        b,_=build(altered,targets,False); self.assertTrue(np.allclose(a[cols],b[cols]))
    def test_strict_cv_roles(self):
        for fold in FOLDS:
            self.assertLess(fold['train_end'],fold['cal_start']); self.assertLess(fold['cal_end'],fold['val_start']); self.assertLess(fold['val_end'],'2026-01-01')
        if (HOME/'model_study/qualification.json').exists():
            results=json.loads((HOME/'model_study/qualification.json').read_text()); self.assertTrue(all(r['not_promoted'] for r in results['results']))

if __name__=='__main__': unittest.main()
