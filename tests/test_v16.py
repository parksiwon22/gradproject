import unittest,json
from pathlib import Path
import numpy as np,pandas as pd
from v16.common import eligible,HOME
from v16.definitions import top6,masks,thresholds
from v16.economics import cycle_margin

class V16Tests(unittest.TestCase):
    def test_receipt_issue_and_original_vintage_all_required(self):
        origin='2026-10-16T23:00:00+09:00';r=dict(kind='forecast',issued_at='2026-10-16T17:00:00+09:00',first_seen='2026-10-16T18:00:00+09:00',original_vintage_verified=True)
        self.assertTrue(eligible(r,origin))
        for patch in [dict(issued_at=None),dict(first_seen='2026-10-17T00:00:00+09:00'),dict(issued_at='2026-10-17T00:00:00+09:00'),dict(original_vintage_verified=False),dict(kind='observation'),dict(issued_at='2026-10-16T17:00:00')]:self.assertFalse(eligible({**r,**patch},origin))
    def test_high_stable_and_low_abrupt_are_distinct(self):
        t=pd.date_range('2025-04-01',periods=24,freq='h',tz='Asia/Seoul');a=np.full(24,180.);f=pd.DataFrame(dict(timestamp=t,target_date='2025-04-01',actual_smp=a,previous_actual_smp=a,daily=a))
        m=masks(f,a,dict(up=20,down=20));self.assertEqual(m['absolute150'][0].sum(),24);self.assertEqual(m['up_ramp'][0].sum(),0);self.assertEqual(m['daily_top6'][0].sum(),0)
        a=np.r_[np.full(12,50.),np.full(12,90.)];f.actual_smp=a;f.previous_actual_smp=np.r_[50,a[:-1]];f.daily=50;m=masks(f,a,dict(up=20,down=20))
        self.assertEqual(m['absolute150'][0].sum(),0);self.assertEqual(m['up_ramp'][0].sum(),1);self.assertEqual(m['known_uplift20'][0].sum(),12)
    def test_predicted_ramp_boundary_excluded_without_actual_input(self):
        t=pd.date_range('2025-04-01',periods=24,freq='h',tz='Asia/Seoul');p=np.arange(24.)
        f=pd.DataFrame(dict(timestamp=t,target_date='2025-04-01',actual_smp=p,previous_actual_smp=9999,daily=0));m=masks(f,p,dict(up=.5,down=.5))
        self.assertEqual(m['up_ramp'][2].sum(),23);self.assertTrue(m['up_ramp'][1][1:].all());self.assertFalse(m['up_ramp'][2][0])
    def test_relative_rank_invariant_to_daily_level(self):
        a=np.arange(24.);self.assertTrue(np.array_equal(top6(a),top6(a+1000)));self.assertEqual(top6(a).sum(),6)
    def test_break_even_includes_efficiencies_wear_and_negative_prices(self):
        r=cycle_margin(100,150,.9,.9,5,1);self.assertAlmostEqual(r['break_even_sell_price'],134.5679012346,places=7)
        self.assertGreater(r['margin_per_export_kwh'],0);self.assertLess(cycle_margin(175,175)['margin_per_export_kwh'],0)
        self.assertGreater(cycle_margin(-100,-10)['margin_per_export_kwh'],0);self.assertIsNone(cycle_margin(10,100,sell_ratio=0)['break_even_sell_price'])
        with self.assertRaises(ValueError):cycle_margin(100,200,charge_eff=0)
    def test_later_posted_maintenance_never_selected_for_earlier_origin(self):
        from v16.maintenance import attach
        times=pd.date_range('2025-04-03',periods=24,freq='h',tz='Asia/Seoul');f=pd.DataFrame(dict(timestamp=times,target_date='2025-04-03',origin_timestamp=pd.Timestamp('2025-04-01T23:00:00+09:00')))
        common=dict(unit='test',fuel='LNG',planned_start='2025-04-03 00:00',planned_end='2025-04-04 00:00',plan_period_start='2025-04-01',plan_period_end='2025-04-08',first_seen='2026-10-16T00:00:00Z')
        records=[dict(**common,document_id='a',capacity_mw=100,posted_at='2025-04-01T12:00:00+09:00'),dict(**common,document_id='b',capacity_mw=9999,posted_at='2025-04-02T12:00:00+09:00')]
        checks=[dict(document_id=x,parsed_rows=1,ambiguous_rows=0,period_start='2025-04-01',period_end='2025-04-08') for x in ['a','b']]
        out=attach(f,records,checks);self.assertEqual(len(out),24);self.assertTrue((out.maintenance_total_mw==100).all());self.assertFalse(out.historical_vintage_certified.any())
    def test_information_oracles_explicitly_nonforecast_and_default_not_promoted(self):
        config=json.loads((HOME/'design_before_scores.json').read_text());self.assertFalse(config['operating_model_replaced'])
        path=HOME/'information/audit.csv'
        if path.exists():self.assertFalse(pd.read_csv(path).is_future_prediction.any())
    def test_observed_information_oracle_invariants(self):
        path=HOME/'information/audit.csv'
        if not path.exists():self.skipTest('Local completed information study required')
        f=pd.read_csv(path,float_precision='round_trip')
        costs=f.pivot(index='day',columns='information',values='actual_G2V_cost')
        self.assertTrue(np.allclose(costs.point_baseline,costs.perfect_daily_mean,atol=1e-6))
        self.assertLess(f[f.information.isin(['perfect_order_only','perfect_shape'])].G2V_regret.abs().max(),1e-5)
        self.assertLess(f[f.information=='perfect_prices'].V2G_regret.abs().max(),1e-5)
    def test_maintenance_train_end_before_validation_and_no_promotion(self):
        path=HOME/'maintenance_study/design_before_scores.json'
        if not path.exists():self.skipTest('Local maintenance study required')
        design=json.loads(path.read_text(encoding='utf-8'))
        for fold in design['folds']:self.assertLess(fold['train_end'],fold['val_start'])
        self.assertTrue(design['no_2026_selection']);self.assertTrue(design['no_operational_candidate_even_if_improved'])
        result=HOME/'maintenance_study/qualification.json'
        if result.exists():self.assertEqual(json.loads(result.read_text())['qualified_operating_models'],[])
    def test_web_buttons_and_v16_explanation(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file('app_v16.py').run(timeout=60);self.assertFalse(app.exception);self.assertEqual(len(app.tabs),6)
        app.button[0].click().run(timeout=60);self.assertFalse(app.exception);self.assertFalse(app.error)
        button=next(b for b in app.button if b.label=='V16 손익분기와 권장 근거 계산');button.click().run(timeout=60);self.assertFalse(app.exception);self.assertFalse(app.error)

if __name__=='__main__':unittest.main()
