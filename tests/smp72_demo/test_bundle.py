"""Verify the published real prediction bundle without model dependencies."""
import json,math,unittest
from pathlib import Path
from datetime import datetime,timedelta
ROOT=Path(__file__).resolve().parents[2]
class BundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.d=json.loads((ROOT/'docs/demo/smp72-2025.json').read_text())
    def test_all_models_have_identical_continuous_targets(self):
        d=self.d;self.assertEqual(d['mode'],'historical_replay');self.assertEqual(len(d['origins']),182)
        self.assertEqual({m['id'] for m in d['models']},{'sarimax','lstm','xgboost','lightgbm'})
        for o in d['origins']:
            origin=datetime.fromisoformat(o['forecast_origin']);self.assertEqual(origin.hour,23)
            self.assertEqual(datetime.fromisoformat(o['observed_cutoff']),origin-timedelta(hours=72))
            self.assertEqual(len(o['targets']),72)
            for i,t in enumerate(o['targets']):
                self.assertEqual(datetime.fromisoformat(t),origin+timedelta(hours=i+1))
                self.assertGreaterEqual(t[:10],'2025-07-01');self.assertLessEqual(t[:10],'2025-12-31')
            for a in [o['actual'],o['weekly'],*o['predictions'].values()]:
                self.assertEqual(len(a),72);self.assertTrue(all(math.isfinite(v) for v in a))
    def test_metrics_are_recomputed_from_shown_predictions(self):
        for m in self.d['models']:
            ys=[y for o in self.d['origins'] for y in o['actual']]
            ps=[p for o in self.d['origins'] for p in o['predictions'][m['id']]]
            errors=[p-y for p,y in zip(ps,ys)];mean=sum(ys)/len(ys)
            computed={'mae':sum(abs(e) for e in errors)/len(errors),'rmse':math.sqrt(sum(e*e for e in errors)/len(errors)),'r2':1-sum(e*e for e in errors)/sum((y-mean)**2 for y in ys)}
            for k,v in computed.items():self.assertAlmostEqual(v,m['metrics'][k],places=6)
    def test_no_claimed_live_forecast_and_offline_is_self_contained(self):
        status=json.loads((ROOT/'docs/demo/live-status.json').read_text());self.assertEqual(status['status'],'not_connected')
        html=(ROOT/'docs/downloads/smp72-offline.html').read_text()
        self.assertIn('id="demo-json"',html);self.assertNotIn('<script defer src=',html);self.assertNotIn('<link rel="stylesheet"',html)
        self.assertIn('과거 검증 재생',html)
if __name__=='__main__':unittest.main()
