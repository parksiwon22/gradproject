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
        status=json.loads((ROOT/'docs/demo/live-status.json').read_text());self.assertIn(status['status'],['not_connected','preliminary','issued'])
        html=(ROOT/'docs/downloads/smp72-offline.html').read_text()
        self.assertIn('id="demo-json"',html);self.assertNotIn('<script defer src=',html);self.assertNotIn('<link rel="stylesheet"',html)
        self.assertIn('과거 검증 재생',html)
    def test_new_forecast_is_genuine_and_explicit_about_time_and_imputation(self):
        f=json.loads((ROOT/'docs/demo/forecast-20261120.json').read_text())
        origin=datetime.fromisoformat(f['forecast_origin']);cutoff=datetime.fromisoformat(f['observed_cutoff'])
        self.assertEqual(f['mode'],'new_forecast');self.assertEqual(f['feature_count'],83)
        self.assertEqual(origin-cutoff,timedelta(hours=72));self.assertEqual(len(f['rows']),72)
        if f['status']=='issued':self.assertGreaterEqual(datetime.fromisoformat(f['calculated_at']),origin)
        else:self.assertEqual(f['status'],'preliminary')
        self.assertLessEqual(datetime.fromisoformat(f['input_snapshot_at']),origin)
        self.assertEqual(f['domestic_lng']['reference_month'],'2026-10')
        for i,r in enumerate(f['rows']):
            self.assertEqual(datetime.fromisoformat(r['target_timestamp']),origin+timedelta(hours=i+1))
            self.assertIsNone(r['actual_smp']);self.assertAlmostEqual(r['predicted_smp'],.4*r['sarimax_prediction']+.6*r['weekly_baseline'],places=7)
            self.assertLessEqual(datetime.fromisoformat(r['weekly_reference_timestamp'])+timedelta(hours=1),cutoff)
        self.assertEqual({q['variable'] for q in f['quality'] if q['missing_hours']==168},{'generation_lng','generation_coal','generation_nuclear','solar_actual','wind_actual'})
        offline=(ROOT/'docs/downloads/smp72-offline.html').read_text();self.assertNotIn('id="live-json"',offline)
    def test_2025_presentation_and_full_semester_coverage(self):
        o=next(o for o in self.d['origins'] if o['forecast_origin']=='2025-11-19T23:00:00+09:00')
        self.assertEqual(o['targets'][0],'2025-11-20T00:00:00+09:00')
        self.assertEqual(o['targets'][-1],'2025-11-22T23:00:00+09:00')
        latest={}
        for o in self.d['origins']:
            for i,t in enumerate(o['targets']):
                if t not in latest or latest[t]['origin']<o['forecast_origin']:latest[t]={'origin':o['forecast_origin'],'horizon':i+1}
        self.assertEqual(len(latest),4416)
        self.assertEqual([sum(t[5:7]==f'{m:02d}' for t in latest) for m in range(7,13)],[744,744,720,744,720,744])
        html=(ROOT/'docs/index.html').read_text();self.assertNotIn('legacy-v20.html',html)
        self.assertIn('2025-11-20T18:00',html)
        self.assertIn('forecastDateSelect',html)
        self.assertIn('forecast-20261120.json',(ROOT/'docs/smp72_live.js').read_text())
        self.assertIn('id="latest-json"',(ROOT/'docs/downloads/smp72-offline.html').read_text())
if __name__=='__main__':unittest.main()
