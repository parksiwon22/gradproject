import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import tempfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_v5 import smp_predictor_v5 as model
from services import v5_web_adapter as adapter

class V5Contracts(unittest.TestCase):
    def history(self):
        ts = pd.date_range('2023-01-01', periods=12*24, freq='h')
        return pd.DataFrame({'timestamp': ts, 'smp': 100 + np.sin(np.arange(len(ts))),
                             'national_demand_mw': 50000 + np.arange(len(ts))})

    def rows(self, raw):
        prepared = model.preprocess_hourly(raw)
        prepared.df = prepared.df.set_index('timestamp', drop=False)
        rows = pd.DataFrame(model.build_one_day_rows(prepared, pd.Timestamp('2023-01-10'), False))
        return prepared, rows

    def test_reference_is_exact_original(self):
        entries = json.loads((ROOT/'benchmarks/v5/source_manifest.json').read_text())
        expected = next(e['sha256'] for e in entries if e['zip_path']=='pkg/01_model_v5/smp_predictor_v5.py')
        self.assertEqual(hashlib.sha256((ROOT/'model_v5/reference/smp_predictor_v5.py').read_bytes()).hexdigest(), expected)

    def test_future_actuals_do_not_change_features(self):
        raw = self.history()
        prepared, before = self.rows(raw)
        raw.loc[raw.timestamp >= pd.Timestamp('2023-01-10'), ['smp', 'national_demand_mw']] = 999999
        _, after = self.rows(raw)
        pd.testing.assert_frame_equal(before[prepared.model_features], after[prepared.model_features])
        self.assertEqual(len(before), 24)
        self.assertEqual(list(before.horizon), list(range(1, 25)))
        self.assertTrue(before.origin_timestamp.eq(pd.Timestamp('2023-01-09 23:00')).all())

    def test_missing_cutoff_rejected(self):
        raw = self.history()
        raw = raw[raw.timestamp != pd.Timestamp('2023-01-09 23:00')]
        _, rows = self.rows(raw)
        self.assertTrue(rows.empty)

    def test_charging_gap_not_bridged(self):
        df = pd.DataFrame({'datetime': pd.to_datetime(['2023-01-01 00:00','2023-01-01 02:00','2023-01-01 03:00']), 'predicted_smp': [0, 10, 11]})
        block = adapter.cheapest_consecutive_block(df, 2)
        self.assertEqual(block.datetime.iloc[0].hour, 2)
        for duration in [0, -1, 25]:
            with self.assertRaises(ValueError): adapter.cheapest_consecutive_block(df, duration)

    def test_tie_chooses_earliest(self):
        df = pd.DataFrame({'datetime': pd.date_range('2023-01-01', periods=24, freq='h'), 'predicted_smp': 100.0})
        self.assertEqual(adapter.cheapest_consecutive_block(df, 3).datetime.iloc[0].hour, 0)

    def test_regret_uses_actual_cost(self):
        actual = np.array([10, 1, 1] + [10]*21)
        start, cost = model.cheapest_window(actual, 2)
        self.assertEqual(start, 1)
        self.assertEqual(cost, 2)

    def test_adapter_rejects_wrong_date_and_duplicate_horizon(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'pred.csv'
            ts = pd.date_range('2023-01-01', periods=24, freq='h')
            df = pd.DataFrame({'timestamp': ts, 'target_date': '2023-01-01', 'horizon': list(range(1,25)),
                'profile_hgb_blend': 100., 'daily_profile_baseline': 110., 'weekly_profile_baseline': 120.,
                'ramp_probability': .1, 'risk_level': '낮음', 'price_weather': '충전 맑음'})
            df.to_csv(path,index=False)
            with patch.object(adapter, 'V5_PREDICTIONS', path):
                self.assertEqual(len(adapter.build_forecast_for_web('2023-01-01')),24)
                with self.assertRaises(ValueError): adapter.build_forecast_for_web('2023-01-02')
                df.loc[23,'horizon']=23
                df.to_csv(path,index=False)
                with self.assertRaises(ValueError): adapter.build_forecast_for_web('2023-01-01')

if __name__ == '__main__': unittest.main()
