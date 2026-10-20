import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from v20 import research


class V20ResearchTests(unittest.TestCase):
    def test_baseline_uses_only_same_hour_at_or_before_each_horizon_origin(self):
        ts = pd.date_range('2026-10-17', periods=24*6, freq='h', tz='Asia/Seoul')
        prices = pd.DataFrame({'timestamp': ts, 'smp': np.arange(len(ts), dtype=float)})
        for horizon in (1, 2, 3):
            target = pd.date_range('2026-10-20', periods=24, freq='h', tz='Asia/Seoul')
            source, predicted = research.baseline_prediction(prices, target, horizon)
            origin = target[0].normalize()-pd.Timedelta(days=horizon)+pd.Timedelta(hours=23)
            self.assertTrue((source <= origin).all())
            np.testing.assert_allclose(predicted, prices.set_index('timestamp').smp.reindex(source))

    def test_gate_independent_generation_saves_all_three_horizons_at_input_cutoff(self):
        issued = pd.Timestamp('2026-10-20 23:00:05', tz='Asia/Seoul')
        ts = pd.date_range(pd.Timestamp('2026-10-10', tz='Asia/Seoul'), issued, freq='h')
        ts = ts[ts <= issued.floor('h')]
        prices = pd.DataFrame({'timestamp': ts, 'smp': np.arange(len(ts), dtype=float),
                               'first_seen': issued.tz_convert('UTC').isoformat(), 'snapshot': 'fixture-hash'})
        temp_root = Path(__file__).resolve().parents[1] / 'artifacts/v20/test_temp'
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as folder:
            root = Path(folder)
            model = root / 'frozen.joblib'
            model.write_bytes(b'frozen model fixture')

            def cutoff(_):
                return prices[prices.timestamp <= issued].copy()

            def predict(_name, _prices, _day, _horizon):
                return None, np.arange(24, dtype=float)+150, None, None

            with patch.object(research, 'RESEARCH', root/'research'), \
                 patch.object(research, 'MODEL_FILE', model), \
                 patch.object(research, '_verified_model_hash', return_value='fixture-model-hash'), \
                 patch.object(research, '_cutoff_prices', side_effect=cutoff), \
                 patch.object(research, 'predict_model', side_effect=predict):
                result = research.generate(clock=issued)

            self.assertEqual(result['forecast_files'], 3)
            records = sorted((root/'research'/'forecasts').glob('*.json'))
            self.assertEqual(len(records), 3)
            horizons = set()
            for path in records:
                row = json.loads(path.read_text(encoding='utf-8'))
                horizons.add(row['horizon_days'])
                self.assertEqual(row['mode'], 'RESEARCH_ONLY_NOT_OFFICIAL_ISSUANCE')
                self.assertFalse(row['official_issuance'])
                self.assertEqual(row['input_cutoff'], row['issued_at'])
                self.assertEqual(row['future_input_rows'], 0)
                self.assertEqual(row['late_receipt_rows'], 0)
                self.assertEqual(len(row['rows']), 24)
                self.assertTrue(all(pd.Timestamp(x['timestamp']) > issued for x in row['rows']))
                if row['horizon_days'] == 1:
                    self.assertTrue(all(pd.Timestamp(x['baseline_source_timestamp']) <= issued for x in row['features']))
            self.assertEqual(horizons, {1, 2, 3})


if __name__ == '__main__':
    unittest.main()
