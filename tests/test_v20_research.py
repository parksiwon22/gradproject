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

    def test_d1_same_hour_missing_at_issue_falls_back_to_nearest_prior_day(self):
        issued = pd.Timestamp('2026-10-20 23:00', tz='Asia/Seoul')
        ts = pd.date_range('2026-10-10', periods=24*11, freq='h', tz='Asia/Seoul')
        # The prior-day 23:00 value is missing; the 48-hour-old value remains usable.
        prices = pd.DataFrame({'timestamp': ts.delete(ts.get_loc(issued)),
                               'smp': np.arange(len(ts)-1, dtype=float)})
        target = pd.date_range('2026-10-21', periods=24, freq='h', tz='Asia/Seoul')
        sources, values, lags = research.available_same_hour_baseline(prices, target)
        self.assertEqual(sources[23], issued-pd.Timedelta(days=1))
        self.assertEqual(lags[23], 2)
        self.assertEqual(values[23], float(prices.set_index('timestamp').loc[sources[23], 'smp']))
        self.assertTrue((sources <= issued).all())

    def test_retry_is_excluded_and_known_target_is_not_scored(self):
        temp_root = Path(__file__).resolve().parents[1] / 'artifacts/v20/test_temp'
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as folder:
            root = Path(folder)
            target = pd.date_range('2026-10-25', periods=24, freq='h', tz='Asia/Seoul')
            records = []
            for minute in (0, 2):
                issue = pd.Timestamp(f'2026-10-24 23:{minute:02d}', tz='Asia/Seoul')
                path = root / f'retry{minute}.json'
                path.write_text(str(minute), encoding='utf-8')
                record = dict(mode='RESEARCH_ONLY_NOT_OFFICIAL_ISSUANCE', target_date='2026-10-25',
                    horizon_days=1, model='same_hour_previous_day_baseline', model_sha256='baseline',
                    reference_origin=issue.isoformat(), issued_at=issue.isoformat(), input_cutoff=issue.isoformat(),
                    target_prices_known_at_issue=True, future_input_rows=0, late_receipt_rows=0)
                forecast = pd.DataFrame({'timestamp': target, 'predicted_smp': np.arange(24, dtype=float)})
                records.append((path, record, forecast))
            actual = pd.DataFrame({'timestamp': target, 'smp': np.arange(24, dtype=float),
                                   'first_seen': target[0].isoformat(), 'snapshot': 'fixture'})
            issue_prices = pd.DataFrame({'timestamp': pd.date_range('2026-10-24', periods=24, freq='h', tz='Asia/Seoul'),
                                         'smp': np.arange(24, dtype=float)})
            issue_prices.attrs['available_future_timestamp_counts'] = {'2026-10-25': 24}
            with patch.object(research, 'RESEARCH', root/'research'), \
                 patch.object(research, 'inventory', return_value=records), \
                 patch.object(research, 'reference', return_value=(actual, [])), \
                 patch.object(research, '_cutoff_prices', return_value=issue_prices):
                result = research.evaluate(backtest=False)
            evaluated = pd.read_csv(root/'research'/'operational_metrics.csv')
            duplicates = pd.read_csv(root/'research'/'duplicate_forecasts.csv')
            self.assertEqual(result['operational_unique_forecasts'], 1)
            self.assertEqual(result['operational_duplicates_excluded'], 1)
            self.assertEqual(len(evaluated), 1)
            self.assertEqual(evaluated.iloc[0].status, 'excluded_target_prices_known_at_issue')
            self.assertTrue(bool(duplicates.iloc[0].identical_prediction))
            self.assertTrue(bool(duplicates.iloc[0].excluded_from_evaluation))


if __name__ == '__main__':
    unittest.main()
