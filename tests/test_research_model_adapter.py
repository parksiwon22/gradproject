import unittest

import numpy as np
import pandas as pd

from v20.research_model_adapter import predict


class ResearchModelAdapterTests(unittest.TestCase):
    def setUp(self):
        timestamps = pd.date_range('2025-01-01', '2025-06-05 23:00', freq='h', tz='Asia/Seoul')
        hours = np.arange(len(timestamps))
        self.prices = pd.DataFrame({
            'timestamp': timestamps,
            'smp': 100 + 8 * np.sin(hours * 2 * np.pi / 24) + 2 * np.cos(hours * 2 * np.pi / (24 * 7)),
            'first_seen': (timestamps - pd.Timedelta(hours=2)).tz_convert('UTC'),
        })
        self.config = {
            'algorithm': 'test-shadow',
            'calibration': {
                'window_days': 90,
                'exclude_issue_date': True,
                'anchors': {'median2': [7, 14], 'median4': [7, 14, 21, 28],
                            'median8': list(range(7, 57, 7)), 'mean4': [7, 14, 21, 28]},
                'weights': [0, .25, .5, .75, 1],
            },
        }

    def test_24_hour_forecast_is_stable_and_asof_filtered(self):
        a = predict(self.prices, '2025-06-05', 1, self.config)
        mutated = self.prices.copy()
        target_mask = mutated.timestamp.dt.strftime('%Y-%m-%d').eq('2025-06-05')
        mutated.loc[target_mask, 'smp'] = 999.0
        future = pd.DataFrame({
            'timestamp': pd.date_range('2025-06-06', periods=24, freq='h', tz='Asia/Seoul'),
            'smp': np.full(24, 999.0),
            'first_seen': pd.date_range('2025-06-07', periods=24, freq='h', tz='Asia/Seoul').tz_convert('UTC'),
        })
        b = predict(pd.concat([mutated, future], ignore_index=True), '2025-06-05', 1, self.config)
        np.testing.assert_allclose(a.predicted_smp.to_numpy(), b.predicted_smp.to_numpy())
        self.assertEqual(len(a), 24)
        self.assertEqual(a.timestamp.iloc[0], pd.Timestamp('2025-06-05', tz='Asia/Seoul'))
        self.assertTrue((a.origin_timestamp.iloc[0] == pd.Timestamp('2025-06-04 23:00', tz='Asia/Seoul')))

    def test_invalid_horizon_rejected(self):
        with self.assertRaises(ValueError):
            predict(self.prices, '2025-06-05', 4, self.config)


if __name__ == '__main__':
    unittest.main()
