import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from v20.ml_shadow_adapter import predict
from v11.features import FEATURES as D1_FEATURES
from v17.features import FEATURES as DX_FEATURES


class ZeroResidual:
    def predict(self, matrix):
        return np.zeros(len(matrix))


class MLShadowAdapterTests(unittest.TestCase):
    def setUp(self):
        self.prices = pd.DataFrame({'timestamp': pd.date_range('2024-01-01', '2024-09-10 23:00', freq='h', tz='Asia/Seoul')})
        self.prices['smp'] = 100 + 12*np.sin(np.arange(len(self.prices))*2*np.pi/24) + np.arange(len(self.prices))*.0001
        self.prices['first_seen'] = (self.prices.timestamp - pd.Timedelta(minutes=5)).dt.tz_convert('UTC')

    def bundle_path(self, horizon):
        path = Path(tempfile.gettempdir()) / f'ml_shadow_test_d{horizon}.joblib'
        joblib.dump({'horizon_days': horizon, 'model': ZeroResidual(), 'features': D1_FEATURES if horizon == 1 else DX_FEATURES,
                     'mode': 'test', 'anchor_weight': .25, 'trained_through': '2024-08-30'}, path)
        return path

    def test_forecasts_all_horizons_and_filters_future_rows(self):
        targets = {1:'2024-09-01', 2:'2024-09-02', 3:'2024-09-03'}
        for horizon, target in targets.items():
            path = self.bundle_path(horizon)
            original = predict(self.prices, target, horizon, path)
            origin = pd.Timestamp(original.origin_timestamp.iloc[0])
            future = self.prices[self.prices.timestamp > origin].copy()
            future.loc[:, 'smp'] = 99999
            future.loc[:, 'first_seen'] = origin.tz_convert('UTC') + pd.Timedelta(hours=1)
            altered = pd.concat([self.prices[self.prices.timestamp <= origin], future], ignore_index=True)
            replay = predict(altered, target, horizon, path)
            self.assertEqual(len(original), 24)
            np.testing.assert_allclose(original.predicted_smp, replay.predicted_smp)
            self.assertTrue((original.timestamp > origin).all())

    def test_rejects_horizon_mismatch(self):
        with self.assertRaises(ValueError):
            predict(self.prices, '2024-09-01', 4, self.bundle_path(1))

    def test_missing_2300_origin_smp_fails_closed(self):
        for horizon, target in {1:'2024-09-01', 2:'2024-09-02', 3:'2024-09-03'}.items():
            origin = pd.Timestamp(target, tz='Asia/Seoul') - pd.Timedelta(days=horizon) + pd.Timedelta(hours=23)
            missing = self.prices[self.prices.timestamp != origin].copy()
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                predict(missing, target, horizon, self.bundle_path(horizon))


if __name__ == '__main__':
    unittest.main()
