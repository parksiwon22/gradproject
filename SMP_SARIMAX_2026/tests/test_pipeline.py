import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.rows import make_forecast_row
from src.preprocessing.build_dataset import extract_smp
from src.preprocessing.monthly import merge_monthly_with_conservative_lag


class PipelineDataTests(unittest.TestCase):
    def test_72_hour_output_is_aligned_and_horizons_are_grouped(self):
        origin = pd.Timestamp("2025-01-01 23:00")
        idx = pd.date_range("2024-12-01", "2025-01-05", freq="h")
        y = pd.Series(np.arange(len(idx), dtype=float), index=idx)
        target = pd.date_range(origin + pd.Timedelta(hours=1), periods=72, freq="h")
        predictions = pd.Series(np.ones(72), index=target)
        rows = make_forecast_row(origin, y, predictions, "validation")
        self.assertEqual(len(rows), 72)
        self.assertTrue(rows.target_timestamp.equals(pd.Series(target)))
        self.assertEqual(rows.forecast_day.value_counts().to_dict(), {"D+1": 24, "D+2": 24, "D+3": 24})

    def test_baseline_sources_are_never_after_origin(self):
        origin = pd.Timestamp("2025-01-01 23:00")
        idx = pd.date_range("2024-12-01", "2025-01-05", freq="h")
        y = pd.Series(np.arange(len(idx), dtype=float), index=idx)
        target = pd.date_range(origin + pd.Timedelta(hours=1), periods=72, freq="h")
        rows = make_forecast_row(origin, y, pd.Series(0.0, index=target), "validation")
        for h, t in zip(rows.horizon, rows.target_timestamp):
            source = t - pd.Timedelta(hours=24 * int(np.ceil(h / 24)))
            self.assertLessEqual(source, origin)
            self.assertLessEqual(t - pd.Timedelta(hours=168), origin)

    def test_kpx_end_hour_maps_to_interval_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "epsis.csv"
            pd.DataFrame({"date": ["20250101", "20250101"], "hour": [1, 24], "areaName": ["육지", "육지"], "smp": [100, 200]}).to_csv(p, index=False)
            got = extract_smp(p)
        self.assertEqual(got.timestamp.tolist(), [pd.Timestamp("2025-01-01 00:00"), pd.Timestamp("2025-01-01 23:00")])

    def test_monthly_join_applies_conservative_publication_lag(self):
        hourly = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", "2025-04-01", freq="MS")})
        monthly = pd.DataFrame({"month": pd.date_range("2025-01-01", periods=3, freq="MS"), "fuel_cost": [10, 20, 30]})
        got = merge_monthly_with_conservative_lag(hourly, monthly, "timestamp", ["fuel_cost"], lag_months=1, monthly_date_col="month")
        values = got.set_index("timestamp").fuel_cost
        self.assertTrue(pd.isna(values.loc["2025-02-01"]))
        self.assertEqual(values.loc["2025-03-01"], 10)
        self.assertEqual(values.loc["2025-04-01"], 20)


if __name__ == "__main__":
    unittest.main()
