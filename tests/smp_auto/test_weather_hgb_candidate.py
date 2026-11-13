import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from research.smp_weather.weather_features import (
    REGIONS, WEATHER_MODEL_FEATURES, WEATHER_VARIABLES, target_feature_frame,
    target_weather_features,
)
from research.smp_weather.train_weather_hgb import fit_weather_hgb
from research.smp_weather.walkforward_weather import run


class WeatherHGBCandidateTests(unittest.TestCase):
    day = "2025-03-01"
    origin = pd.Timestamp("2025-02-26T23:00:00+09:00")

    def _records(self):
        target = pd.date_range(pd.Timestamp(self.day, tz="Asia/Seoul"), periods=24, freq="h")
        rows = []
        for region_i, region in enumerate(REGIONS):
            for variable_i, variable in enumerate(WEATHER_VARIABLES):
                for hour_i, hour in enumerate(target):
                    rows.append({"record_type": "forecast", "source": "KMA API Hub test fixture",
                        "location": region, "category": variable,
                        "issued_at_kst": "2025-02-26T20:00:00+09:00",
                        "received_at_utc": "2025-02-26T11:05:00+00:00",
                        "forecast_origin_kst": self.origin.isoformat(),
                        "target_at_kst": hour.isoformat(), "target_date": self.day,
                        "value": str(10 + region_i + variable_i + hour_i / 10)})
        return rows

    def test_features_require_complete_forecast_and_ignore_observations(self):
        rows = self._records()
        rows.append({"record_type": "observation", "source": "KMA API Hub ASOS",
            "location": "Seoul", "category": "TMP", "observed_at_kst": self.day + "T00:00:00+09:00",
            "received_at_utc": "2025-03-03T00:00:00+00:00", "value": "999"})
        target = pd.date_range(pd.Timestamp(self.day, tz="Asia/Seoul"), periods=24, freq="h")
        result = target_weather_features(rows, target, self.origin.isoformat())
        self.assertEqual(result.shape, (24, 1 + len(REGIONS) * len(WEATHER_VARIABLES)))
        self.assertNotIn(999, result.to_numpy())
        with self.assertRaisesRegex(ValueError, "incomplete"):
            target_weather_features(rows[:-2], target, self.origin.isoformat())

    def test_forecast_issue_and_first_receipt_after_origin_are_excluded(self):
        rows = self._records()
        late = dict(rows[0])
        late.update(issued_at_kst="2025-02-26T23:01:00+09:00", received_at_utc="2025-02-26T14:02:00+00:00", value="999")
        rows.append(late)
        late_receive = dict(rows[1])
        late_receive.update(issued_at_kst="2025-02-26T22:30:00+09:00", received_at_utc="2025-02-26T14:01:00+00:00", value="888")
        rows.append(late_receive)
        target = pd.date_range(pd.Timestamp(self.day, tz="Asia/Seoul"), periods=24, freq="h")
        result = target_weather_features(rows, target, self.origin.isoformat())
        self.assertAlmostEqual(result.iloc[0]["weather_Gangneung_TMP"], 10.0)
        self.assertAlmostEqual(result.iloc[0]["weather_Gangneung_REH"], 11.0)

    def test_smp_weather_predictors_are_truncated_at_each_target_origin(self):
        idx = pd.date_range("2025-01-01", "2025-03-02 23:00", freq="h", tz="Asia/Seoul")
        values = 100 + np.sin(np.arange(len(idx)) / 24) * 5
        prices = pd.DataFrame({"timestamp": idx, "smp": values})
        rows = self._records()
        first, origin = target_feature_frame(prices, rows, self.day)
        changed = prices.copy()
        changed.loc[changed.timestamp > origin, "smp"] += 100000
        second, _ = target_feature_frame(changed, rows, self.day)
        np.testing.assert_allclose(first[list(WEATHER_MODEL_FEATURES)], second[list(WEATHER_MODEL_FEATURES)])
        self.assertTrue((first.origin_timestamp <= origin).all())

    def test_candidate_fit_refuses_small_training_sample(self):
        X = pd.DataFrame([[0.0] * len(WEATHER_MODEL_FEATURES)], columns=WEATHER_MODEL_FEATURES)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "at least 365"):
                fit_weather_hgb(X, pd.Series([0.0]), target_dates=["2025-01-01"],
                    trained_through="2025-01-01", output_dir=Path(folder))
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_current_artifact_is_read_only_and_weather_score_is_withheld(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run(weather_path=Path(folder) / "no-vintages.jsonl",
                out=Path(folder) / "weather-result.json")
        self.assertEqual(result["status"], "blocked_no_forecast_vintages")
        self.assertEqual(result["paired_comparison"]["smp_only_days"], 316)
        self.assertEqual(result["paired_comparison"]["weather_hgb_days"], 0)
        self.assertIsNone(result["paired_comparison"]["metrics"])


if __name__ == "__main__":
    unittest.main()
