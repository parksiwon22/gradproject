from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
from evaluate_upgrade import metric_values, select_blend_weights  # noqa: E402
from run_upgrade import _verify_baselines  # noqa: E402


class UpgradeEvaluationTests(unittest.TestCase):
    def test_metrics_report_signed_bias_and_direction_share(self) -> None:
        values = metric_values(pd.Series([10.0, 20.0, 30.0]), pd.Series([12.0, 20.0, 28.0]))
        self.assertAlmostEqual(values["bias_pred_minus_actual"], 0.0)
        self.assertAlmostEqual(values["overprediction_share"], 1 / 3)
        self.assertAlmostEqual(values["underprediction_share"], 1 / 3)
        self.assertAlmostEqual(values["mae"], 4 / 3)

    def test_blend_selection_uses_validation_inputs_and_includes_endpoints(self) -> None:
        actual = np.array([100.0, 200.0, 300.0, 400.0])
        frame = pd.DataFrame({
            "split": ["validation"] * 4,
            "target_timestamp": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
            "forecast_day": [1, 1, 2, 2],
            "actual_smp": actual,
            "predicted_e0": actual + 50,
            "baseline_recent_daily": actual,
            "baseline_weekly": actual + 100,
        })
        selected = select_blend_weights(frame)
        self.assertEqual(selected["baseline"], "recent_daily")
        self.assertEqual(selected["alpha"], 0.0)
        self.assertEqual(selected["validation_mae"], 0.0)
        self.assertEqual({row["alpha"] for row in selected["candidate_rows"]}, {round(i / 10, 1) for i in range(11)})
        self.assertTrue(all(row["baseline"] in {"recent_daily", "weekly"} for row in selected["walk_forward_rows"]))

    def test_empty_metric_slice_is_explicit(self) -> None:
        values = metric_values(pd.Series([np.nan]), pd.Series([3.0]))
        self.assertEqual(values["n_rows"], 0)
        self.assertEqual(values["coverage"], 0.0)
        self.assertTrue(np.isnan(values["mae"]))

    def test_baseline_sources_are_before_common_cutoff(self) -> None:
        origin = pd.Timestamp("2025-07-01 23:00:00")
        cutoff = origin - pd.Timedelta(hours=72)
        frame = pd.DataFrame({
            "forecast_origin": [origin], "observed_cutoff": [cutoff],
            "target_timestamp": [pd.Timestamp("2025-07-02 00:00:00")],
            "horizon": [1], "baseline_origin_day": ["2025-06-27"],
            "baseline_recent_daily": [110.0], "baseline_weekly": [120.0],
        })
        _verify_baselines(frame)
        frame["baseline_origin_day"] = "2025-06-29"
        with self.assertRaisesRegex(ValueError, "Recent-daily baseline uses SMP after observed_cutoff"):
            _verify_baselines(frame)


if __name__ == "__main__":
    unittest.main()
