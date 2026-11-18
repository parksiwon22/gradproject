import unittest
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from automation.smp_auto.walkforward_d3_2026 import (
    KST, _fit_checkpoint, _forecast_fold, _metrics, validate_payload,
)


class WalkForwardD3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        timestamps = pd.date_range("2024-01-01", "2025-06-20 23:00", freq="h", tz=KST)
        hour = timestamps.hour.to_numpy()
        day = timestamps.dayofyear.to_numpy()
        # Deterministic SMP-shaped series with daily and slow seasonal variation.
        values = 100 + 14 * np.sin(hour * np.pi / 12) + 5 * np.cos(day * 2 * np.pi / 365)
        cls.prices = pd.DataFrame({"timestamp": timestamps, "smp": values})
        cls.origin = pd.Timestamp("2025-06-09 23:00", tz=KST)
        cls.target = "2025-06-12"
        cls.checkpoint = _fit_checkpoint(cls.prices[cls.prices.timestamp <= cls.origin],
                                         pd.Timestamp("2025-06-09 00:00", tz=KST))

    def test_training_cutoff_precedes_issue_origin(self):
        self.assertLess(self.checkpoint["train_end"], "2025-06-09")
        self.assertLessEqual(pd.Timestamp(self.checkpoint["model_trained_at_kst"]), self.origin)

    def test_future_smp_mutation_cannot_change_prediction(self):
        original = _forecast_fold(self.prices, self.target, self.checkpoint)["prediction"]
        changed = self.prices.copy()
        changed.loc[changed.timestamp > self.origin, "smp"] += 100_000
        mutated = _forecast_fold(changed, self.target, self.checkpoint)["prediction"]
        np.testing.assert_allclose(original, mutated, rtol=0, atol=0)

    def test_metrics_recalculate_and_hourly_vectors_are_24(self):
        pred = np.linspace(90, 120, 24)
        actual = pred + np.sin(np.arange(24))
        baseline = np.full(24, 105.0)
        metrics = _metrics(pred, actual, baseline)
        payload = {"records": [{
            "target_date": self.target, "forecast_origin_kst": self.origin.isoformat(),
            "hours_kst": list(range(24)),
            "model_trained_at_kst": self.checkpoint["model_trained_at_kst"],
            "trained_through": self.checkpoint["train_end"],
            "input_max_timestamp_kst": self.origin.isoformat(),
            "predictions": pred.tolist(), "actual_smp": actual.tolist(),
            "baseline_predictions": baseline.tolist(), "metrics": metrics,
        }]}
        validate_payload(payload)
        payload["records"][0]["metrics"]["mae"] += 1
        with self.assertRaisesRegex(AssertionError, "MAE mismatch"):
            validate_payload(payload)

    def test_g2v_regret_uses_frozen_prediction_plan_at_actual_prices(self):
        actual = np.array([40 + 2 * h for h in range(24)], dtype=float)
        pred = actual[::-1].copy()  # forecast orders the expensive hours first
        metrics = _metrics(pred, actual, actual, include_v2g=False)
        self.assertGreater(metrics["g2v_regret_won"], 0)
        self.assertAlmostEqual(
            metrics["g2v_regret_won"],
            metrics["g2v_predicted_plan_actual_cost_won"] - metrics["g2v_oracle_actual_cost_won"],
            places=9,
        )
        self.assertAlmostEqual(metrics["baseline_g2v_regret_won"], 0, places=9)
        # 30 kWh of battery energy at 90% efficiency is not an integer
        # number of 7 kW charging slots, so the last selected interval is partial.
        selected = np.asarray(metrics["predicted_charge_kw"])
        self.assertTrue(np.any((selected > 1e-8) & (selected < 7 - 1e-8)))

    def test_matching_prediction_and_actual_has_zero_regret(self):
        actual = np.array([80 + 3 * np.sin(h) for h in range(24)])
        metrics = _metrics(actual, actual, actual, include_v2g=False)
        self.assertAlmostEqual(metrics["g2v_regret_won"], 0, places=8)
        self.assertAlmostEqual(metrics["baseline_g2v_regret_won"], 0, places=8)

    def test_duplicate_evaluation_day_is_rejected(self):
        pred = np.full(24, 100.0)
        actual = np.full(24, 101.0)
        baseline = np.full(24, 102.0)
        row = {"target_date": self.target, "forecast_origin_kst": self.origin.isoformat(),
               "hours_kst": list(range(24)),
               "model_trained_at_kst": self.checkpoint["model_trained_at_kst"],
               "trained_through": self.checkpoint["train_end"],
               "input_max_timestamp_kst": self.origin.isoformat(),
               "predictions": pred.tolist(), "actual_smp": actual.tolist(),
               "baseline_predictions": baseline.tolist(),
               "metrics": _metrics(pred, actual, baseline)}
        with self.assertRaisesRegex(AssertionError, "duplicate target day"):
            validate_payload({"records": [row, dict(row)]})

    def test_saved_2026_replay_artifact_and_summary_recalculate(self):
        path = Path(__file__).resolve().parents[2] / "docs" / "live" / "walkforward_d3_2026.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        validate_payload(payload)
        rows = payload["records"]
        self.assertEqual(len(rows), 316)
        self.assertEqual(payload["excluded_target_count"], 0)
        self.assertEqual((rows[0]["target_date"], rows[-1]["target_date"]),
                         ("2026-01-01", "2026-11-12"))
        errors = np.concatenate([np.asarray(r["metrics"]["hourly_signed_error"]) for r in rows])
        baseline_errors = np.concatenate([
            np.asarray(r["baseline_predictions"]) - np.asarray(r["actual_smp"]) for r in rows
        ])
        summary = payload["summary"]
        self.assertIn("realized_cost(forecast_decision_schedule", payload["regret_formula"])
        self.assertEqual(summary["v2g_vs_g2v_days_better"], 93)
        self.assertEqual(summary["v2g_vs_g2v_days_worse"], 50)
        self.assertEqual(summary["v2g_vs_g2v_days_equal_0_01_won"], 173)
        self.assertAlmostEqual(float(np.mean(np.abs(errors))), summary["mae"], places=8)
        self.assertAlmostEqual(float(np.sqrt(np.mean(errors ** 2))), summary["rmse"], places=8)
        self.assertAlmostEqual(float(np.mean(np.abs(baseline_errors))), summary["baseline_mae"], places=8)
        self.assertAlmostEqual(float(np.sqrt(np.mean(baseline_errors ** 2))), summary["baseline_rmse"], places=8)
        self.assertAlmostEqual(np.mean([r["metrics"]["g2v_regret_won"] for r in rows]),
                               summary["mean_daily_g2v_regret_won"], places=8)


if __name__ == "__main__":
    unittest.main()
