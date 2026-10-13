import json
import unittest
import numpy as np
import pandas as pd
from model import v5
from services import v5_web_adapter as web


class UnitTests(unittest.TestCase):
    def test_split_uses_target_day(self):
        frame = pd.DataFrame({"target_date": ["2023-08-31", "2023-09-01", "2023-09-30", "2023-10-01"]})
        split = v5.split_stacked(frame)
        self.assertEqual([len(split.train), len(split.validation), len(split.test)], [1, 2, 1])

    def test_charging_ties_and_gaps(self):
        frame = pd.DataFrame({"datetime": pd.to_datetime(["2023-10-01 00:00", "2023-10-01 01:00", "2023-10-01 04:00"]), "predicted_smp": [10, 10, 0]})
        block = web.cheapest_consecutive_block(frame, 2)
        self.assertEqual(block.index.tolist(), [0, 1])
        with self.assertRaises(ValueError):
            web.cheapest_consecutive_block(frame, 3)
        with self.assertRaises(ValueError):
            web.cheapest_consecutive_block(frame, 0)

    def test_missing_complete_day(self):
        with self.assertRaises(ValueError):
            web.choose_demo_day(pd.DataFrame({"target_date": ["2023-10-01"], "horizon": [1]}))


class IntegrationTests(unittest.TestCase):
    def test_saved_model_inference_and_cutoff(self):
        raw = pd.read_csv(v5.DATA_PATH)
        history = raw[pd.to_datetime(raw.timestamp) < pd.Timestamp("2023-10-01")]
        v5.DEPLOYMENT_STATE.clear()
        forecast = v5.generate_day_ahead_forecast(history, "2023-10-01")
        reference = pd.read_csv(v5.RESULTS_DIR / "v5_day_ahead_predictions.csv")
        expected = reference[reference.target_date == "2023-10-01"].sort_values("horizon")
        self.assertEqual(forecast.horizon.tolist(), list(range(1, 25)))
        np.testing.assert_allclose(forecast.predicted_smp, expected.profile_hgb_blend, atol=1e-8)
        self.assertTrue(forecast.ramp_probability.between(0, 1).all())

    def test_full_reproduction(self):
        summary = json.loads((v5.RESULTS_DIR / "v5_training_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["mode"], "full")
        self.assertEqual(summary["row_counts"], {"train": 5640, "validation": 720, "test": 2208})
        metrics = pd.read_csv(v5.RESULTS_DIR / "v5_model_comparison.csv")
        row = metrics[(metrics.split == "final_test") & (metrics.model == summary["selected_model"])].iloc[0]
        self.assertAlmostEqual(row.MAE, 11.093786148024114, places=6)
        self.assertAlmostEqual(row.R2, 0.6429194848696833, places=6)

    def test_web_outputs(self):
        result = web.generate_web_outputs()
        self.assertEqual(result["forecast_rows"], 24)
        self.assertEqual(result["recommendation_rows"], 3)
        metrics = json.loads(web.METRICS_OUT.read_text(encoding="utf-8"))
        self.assertAlmostEqual(metrics["final_test_mae"], 11.093786148024114, places=6)

    def test_streamlit_interaction(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file("app.py").run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        app.slider[0].set_value(90).run()
        self.assertEqual(len(app.exception), 0)
        self.assertGreater(len(app.warning), 0)
        app.slider[0].set_value(35).run()
        app.selectbox[0].select(0).run()
        app.selectbox[1].select(0).run()
        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
