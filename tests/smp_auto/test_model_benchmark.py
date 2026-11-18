import unittest
import importlib.util

import numpy as np
import pandas as pd

from research.smp_model_benchmark.benchmark import (
    FEATURES, _build_sklearn_model, make_day_map, make_recurrent_model,
    validate_no_future_leak,
)


class SMPOnlyBenchmarkTests(unittest.TestCase):
    def test_features_have_exact_d3_origin_and_future_mutation_invariance(self):
        timestamps=pd.date_range("2024-01-01", "2024-03-04 23:00", freq="h", tz="Asia/Seoul")
        prices=pd.DataFrame({"timestamp":timestamps,"smp":100+np.sin(np.arange(len(timestamps))/24)*8})
        target="2024-03-01"
        item=make_day_map(prices,[target])[target]
        self.assertEqual(item["origin"],pd.Timestamp("2024-02-27 23:00",tz="Asia/Seoul"))
        self.assertEqual(item["X"].shape,(24,len(FEATURES)))
        self.assertEqual(item["actual"].shape,(24,))
        self.assertEqual(set(item["sequences"]),{168,672})
        validate_no_future_leak(prices,item)

    def test_all_tabular_estimators_construct_and_fit_common_residual_matrix(self):
        rng=np.random.default_rng(42)
        X=rng.normal(size=(96,len(FEATURES)))
        y=0.5*X[:,0]-0.2*X[:,1]
        for name in ("HGB","LightGBM","ExtraTrees","LinearRegression","RandomForest","XGBoost"):
            with self.subTest(model=name):
                optional = {"LightGBM":"lightgbm", "XGBoost":"xgboost"}.get(name)
                if optional and importlib.util.find_spec(optional) is None:
                    continue
                model=_build_sklearn_model(name)
                model.fit(X,y)
                pred=model.predict(X[:24])
                self.assertEqual(pred.shape,(24,))
                self.assertTrue(np.isfinite(pred).all())

    def test_lstm_and_gru_share_input_output_shapes_and_fixed_hidden_width(self):
        if importlib.util.find_spec("tensorflow") is None:
            self.skipTest("TensorFlow is an isolated benchmark dependency")
        for kind in ("LSTM","GRU"):
            with self.subTest(model=kind):
                model=make_recurrent_model(kind,168,17)
                pred=model.predict([np.zeros((2,168,1),np.float32),np.zeros((2,24,6),np.float32)],verbose=0)
                self.assertEqual(pred.shape,(2,24))
                self.assertGreater(model.count_params(),0)


if __name__=="__main__":
    unittest.main()
