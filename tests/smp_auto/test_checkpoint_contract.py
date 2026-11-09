import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(os.environ["SMP_SOURCE"]).resolve()
sys.path.insert(0,str(SOURCE))
from v11.features import build as build_d1
from v17.features import build as build_dx

class CheckpointContractTests(unittest.TestCase):
    def test_selected_family_checkpoints_make_24_point_vectors(self):
        model_dir=ROOT/"automation"/"models"
        manifest=json.loads((model_dir/"manifest.json").read_text(encoding="utf-8"))
        origin=pd.Timestamp("2026-11-09 23:00",tz="Asia/Seoul")
        idx=pd.date_range(pd.Timestamp("2023-01-01",tz="Asia/Seoul"),origin,freq="h")
        smp=100+8*np.sin(np.arange(len(idx))*2*np.pi/24/7)+4*np.sin(np.arange(len(idx))*2*np.pi/24)
        prices=pd.DataFrame({"timestamp":idx,"smp":smp})
        for h in (1,2,3):
            meta=manifest["models"][str(h)];path=model_dir/meta["file"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),meta["sha256"])
            bundle=joblib.load(path)
            target=origin.normalize()+pd.Timedelta(days=h)
            targets=pd.date_range(target,target+pd.Timedelta(hours=23),freq="h")
            frame=(build_d1(prices,targets=targets,days_ahead=1,include_labels=False) if h==1 else
                   build_dx(prices,targets=targets,days_ahead=h,include_labels=False))
            self.assertEqual(len(frame),24)
            residual=bundle["model"].predict(frame[bundle["features"]].to_numpy(float))
            prediction=frame.daily.to_numpy()+residual
            self.assertEqual(prediction.shape,(24,))
            self.assertTrue(np.isfinite(prediction).all())

if __name__=="__main__": unittest.main()
