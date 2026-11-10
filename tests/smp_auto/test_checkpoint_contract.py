import hashlib
import json
import os
import sys
import datetime
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(os.environ["SMP_SOURCE"]).resolve()
sys.path.insert(0,str(SOURCE))
from v11.features import build as build_d1
from v17.features import build as build_dx
from v11.store import connect,ingest
from automation.smp_auto.publish_forecast import generate,seasonal_anchor,load_history,_enrich_history,_g2v_regret,_reference_metrics
from automation.smp_auto.probe_target_publication import parse_public_status

class CheckpointContractTests(unittest.TestCase):
    def test_anchor_is_median_of_the_four_prior_same_weekday_days(self):
        timestamps=pd.date_range(pd.Timestamp("2026-10-01",tz="Asia/Seoul"),periods=24*40,freq="h")
        prices=pd.DataFrame({"timestamp":timestamps,"smp":np.repeat(np.arange(40,dtype=float),24)})
        target=pd.date_range(pd.Timestamp("2026-11-10",tz="Asia/Seoul"),periods=24,freq="h")
        anchor=seasonal_anchor(prices,target)
        # 7/14/21/28-day lags are 33, 26, 19, and 12 in this fixture.
        np.testing.assert_allclose(anchor,np.full(24,22.5))

    def test_anchor_rejects_missing_required_lag_instead_of_emitting_nan(self):
        timestamps=pd.date_range(pd.Timestamp("2026-10-01",tz="Asia/Seoul"),periods=24*40,freq="h")
        prices=pd.DataFrame({"timestamp":timestamps,"smp":np.repeat(np.arange(40,dtype=float),24)})
        target=pd.date_range(pd.Timestamp("2026-11-10",tz="Asia/Seoul"),periods=24,freq="h")
        prices=prices[prices.timestamp!=target[0]-pd.Timedelta(days=21)]
        with self.assertRaisesRegex(ValueError,"7·14·21·28일"):
            seasonal_anchor(prices,target)

    def test_publication_probe_only_calls_explicit_zero_placeholder_unpublished(self):
        day="2026-11-11"
        def page(values):
            rows="".join(f"<tr><td>{h}h</td><td>{value}</td></tr>" for h,value in enumerate(values,1))
            return f'<input value="{day}">{rows}'
        frame,status=parse_public_status(page(["0"]*24),day)
        self.assertIsNone(frame)
        self.assertEqual(status,"not_observed_on_this_page")
        with self.assertRaisesRegex(ValueError,"unverified"):
            parse_public_status(page(["0"]*23),day)
        frame,status=parse_public_status(page(["100.5"]*24),day)
        self.assertEqual(status,"official_available")
        self.assertEqual(len(frame),24)

    def test_empty_actions_state_recovers_forecast_history_from_published_latest(self):
        payload=json.dumps({"forecast_history":[{"origin_kst":"2026-11-09T23:00:00+09:00"}]}).encode()
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{"SMP_PUBLISHED_HISTORY_URL":"https://example.invalid/latest.json"}):
            with patch("automation.smp_auto.publish_forecast.urllib.request.urlopen",return_value=__import__("io").BytesIO(payload)):
                history,source,error=load_history(Path(d)/"missing.json")
        self.assertEqual(source,"published_pages")
        self.assertIsNone(error)
        self.assertEqual(len(history),1)

    def test_target_date_enrichment_preserves_origin_prediction_and_scores_references(self):
        origin=pd.Timestamp("2026-11-10T23:00:00+09:00")
        target="2026-11-12"
        times=pd.date_range(pd.Timestamp(target,tz="Asia/Seoul"),periods=24,freq="h")
        received=pd.Timestamp("2026-11-13T00:10:00+09:00").tz_convert("UTC")
        prices=np.linspace(90,140,24)
        frame=pd.DataFrame({"timestamp":times,"smp":prices,"first_seen":received.isoformat(),"last_seen":received.isoformat()})
        forecast={"origin_kst":origin.isoformat(),"issued_at_kst":origin.isoformat(),"issued_at_utc":origin.tz_convert("UTC").isoformat(),
          "status":"issued_research_only","horizons":[{"horizon_days":2,"target_date":target,"predictions":[100.]*24,
          "model_family":"test-hgb-anchor","model_sha256":"abc123","prospective_forecast_eligible":True}]}
        before=list(forecast["horizons"][0]["predictions"])
        probe={target:{"status":"official_available","checked_at_kst":"2026-11-11T23:00:10+09:00"}}
        _enrich_history([forecast],frame,frame,probe,pd.Timestamp("2026-11-13T01:00:00+09:00"))
        item=forecast["horizons"][0]
        self.assertEqual(item["predictions"],before)
        self.assertEqual(item["forecast_origin_date"],"2026-11-10")
        self.assertEqual(item["original_horizon_days"],2)
        self.assertEqual(item["official_publication"]["source"],"KPX inland SMP public page")
        self.assertEqual(item["post_target_observation"]["source"],"EPSIS land SMP collection")
        self.assertEqual(item["post_target_observation"]["price_nature"].split(";")[0],"EPSIS SMP observed again after target end")
        self.assertTrue(item["official_metrics"]["prospective_evaluation_eligible"])
        self.assertIn("g2v_regret",item["post_target_metrics"])
        self.assertEqual(item["post_target_metrics"]["g2v_regret"]["conditions"]["battery_kwh"],60.)

    def test_unverified_publication_status_is_not_marked_evaluation_eligible(self):
        m=_reference_metrics([100.]*24,[110.]*24,False,"publication status unverified")
        self.assertFalse(m["prospective_evaluation_eligible"])
        self.assertEqual(m["exclusion_reason"],"publication status unverified")

    def test_g2v_regret_uses_identical_vehicle_and_connection_conditions(self):
        regret=_g2v_regret([100.]*24,list(range(100,124)))
        self.assertEqual(regret["conditions"],{"battery_kwh":60.,"soc_start_pct":30.,"soc_goal_pct":80.,
            "charge_kw":7.,"efficiency":.9,"start_hour":0,"end_hour":24})
        self.assertGreaterEqual(regret["regret_won"],0.)

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

    def test_full_issue_replay_emits_three_24_hour_vectors_and_excludes_known_d1(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d); home=source/"artifacts"/"v11"; db=connect(home)
            times=pd.date_range(pd.Timestamp("2026-09-01",tz="Asia/Seoul"),pd.Timestamp("2026-11-10 23:00",tz="Asia/Seoul"),freq="h")
            price=110+8*np.sin(np.arange(len(times))*2*np.pi/24)+3*np.cos(np.arange(len(times))*2*np.pi/(24*7))
            frame=pd.DataFrame({"timestamp":times,"smp":price})
            ingest(db,"land",frame,b"synthetic replay fixture","test",home=home,retrieved="2026-11-09T13:00:00+00:00")
            db.close()
            probe={"origin_kst":"2026-11-09T23:00:00+09:00","results":[
              {"horizon_days":1,"target_date":"2026-11-10","checked_at_kst":"2026-11-09T22:45:00+09:00","status":"official_available","before_origin":True},
              {"horizon_days":2,"target_date":"2026-11-11","checked_at_kst":"2026-11-09T22:45:00+09:00","status":"not_observed_on_this_page","before_origin":True},
              {"horizon_days":3,"target_date":"2026-11-12","checked_at_kst":"2026-11-09T22:45:00+09:00","status":"not_observed_on_this_page","before_origin":True}]}
            (home/"public_availability_live.json").write_text(json.dumps(probe),encoding="utf-8")
            result=generate(source=source,model_dir=ROOT/"automation"/"models",out_path=source/"latest.json",
                state_path=source/"history.json",now=pd.Timestamp("2026-11-09T23:01:00+09:00"),origin_date="2026-11-09")
            self.assertEqual(result["run"]["schedule_delay_seconds"],60.0)
            horizons=result["last_successful_forecast"]["horizons"]
            self.assertEqual([len(x["predictions"]) for x in horizons],[24,24,24])
            self.assertFalse(horizons[0]["prospective_forecast_eligible"])
            self.assertTrue(horizons[1]["prospective_forecast_eligible"])
            self.assertTrue(horizons[2]["prospective_forecast_eligible"])
            original=list(horizons[1]["predictions"])
            retry=generate(source=source,model_dir=ROOT/"automation"/"models",out_path=source/"retry.json",
                state_path=source/"history.json",now=pd.Timestamp("2026-11-09T23:02:00+09:00"),origin_date="2026-11-09")
            self.assertEqual(retry["run"]["status"],"already_issued_for_origin")
            self.assertEqual(len(retry["forecast_history"]),1)
            self.assertEqual(retry["last_successful_forecast"]["horizons"][1]["predictions"],original)

if __name__=="__main__": unittest.main()
