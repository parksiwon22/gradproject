import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SOURCE=Path(os.environ["SMP_SOURCE"]).resolve()
sys.path.insert(0,str(SOURCE))
from v11.store import connect,ingest,latest

class AsOfVintageTests(unittest.TestCase):
    def test_late_received_smp_is_not_in_issue_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            home=Path(d)
            db=connect(home)
            ts=pd.date_range("2026-11-09 00:00",periods=24,freq="h",tz="Asia/Seoul")
            frame=pd.DataFrame({"timestamp":ts,"smp":[100.0]*24})
            payload=b"synthetic fixture only"
            ingest(db,"land",frame,payload,"test",home=home,retrieved="2026-11-09T14:01:00+00:00")
            before=latest(db,"land",as_of="2026-11-09T14:00:00+00:00")
            after=latest(db,"land",as_of="2026-11-09T14:02:00+00:00")
            self.assertTrue(before.empty)
            self.assertEqual(len(after),24)
            db.close()

    def test_same_public_response_does_not_create_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as d:
            home=Path(d); db=connect(home)
            ts=pd.date_range("2026-11-08 00:00",periods=24,freq="h",tz="Asia/Seoul")
            frame=pd.DataFrame({"timestamp":ts,"smp":[90.0]*24})
            ingest(db,"land",frame,b"response-a","test",home=home,retrieved="2026-11-09T13:59:00+00:00")
            ingest(db,"land",frame,b"response-b","test",home=home,retrieved="2026-11-09T14:02:00+00:00")
            self.assertEqual(db.execute("select count(*) from prices where market='land'").fetchone()[0],24)
            self.assertEqual(len(latest(db,"land")),24)
            db.close()

if __name__=="__main__": unittest.main()
