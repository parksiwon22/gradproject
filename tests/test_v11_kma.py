import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import io
import pandas as pd
from pandas.testing import assert_frame_equal
from v11 import kma_hourly as hourly
from v11.kma_apihub import request
from v11.kma_hourly_model import attach

def payload(temperature='-9.0',ww=''):
    # Numeric prefix through WP; WW can be absent, one code, or several codes.
    prefix='202301010000 108 23 2.2 -9 -9 -9 1000 1010 7 -1 '+temperature+' -1 85 16 0 0 0 -9 -9 -9 -9 21 62 '
    tail=' 8 3 4 - -9 -9 -9 1363 0.0 0.89 -9 21 18 19 20 21 -9 -9 -9 4 1\n'
    return ('#START7777\n'+prefix+ww+tail+'#7777END').encode()

class KmaTests(unittest.TestCase):
    def test_variable_width_weather_code_and_negative_temperature(self):
        for ww in ('','01','01 02'):
            row=hourly.parse(payload(ww=ww)).iloc[0]
            self.assertEqual(row.temperature_c,-9.0)
            self.assertEqual(row.sunshine_hours,0)
            self.assertEqual(row.solar_radiation_mj_m2,.89)
            self.assertEqual(row.timestamp.hour,0)
    def test_missing_never_filled(self):
        self.assertTrue(pd.isna(hourly.parse(payload('-99.0')).iloc[0].temperature_c))
    def test_http_error_never_exposes_key_or_url(self):
        secret='TEST_PRIVATE_CREDENTIAL'
        err=urllib.error.HTTPError('https://example/?authKey='+secret,403,'denied',{},io.BytesIO('API 활용신청 필요'.encode()))
        with patch('urllib.request.urlopen',side_effect=err):
            body,status=request('x',{},secret)
        self.assertIsNone(body); self.assertNotIn(secret,json.dumps(status))
        self.assertEqual(status['reason'],'service_application_or_approval')
    def test_echo_response_suppressed(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read(self): return b'authKey=TEST_SECRET'
        with patch('urllib.request.urlopen',return_value=Response()):
            body,status=request('x',{},'TEST_SECRET')
        self.assertIsNone(body)
    def test_cache_revision_and_failure_preservation(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(hourly,'FOLDER',Path(tmp)),patch.object(hourly,'key',return_value='SECRET'):
            t=pd.Timestamp('2023-01-01',tz='Asia/Seoul')
            with patch.object(hourly,'request',return_value=(payload(),{})): hourly.collect_one(t)
            marker=Path(tmp)/'hourly_raw/202301010000.json'; original=marker.read_bytes()
            with patch.object(hourly,'request',return_value=(None,{'http_status':403})):
                hourly.collect_one(t,True)
            self.assertEqual(marker.read_bytes(),original)
            with patch.object(hourly,'request',return_value=(payload('1.0'),{})): hourly.collect_one(t,True)
            self.assertEqual(len(list(marker.parent.glob('*.raw'))),2)
            self.assertEqual(len(list(marker.parent.glob('*.manifest.json'))),2)
    def test_future_observations_cannot_change_lag_features(self):
        target=pd.Timestamp('2026-10-03',tz='Asia/Seoul')
        base=pd.DataFrame({'timestamp':[target]})
        obs=pd.DataFrame([dict(timestamp=t,station_id=stn,**{f:float(i) for f in hourly.FIELDS}) for stn in hourly.STATIONS for i,t in enumerate([target-pd.Timedelta(hours=48),target])])
        a,cols=attach(base,obs)
        obs.loc[obs.timestamp==target,hourly.FIELDS]=9999
        b,_=attach(base,obs)
        assert_frame_equal(a[cols],b[cols])

if __name__=='__main__': unittest.main()
