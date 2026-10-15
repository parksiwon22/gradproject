import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import urllib.error
import numpy as np
import pandas as pd
from v11.exogenous_collect import parse_daily,parse_power,EXO,ELEMENTS
from v11.demand_forecasts import parse_week
from v11.exogenous_model import attach,read_date
from v11.exogenous_live import parse_forecast,record_observations,asof_observations
from v11.authenticated_data import call
from v11.features import build,FEATURES
from v11.store import HOME

TEMP=HOME/'test_tmp'; TEMP.mkdir(exist_ok=True)

class ExogenousTests(unittest.TestCase):
    def test_wind_numeric_literal_not_javascript_execution(self):
        cells='<td>1일</td>'+''.join("<td>writeWindSpeed('2.7', false,'','',1);</td>" for _ in range(12))
        html='<table id="weather_table"><tr>'+cells+'</tr></table>'
        f=parse_daily(html,2024,'wind'); self.assertEqual(len(f),12); self.assertTrue(f.wind.eq(2.7).all())
        with self.assertRaises(ValueError): parse_daily('empty',2024,'wind')

    def test_archived_forecast_market_and_hour_alignment(self):
        text='<input value="2024-02-29">'+''.join('<tr><td>'+str(h)+'h</td>'+''.join('<td>65,000</td>' for _ in range(7))+'</tr>' for h in range(1,25))
        f=parse_week(text,'2024-02-29'); self.assertEqual(len(f),168)
        self.assertEqual(f.timestamp.max(),pd.Timestamp('2024-02-29 23:00',tz='Asia/Seoul'))
        self.assertTrue(f.day_ahead_demand_mw.eq(65000).all())

    def test_target_observations_cannot_change_causal_features(self):
        raw=pd.DataFrame({'timestamp':pd.date_range('2026-01-01',periods=25*24,freq='h',tz='Asia/Seoul'),'smp':np.arange(25*24,dtype=float)})
        dates=pd.date_range('2026-01-01',periods=25,freq='D',tz='Asia/Seoul')
        weather=pd.DataFrame({'date':dates,**{name:np.arange(25,dtype=float) for name in ELEMENTS.values()}})
        power=pd.DataFrame({'date':dates,**{name:np.arange(25,dtype=float)+50000 for name in ['daily_max_demand_mw','supply_capacity_mw','reserve_mw','reserve_rate_percent']}})
        for ahead in (1,2):
            base=build(raw,days_ahead=ahead); a,w,p,_=attach(base,weather,power,days_ahead=ahead)
            changed=weather.copy(); changed.loc[changed.date>=pd.Timestamp('2026-01-20',tz='Asia/Seoul'),list(ELEMENTS.values())]=999999
            b,_,_,_=attach(base,changed,power,days_ahead=ahead)
            same=a.target_date.eq('2026-01-20')
            np.testing.assert_equal(a.loc[same,w+p].to_numpy(),b.loc[same,w+p].to_numpy())

    def test_receipt_time_and_revision_guard(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as tmp:
            folder=Path(tmp)
            pd.DataFrame({'date':['2026-10-14T00:00:00+09:00'],'temperature_mean_c':[15.]}).to_csv(folder/'weather_daily_all.csv',index=False)
            pd.DataFrame({'date':['2026-10-14T00:00:00+09:00'],'daily_max_demand_mw':[70000.]}).to_csv(folder/'power_daily_all.csv',index=False)
            with patch('v11.exogenous_live.EXO',folder):
                record_observations('2026-10-15T12:00:00+00:00')
                pd.DataFrame({'date':['2026-10-14T00:00:00+09:00'],'temperature_mean_c':[99.]}).to_csv(folder/'weather_daily_all.csv',index=False)
                record_observations('2026-10-15T15:00:00+00:00')
                weather,_=asof_observations('2026-10-15T14:00:00+00:00')
                self.assertEqual(weather.temperature_mean_c.iloc[0],15.)

    def test_credentials_never_appear_in_exception(self):
        secret='TEST_CREDENTIAL_DO_NOT_LOG'
        error=urllib.error.HTTPError('https://example.invalid?serviceKey='+secret,403,'Forbidden',None,None)
        with patch('urllib.request.urlopen',side_effect=error):
            with self.assertRaises(RuntimeError) as captured: call('https://example.invalid',{},secret)
        self.assertNotIn(secret,str(captured.exception)); self.assertEqual(str(captured.exception),'API HTTP 403')

    def test_real_kma_forecast_timestamps(self):
        source=EXO/'latest_weather_forecast.html'
        if not source.exists(): self.skipTest('private observation asset absent')
        meta=json.loads(source.with_suffix('.html.manifest.json').read_text(encoding='utf-8'))
        f=parse_forecast(source.read_text(encoding='utf-8'),meta['retrieved_at'])
        self.assertFalse(f.timestamp.duplicated().any())
        self.assertTrue((pd.to_datetime(f.issued_at,utc=True)<=pd.to_datetime(f.first_seen,utc=True)).all())
        self.assertTrue(f.humidity_percent.between(0,100).all())
        self.assertGreaterEqual(len(f),48)

    def test_executed_common_rows_and_actual_future_inputs(self):
        path=EXO/'experiments/experiments.json'
        if not path.exists(): self.skipTest('private experiment asset absent')
        records=json.loads(path.read_text(encoding='utf-8'))
        for study in ['2025_hourly_demand','2026_daily_exogenous']:
            selected=[r for r in records if r['key'].startswith(study)]
            self.assertEqual(len(selected),8)
            self.assertEqual(len({tuple(sorted(r['rows'].items())) for r in selected}),1)
            for r in selected: self.assertLess(r['train_end'],r['validation_start']); self.assertLess(r['validation_end'],r['test_start'])
        for file in (EXO/'forecasts').glob('*.json'):
            record=json.loads(file.read_text(encoding='utf-8'))
            self.assertEqual(len(record['rows']),24)
            self.assertFalse(record['weather_forecast_used'])
            self.assertLess(pd.Timestamp(record['first_exogenous_day']),pd.Timestamp(record['reference_23h_origin']))

if __name__=='__main__': unittest.main()
