import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,MagicMock
import numpy as np,pandas as pd
from v12.ev import EV,optimize
from v17.common import HOME,sha
from v17.charging import priority_plan,require_prices
from v17.legacy import fresh_continuation

class V17Tests(unittest.TestCase):
    def test_priority_dispatch_matches_price_lp_with_negative_partial_hours(self):
        t=pd.date_range('2026-10-18',periods=6,freq='h',tz='Asia/Seoul');p=np.array([80,-20,90,40,10,-5.]);ev=EV(capacity=10,initial_soc=30,target_soc=80,charge_kw=3,degradation=5)
        start=t[0]+pd.Timedelta(minutes=15);end=t[-1]+pd.Timedelta(minutes=45);g=priority_plan(p,t,ev=ev,start=start,end=end);old=optimize(p,t,ev,start,end)
        self.assertAlmostEqual(g['net_cost'],old['net_cost'],places=6);self.assertAlmostEqual(g['ending_soc'],80,places=6)
        s=g['schedule'];self.assertTrue((s.charge_kw<=3+1e-8).all());self.assertTrue((s.soc_end.between(20-1e-8,90+1e-8)).all());self.assertTrue((s.discharge_kw==0).all())
        self.assertTrue(np.allclose(s.soc_end-s.soc_start,s.charge_kw*s.connected_hours*.9/10*100,atol=1e-8))
    def test_short_connection_infeasible_not_filled(self):
        for n in [3,4]:
            t=pd.date_range('2026-10-18',periods=n,freq='h',tz='Asia/Seoul')
            with self.assertRaisesRegex(ValueError,'목표 SOC'):priority_plan(np.arange(n),t)
    def test_overnight_requires_real_second_date(self):
        t=pd.date_range('2026-10-18',periods=48,freq='h',tz='Asia/Seoul');frame=pd.DataFrame(dict(timestamp=t,price=np.r_[np.arange(24),100+np.arange(24)]))
        start=t[18];end=t[32];selected=require_prices(frame,start,end);self.assertEqual(len(selected),14);self.assertTrue((selected.price.iloc[6:].to_numpy()==np.arange(100,108)).all())
        with self.assertRaisesRegex(ValueError,'다음 날짜'):require_prices(frame.iloc[:24],start,end)
    def test_live_clock_refresh_allows_io_delay_but_not_reference_change(self):
        a=pd.Timestamp('2026-10-16T23:00:01+09:00');self.assertEqual(fresh_continuation(a,a+pd.Timedelta(seconds=12)),a+pd.Timedelta(seconds=12))
        with self.assertRaises(ValueError):fresh_continuation(a,a-pd.Timedelta(seconds=1))
        with self.assertRaises(ValueError):fresh_continuation(a,a+pd.Timedelta(days=1))
    def test_cached_official_api_price_validation(self):
        from v17.source_prices import parse_api
        items=[dict(date='20261018',areaName='육지',hour=h,smp=80+h) for h in range(1,25)];data=dict(response=dict(header=dict(resultCode='00'),body=dict(items=dict(item=items))))
        f=parse_api(data,'20261018');self.assertEqual(len(f),24);self.assertEqual(f.timestamp.iloc[0].hour,0);self.assertEqual(f.timestamp.iloc[-1].hour,23)
        with self.assertRaisesRegex(ValueError,'Wrong official'):parse_api(data,'20261019')
        items[1]['hour']=1
        with self.assertRaisesRegex(ValueError,'Duplicate'):parse_api(data,'20261018')
    def test_cached_api_cannot_overwrite_later_price_correction(self):
        import hashlib
        import v17.source_prices as sp
        from v11.store import connect,ingest,latest
        items=[dict(date='20261018',areaName='육지',hour=h,smp=80+h) for h in range(1,25)];data=dict(response=dict(header=dict(resultCode='00'),body=dict(items=dict(item=items))));raw=json.dumps(data).encode();digest=hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            home=Path(temp);(home/'sources').mkdir();(home/'sources'/(digest+'.raw')).write_bytes(raw);(home/'sources/20261018_demand_check.json').write_text(json.dumps(dict(sha256=digest,resultCode='00',target_date='20261018',first_seen='2026-10-16T00:00:00+00:00')))
            with patch.object(sp,'HOME',home),patch.object(sp,'connect',side_effect=lambda:connect(home/'db')):
                sp.run();db=connect(home/'db');f=latest(db,'land');f['smp']=180.;ingest(db,'land',f,b'later correction','test',home=home/'db');db.close();sp.run();db=connect(home/'db');self.assertTrue(latest(db,'land').smp.eq(180).all());db.close()
    def test_shared_origin_d2_d3_no_future_target_leak(self):
        from v11.store import connect,latest
        from v17.study import matrix,query,INPUTS
        db=connect();p=latest(db,'land');db.close();target=pd.Timestamp('2025-05-05',tz='Asia/Seoul');a=matrix(p,False,target);changed=p.copy();changed.loc[changed.timestamp>=target,'smp']=9999;b=matrix(changed,False,target)
        self.assertEqual(len(a),48);self.assertEqual(a.origin_timestamp.nunique(),1);self.assertTrue(np.allclose(a[[c for c in INPUTS if c in a]],b[[c for c in INPUTS if c in b]]));q=query(a,18,14);self.assertEqual(q.timestamp.dt.date.nunique(),2)
    def test_old_rank_focus_and_new_low_price_relevance(self):
        a=np.array([50.,70.,110.]);old=pd.Series(a).rank(method='first').astype(int).to_numpy()-1;new=np.rint(7*(a.max()-a)/(np.ptp(a)+1e-9)).astype(int)
        self.assertEqual(old.argmax(),2);self.assertEqual(new.argmax(),0)
        design=json.loads((HOME/'study/design_before_scores.json').read_text(encoding='utf-8'));self.assertFalse(design['operating_model_changed']);self.assertTrue(design['year2026_only_post_selection_audit'])
    def test_official_known_prevents_new_ai_record(self):
        import v17.forward as fw
        now=pd.Timestamp.now(tz='Asia/Seoul');bound=fw.origin(now);day=bound.normalize()+pd.Timedelta(days=2)
        with tempfile.TemporaryDirectory() as temp:
            home=Path(temp);(home/'study').mkdir();(home/'study/selection_before_audit.json').write_text('{}');(home/'study/winter2025.joblib').write_bytes(b'not-loaded-because-official-known')
            prices=pd.DataFrame(dict(timestamp=pd.date_range(day,periods=24,freq='h')))
            with patch.object(fw,'HOME',home),patch.object(fw,'ensure',return_value=[]),patch.object(fw,'connect',return_value=MagicMock()),patch.object(fw,'latest',return_value=prices):r=fw.issue()
            self.assertIn('Official target price',r['reason']);self.assertEqual(len(list((home/'forecasts').glob('*.json'))),0)
        with self.assertRaisesRegex(ValueError,'backdate'):fw.issue(pd.Timestamp('2025-01-01T23:00:00+09:00'))
    def test_real_forward_record_replays_hash_and_receipt_snapshot(self):
        from v17.study import matrix
        files=list((HOME/'forecasts').glob('*.json'))
        if not files:self.skipTest('No actual eligible V17 forecast yet')
        for p in files:
            r=json.loads(p.read_text());source=pd.DataFrame(r['source_prices']);source.timestamp=pd.to_datetime(source.timestamp,utc=True).dt.tz_convert('Asia/Seoul');origin=pd.Timestamp(r['reference_origin']);self.assertTrue((pd.to_datetime(source.first_seen,utc=True)<=origin).all());self.assertTrue((source.timestamp<=origin).all());self.assertFalse(r['official_known_at_issue'])
            f=matrix(source,False,pd.Timestamp(r['primary_target_date'],tz='Asia/Seoul'));saved=pd.DataFrame(r['price_features']);self.assertTrue(np.allclose(f.iloc[:len(saved)][saved.columns],saved,atol=1e-7));self.assertIn(len(r['price_rows']),[24,48]);self.assertEqual(sha(HOME/'study/winter2025.joblib'),r['model_sha256'])
    def test_primary24_not_blocked_by_failed_d3_and_no_cloned_night(self):
        import shutil
        import v17.forward as fw
        from v11.store import connect,latest
        db=connect();p=latest(db,'land');db.close();bound=fw.origin(pd.Timestamp.now(tz='Asia/Seoul'));p=p[p.timestamp<=bound]
        with tempfile.TemporaryDirectory() as temp:
            home=Path(temp);(home/'study').mkdir()
            for name in ['selection_before_audit.json','winter2025.joblib']:shutil.copyfile(HOME/'study'/name,home/'study'/name)
            def eligible(rows,day,clock):return (day==(bound.normalize()+pd.Timedelta(days=2)).strftime('%Y-%m-%d'),'test primary only')
            with patch.object(fw,'HOME',home),patch.object(fw,'ensure',return_value=[]),patch.object(fw,'connect',return_value=MagicMock()),patch.object(fw,'latest',return_value=p),patch.object(fw,'publication_eligible',side_effect=eligible):result=fw.issue()
            self.assertEqual(result['status'],'created');r=json.loads(next((home/'forecasts').glob('*.json')).read_text());self.assertEqual(len(r['price_rows']),24);self.assertFalse(r['auxiliary_prediction_published']);self.assertEqual(r['decisions']['night18_08']['status'],'awaiting_eligible_D3');self.assertEqual(r['decisions']['full24']['status'],'frozen')
    def test_streamlit_official_example_and_frozen_replay(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file('app_v17.py').run(timeout=60);self.assertFalse(app.exception);self.assertEqual(len(app.tabs),4)
        app.button[0].click().run(timeout=60);self.assertFalse(app.exception);self.assertFalse(app.error)
        app.radio[0].set_value('저장된 실제 전향 예측').run(timeout=60);self.assertFalse(app.exception)
        button=next((b for b in app.button if b.label=='저장된 사전 충전계획 보기'),None)
        if button is not None:button.click().run(timeout=60);self.assertFalse(app.exception);self.assertFalse(app.error)
        app.radio[0].set_value('과거 고정 검증 예제').run(timeout=60);app.button[0].click().run(timeout=60);self.assertFalse(app.exception);self.assertFalse(app.error)

if __name__=='__main__':unittest.main()
