import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,MagicMock
import numpy as np,pandas as pd
from v18.publication import inspect_public,inspect_api,gate,summarize
from v18.common import HOME,ROOT,read,sha
from v18.evaluation import compare_ev,inventory,evaluate,price_metrics
from v18.forward import eligible
from v12.ev import EV

class V18Tests(unittest.TestCase):
    def test_public_empty_blank_zero_partial_and_price_are_distinct(self):
        def page(values):return ('<input value="2026-10-20">'+''.join(f'<tr><td>{h}h</td><td>{v}</td></tr>' for h,v in enumerate(values,1))).encode()
        self.assertEqual(inspect_public(b'','2026-10-20')[0],'empty_response')
        self.assertEqual(inspect_public(page(['-']*24),'2026-10-20')[0],'not_observed')
        self.assertEqual(inspect_public(page([0]*24),'2026-10-20')[0],'unverified_all_zero')
        self.assertEqual(inspect_public(page(['-']+[80]*23),'2026-10-20')[0],'partial_response')
        status,f=inspect_public(page([-10]+[80]*23),'2026-10-20');self.assertEqual(status,'official_acquired');self.assertEqual(f.smp.iloc[0],-10)
        with self.assertRaises(ValueError):inspect_public(page([80]*24),'2026-10-21')
    def test_api_empty_error_normal_no_data_are_not_interchanged(self):
        self.assertEqual(inspect_api(dict(response=dict(header=dict(resultCode='00'),body=dict(totalCount=0,items=[]))),'2026-10-20')[0],'normal_no_data')
        self.assertEqual(inspect_api(dict(response=dict(header=dict(resultCode='00'),body=dict(items=[]))),'2026-10-20')[0],'empty_response')
        self.assertEqual(inspect_api(dict(response=dict(header=dict(resultCode='99'))),'2026-10-20')[0],'api_error')
    def test_api_both_markets_48_rows_are_separated_not_duplicated(self):
        items=[dict(date='20261017',areaName=area,hour=h,smp=price+h) for area,price in [('육지',80),('제주',120)] for h in range(1,25)];data=dict(response=dict(header=dict(resultCode='00'),body=dict(items=dict(item=items))))
        state,land=inspect_api(data,'2026-10-17');state2,jeju=inspect_api(data,'2026-10-17','jeju');self.assertEqual(state,'official_acquired');self.assertEqual(state2,state);self.assertEqual(len(land),24);self.assertEqual(len(jeju),24);self.assertEqual(land.smp.iloc[0],81);self.assertEqual(jeju.smp.iloc[0],121)
    def test_positive_any_source_failed_stale_and_normal_absence_gate(self):
        clock=pd.Timestamp('2026-10-17T23:00:00+09:00');rows=[dict(target_date='2026-10-19',market='land',source=s,status='not_observed',checked_at=(clock-pd.Timedelta(seconds=10)).isoformat()) for s in ['kpx_api','general_public','epsis']]
        self.assertTrue(gate(rows,'2026-10-19',clock)[0]);rows[0]['status']='empty_response';self.assertFalse(gate(rows,'2026-10-19',clock)[0]);rows[0]['status']='normal_no_data';self.assertTrue(gate(rows,'2026-10-19',clock)[0]);rows.append(dict(rows[0],status='official_acquired',checked_at=(clock-pd.Timedelta(minutes=5)).isoformat()));self.assertFalse(gate(rows,'2026-10-19',clock)[0]);rows.pop();self.assertFalse(gate(rows,'2026-10-19',clock+pd.Timedelta(hours=1))[0])
    def test_source_interval_never_uses_failure_or_post_positive_as_negative(self):
        rows=[dict(target_date='2026-10-19',market='land',source='kpx_api',status=s,checked_at=f'2026-10-17T0{h}:00:00+09:00') for s,h in [('not_observed',6),('failed',7),('official_acquired',8),('not_observed',9)]]
        with tempfile.TemporaryDirectory() as t,patch('v18.publication.HOME',Path(t)):
            result=summarize(rows)[0];self.assertEqual(result['last_strict_normal_nonobservation'],rows[0]['checked_at']);self.assertEqual(result['first_normal_receipt'],rows[2]['checked_at']);self.assertTrue(result['first_before_D2_23']);self.assertEqual(result['exact_first_publication'],'unverified')
    def test_model_registration_and_receipt_must_precede_origin(self):
        bound=pd.Timestamp('2026-10-17T23:00:00+09:00');m=dict(registered_at='2026-10-17T08:00:00+09:00',trained_through='2026-06-30');self.assertTrue(eligible(m,bound,pd.Series(['2026-10-17T01:00:00Z']))[0]);self.assertFalse(eligible(dict(m,registered_at='2026-10-17T23:01:00+09:00'),bound,pd.Series([],dtype=str))[0]);self.assertFalse(eligible(m,bound,pd.Series(['2026-10-17T14:01:00Z']))[0])
    def test_known_official_skips_new_ai_and_caller_backdating_rejected(self):
        import v18.forward as fw
        clock=pd.Timestamp.now(tz='Asia/Seoul');day=fw.origin(clock).normalize()+pd.Timedelta(days=2);prices=pd.DataFrame(dict(timestamp=pd.date_range(day,periods=24,freq='h'),first_seen=[clock.isoformat()]*24))
        with tempfile.TemporaryDirectory() as t,patch.object(fw,'HOME',Path(t)),patch.object(fw,'all_rows',return_value=[]),patch.object(fw,'connect',return_value=MagicMock()),patch.object(fw,'latest',return_value=prices):
            result=fw.issue();self.assertEqual(result['mode'],'official_price_optimization');self.assertFalse(list((Path(t)/'forecasts').glob('*.json')))
        with self.assertRaises(ValueError):fw.issue(pd.Timestamp('2025-01-01T23:00:00+09:00'))
    def test_identical_prices_same_ev_regret_zero_with_negative_partial_hours(self):
        times=pd.date_range('2026-10-19',periods=6,freq='h',tz='Asia/Seoul');p=np.array([100,200,-20,50,150,0]);ev=EV(capacity=10,initial_soc=30,target_soc=80,charge_kw=3,degradation=5);r=compare_ev(p,p,times,ev,times[0]+pd.Timedelta(minutes=15),times[-1]+pd.Timedelta(minutes=40));self.assertAlmostEqual(r['G2V_regret'],0,places=6);self.assertAlmostEqual(r['V2G_regret'],0,places=5)
        for name in ['G2V','V2G','official_G2V','official_V2G']:
            s=r[name]['schedule'];self.assertTrue((s.soc_end.between(20-1e-6,90+1e-6)).all());self.assertTrue((np.minimum(s.charge_kw,s.discharge_kw)<1e-6).all());self.assertAlmostEqual(s.soc_end.iloc[-1],80,places=5)
    def test_no_peaks_has_undefined_detection_not_false_success(self):
        m=price_metrics(np.arange(24)+80,np.arange(24)+81);self.assertIsNone(m['peak_MAE']);self.assertIsNone(m['peak_F1']);self.assertAlmostEqual(m['MAE'],1);self.assertEqual(m['cheap5_overlap'],1)
    def test_inventory_and_evaluation_leave_original_bytes_unchanged(self):
        before={str(p):sha(p) for p,_,_,_ in inventory()};status=evaluate();self.assertEqual(before,{str(p):sha(p) for p,_,_,_ in inventory()});self.assertLessEqual(status['strict_on_time_eligible_records'],status['on_time_forecast_records']);self.assertFalse(status['operating_model_changed']);self.assertEqual(status['final_corrected_statistics_verified_days'],0)
    def test_overlapping_service_rejected(self):
        from v18.service import lock
        with tempfile.TemporaryDirectory() as t,patch('v18.service.HOME',Path(t)):
            with lock():
                with self.assertRaises(RuntimeError):
                    with lock():pass
    def test_web_auto_recommendation_archive_verification_status_and_save(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file('app_v18.py').run(timeout=60);self.assertFalse(app.exception);self.assertFalse(app.error);self.assertTrue(any(m.label=='추천 종료 SOC' for m in app.metric))
        next(r for r in app.radio if r.label=='계획 자료').set_value('과거 저장 예측 재생').run(timeout=60);self.assertFalse(app.exception);self.assertFalse(app.error)
        next(b for b in app.button if b.label=='이 시뮬레이션 결과 로컬 저장').click().run(timeout=60);self.assertFalse(app.exception);self.assertTrue(any('로컬 시뮬레이션' in s.value for s in app.success))
        next(r for r in app.radio if r.label=='화면').set_value('예측 검증').run(timeout=60);self.assertFalse(app.exception);self.assertFalse(app.error)
        next(r for r in app.radio if r.label=='화면').set_value('연구·수집 상태').run(timeout=60);self.assertFalse(app.exception);self.assertFalse(app.error)
        next(b for b in app.button if b.label=='화면 자료 새로 고침').click().run(timeout=60);self.assertFalse(app.exception)

if __name__=='__main__':unittest.main()
