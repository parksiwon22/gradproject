import json,tempfile,time,copy
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np,pandas as pd
from v12.ev import EV
from v15.policies import dispatch
from v17.charging import priority_plan,require_prices
from v17.forward import pack
from v19.common import HOME,ROOT,read,write,sha,next_origin
from v19.publisher import check_stage,check_ledger

def fixture(home,bound):
    import v19.publisher as pub
    weight=home/'weights.bin';weight.write_bytes(b'SIMULATION MODEL FIXTURE; not a trained model');locked=bound-pd.Timedelta(days=1)
    registry=dict(locked_at=locked.isoformat(),entries={'V12_HGB':dict(path='weights.bin',horizons=[2],sha256=sha(weight),trained_through='2025-06-30')});write(home/'model_registry.json',registry)
    target=(bound.normalize()+pd.Timedelta(days=2)).strftime('%Y-%m-%d');times=pd.date_range(pd.Timestamp(target,tz='Asia/Seoul'),periods=24,freq='h');p=np.arange(24)+80.;ev=EV(degradation=5);g=priority_plan(p,times,ev=ev);v=dispatch('general',p,times,None,ev)
    stage=dict(simulation_fixture=True,reference_origin=bound.isoformat(),prepared_at=(bound-pd.Timedelta(minutes=1)).isoformat(),input_snapshot_at=(bound-pd.Timedelta(minutes=2)).isoformat(),input_first_seen_max=(bound-pd.Timedelta(minutes=3)).isoformat(),registry_sha256=sha(home/'model_registry.json'),source_prices=[dict(timestamp=(bound-pd.Timedelta(hours=1)).isoformat(),first_seen=(bound-pd.Timedelta(minutes=3)).isoformat(),smp=100,snapshot='test')],packets=[dict(model='V12_HGB',model_sha256=sha(weight),comparison_group='SIMULATION_ONLY',target_date=target,horizon_days=2,trained_through='2025-06-30',price_rows=[dict(timestamp=t.isoformat(),predicted_smp=float(x)) for t,x in zip(times,p)],features=[dict(daily=80)]*24,inference_seconds=.01,EV_seconds=.01,settlement_assumptions=ev.__dict__,decisions=dict(G2V=pack(g),V2G=pack(v)),priority_scores=None,decision_prices_are_proxy=False)])
    path=home/'prepared/fixture.json';write(path,stage);write(home/'ready'/(bound.strftime('%Y%m%dT%H%M')+'.json'),dict(path='prepared/fixture.json',sha256=sha(path)))
    ledger=dict(days={target:dict(ever_positive=False,latest={s:dict(status='not_observed',checked_at=pd.Timestamp.now(tz='Asia/Seoul').isoformat(),legacy_parser=False) for s in ['kpx_api','general_public','epsis']})});write(home/'publication_ledger.json',ledger);return stage,registry,target

class V19Tests(unittest.TestCase):
    def test_next_origin_and_already_passed_reference_not_backfilled(self):
        self.assertEqual(next_origin(pd.Timestamp('2026-10-17T22:00+09:00')),pd.Timestamp('2026-10-17T23:00+09:00'));self.assertEqual(next_origin(pd.Timestamp('2026-10-17T23:01+09:00')),pd.Timestamp('2026-10-18T23:00+09:00'))
    def test_registry_preparation_and_receipts_must_precede_origin(self):
        bound=pd.Timestamp('2026-10-17T23:00+09:00')
        with tempfile.TemporaryDirectory() as t:
            stage,registry,_=fixture(Path(t),bound);self.assertTrue(check_stage(stage,registry,bound)[0]);self.assertFalse(check_stage(dict(stage,prepared_at=bound.isoformat()),registry,bound)[0]);self.assertFalse(check_stage(stage,dict(registry,locked_at=bound.isoformat()),bound)[0]);bad=copy.deepcopy(stage);bad['source_prices'][0]['first_seen']=(bound+pd.Timedelta(seconds=1)).isoformat();self.assertFalse(check_stage(bad,registry,bound)[0])
    def test_publication_positive_zero_failure_and_staleness_block(self):
        clock=pd.Timestamp.now(tz='Asia/Seoul');day='2026-10-20';entry=dict(ever_positive=False,latest={s:dict(status='not_observed',checked_at=clock.isoformat()) for s in ['kpx_api','general_public','epsis']});self.assertTrue(check_ledger(dict(days={day:entry}),day,clock)[0]);entry['latest']['general_public']['status']='unverified_all_zero';self.assertFalse(check_ledger(dict(days={day:entry}),day,clock)[0]);entry['latest']['general_public']['status']='not_observed';self.assertFalse(check_ledger(dict(days={day:entry}),day,clock+pd.Timedelta(minutes=4))[0]);entry['ever_positive']=True;self.assertFalse(check_ledger(dict(days={day:entry}),day,clock)[0])
    def test_hot_publisher_simulation_no_network_inference_or_reoptimization(self):
        import v19.publisher as pub
        with tempfile.TemporaryDirectory() as t:
            home=Path(t);bound=pd.Timestamp.now(tz='Asia/Seoul')-pd.Timedelta(seconds=10);stage,registry,day=fixture(home,bound)
            with patch.object(pub,'HOME',home),patch.object(pub,'ROOT',home),patch.object(pub,'origin',return_value=bound),patch.object(pub,'local_known',return_value=False),patch('urllib.request.urlopen',side_effect=AssertionError('No network in hot path')),patch('v11.authenticated_data.call',side_effect=AssertionError('No API in hot path')),patch('v19.models.predict_model',side_effect=AssertionError('No inference in hot path')),patch('v15.policies.dispatch',side_effect=AssertionError('No EV solve in hot path')):
                result=pub.publish_once();self.assertEqual(result['records'][0]['status'],'created');self.assertEqual(result['network_calls'],0);self.assertLess(result['local_publish_seconds'],5)
                path=next((home/'forecasts').glob('*.json'));before=sha(path);record=read(path);self.assertEqual(record['mode'],'SIMULATION_ONLY');self.assertEqual(record['rows'],stage['packets'][0]['price_rows']);self.assertEqual(result['records'][0]['timing'],'on_time_within_60s');again=pub.publish_once();self.assertEqual(again['records'][0]['status'],'existing_immutable_record');self.assertEqual(sha(path),before)
    def test_simulated_late_commit_separately_classified_and_known_price_skip(self):
        import v19.publisher as pub
        with tempfile.TemporaryDirectory() as t:
            home=Path(t);bound=pd.Timestamp.now(tz='Asia/Seoul')-pd.Timedelta(minutes=2);fixture(home,bound)
            with patch.object(pub,'HOME',home),patch.object(pub,'ROOT',home),patch.object(pub,'origin',return_value=bound),patch.object(pub,'local_known',return_value=True):self.assertIn('official_price',pub.publish_once()['records'][0]['reason']);self.assertFalse(list((home/'forecasts').glob('*.json')))
            with patch.object(pub,'HOME',home),patch.object(pub,'ROOT',home),patch.object(pub,'origin',return_value=bound),patch.object(pub,'local_known',return_value=False):self.assertEqual(pub.publish_once()['records'][0]['timing'],'late')
    def test_packet_and_model_hash_changes_block_without_replacement(self):
        import v19.publisher as pub
        with tempfile.TemporaryDirectory() as t:
            home=Path(t);bound=pd.Timestamp.now(tz='Asia/Seoul')-pd.Timedelta(seconds=10);fixture(home,bound);(home/'weights.bin').write_bytes(b'changed')
            with patch.object(pub,'HOME',home),patch.object(pub,'ROOT',home),patch.object(pub,'origin',return_value=bound),patch.object(pub,'local_known',return_value=False):self.assertEqual(pub.publish_once()['records'][0]['reason'],'model_hash_changed')
    def test_warm_worker_cold_fallback_after23_does_not_wait_tomorrow(self):
        import v19.publisher as pub
        clock=pd.Timestamp('2026-10-17T23:00:02+09:00')
        with tempfile.TemporaryDirectory() as t,patch.object(pub,'HOME',Path(t)),patch.object(pub.pd.Timestamp,'now',return_value=clock),patch.object(pub,'publish_once',return_value={'SIMULATION':'fallback'}) as publish:
            self.assertEqual(pub.run(True),{'SIMULATION':'fallback'});publish.assert_called_once()
    def test_d2_d3_feature_inputs_change_only_with_eligible_history(self):
        from v11.store import connect,latest
        from v17.features import build
        db=connect();prices=latest(db,'land');db.close();target=pd.Timestamp('2025-05-05',tz='Asia/Seoul');times=pd.date_range(target,periods=24,freq='h')
        for ahead in [2,3]:
            a=build(prices,targets=times,days_ahead=ahead,include_labels=False);changed=prices.copy();changed.loc[changed.timestamp>a.origin_timestamp.iloc[0],'smp']=99999;b=build(changed,targets=times,days_ahead=ahead,include_labels=False);self.assertTrue(np.allclose(a.select_dtypes('number'),b.select_dtypes('number')));self.assertNotIn('actual_smp',a)
    def test_all_registered_families_predict24_from_same_origin_without_target_values(self):
        from v11.store import connect,latest
        from v19.models import SPECS,predict_model
        db=connect();prices=latest(db,'land');db.close();bound=pd.Timestamp('2026-10-17T23:00+09:00');prices=prices[prices.timestamp<=bound]
        for name,(_,horizons,_) in SPECS.items():
            for ahead in horizons:
                f,p,priority,planning=predict_model(name,prices,(bound.normalize()+pd.Timedelta(days=ahead)).strftime('%Y-%m-%d'),ahead);self.assertEqual(len(p),24);self.assertTrue(np.isfinite(p).all());self.assertEqual(f.origin_timestamp.iloc[0],bound);self.assertNotIn('actual_smp',f)
    def test_historical_study_one_fit_and_registry_shadow_not_promoted(self):
        design=read(HOME/'study/design_before_scores.json');self.assertEqual(design['new_fits'],1);self.assertTrue(design['october2026_excluded']);selection=read(HOME/'study/selection_before_live.json');self.assertFalse(selection['operating_model_replaced']);registry=read(HOME/'model_registry.json');self.assertTrue(all(sha(ROOT/r['path'])==r['sha256'] for r in registry['entries'].values()))
    def test_existing_records_evaluation_preserves_bytes_and_missing_pairs(self):
        from v19.evaluation import inventory,evaluate
        before={str(p):sha(p) for p,_,_,_ in inventory()};status=evaluate();self.assertEqual(before,{str(p):sha(p) for p,_,_,_ in inventory()});self.assertEqual(status['V19_new_actual_records'],len(list((HOME/'forecasts').glob('*.json'))));self.assertFalse(status['operating_model_replaced'])
    def test_web_missing_horizon_vehicle_overnight_replay_save_status(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file('app_v19.py').run(timeout=60);self.assertFalse(app.exception);self.assertFalse(app.error);self.assertTrue(any('D-3 실제 저장 예측 미확보' in i.value for i in app.info))
        next(s for s in app.selectbox if s.label=='동일 비교 대상일').set_value('2026-10-19').run(timeout=60);self.assertFalse(app.exception);self.assertTrue(any('D-2 실제 저장 예측 미확보' in i.value for i in app.info))
        next(b for b in app.button if b.label=='비교 시뮬레이션 로컬 저장').click().run(timeout=60);self.assertFalse(app.exception)
        next(r for r in app.radio if r.label=='화면').set_value('차량 충전 추천').run(timeout=60);self.assertFalse(app.exception);self.assertFalse(app.error)
        next(r for r in app.radio if r.label=='화면').set_value('연구·발급 현황').run(timeout=60);self.assertFalse(app.exception);self.assertFalse(app.error);next(b for b in app.button if b.label=='저장 자료 새로 고침').click().run(timeout=60);self.assertFalse(app.exception)

if __name__=='__main__':unittest.main()
