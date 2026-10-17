import copy,tempfile,unittest
from pathlib import Path
import numpy as np,pandas as pd
from v20.core import select_auto,eligible,publication_audit,grid_evidence,slots,preserve


class V20Tests(unittest.TestCase):
    def setUp(self):
        self.start=pd.Timestamp('2026-10-20',tz='Asia/Seoul');self.end=self.start+pd.Timedelta(days=1)
        self.clock=pd.Timestamp('2026-10-18',tz='Asia/Seoul')
        self.times=pd.date_range(self.start,periods=24,freq='h')
        self.official=pd.DataFrame(dict(timestamp=self.times,price=np.arange(24)+100.))
        self.none=self.official.iloc[:0]
    def item(self,path,day=None,**changes):
        times=self.times if day is None else pd.date_range(day,periods=24,freq='h')
        r=dict(mode='SIMULATED_TEST_INPUT',model_sha256='same_fixed_weights')
        m=dict(reference_origin='2026-10-17T23:00:00+09:00',issued_at='2026-10-17T23:00:01+09:00',model='test_only',strict_on_time_eligible=True,model_locked_before_origin=True,input_receipts_eligible=True,official_positive_observed_by_issue=False)
        m.update(changes);return (path,r,pd.DataFrame(dict(timestamp=times,predicted_smp=np.arange(24)+101.)),m)
    def test_official_full_period_preferred(self):
        result=select_auto(self.official,[],self.start,self.end,self.clock)
        self.assertEqual(result['mode'],'official');self.assertEqual(len(result['prices']),24)
    def test_verified_saved_ai_selected_without_creating_forecast(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'fixture.json';p.write_text('simulation fixture')
            r=select_auto(self.none,[self.item(p)],self.start,self.end,self.clock)
            self.assertEqual(r['mode'],'saved_ai');self.assertIn('현재 새 발급 아님',r['reason']);self.assertEqual(len(list(Path(t).iterdir())),1)
    def test_late_unknown_lock_receipt_known_price_and_future_issue_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'fixture.json';p.write_text('fixture')
            for changes in [dict(strict_on_time_eligible=False),dict(model_locked_before_origin=None),dict(model_locked_before_origin=False),dict(input_receipts_eligible=False),dict(official_positive_observed_by_issue=True),dict(issued_at='2026-10-17T23:02+09:00'),dict(issued_at='2026-10-19T23:00+09:00')]:
                self.assertEqual(select_auto(self.none,[self.item(p,**changes)],self.start,self.end,self.clock)['mode'],'waiting')
            item=list(self.item(p));item[1]['mode']='SIMULATION_ONLY';self.assertFalse(eligible(item[1],item[3],self.clock))
    def test_partial_official_is_not_mixed_with_incomplete_ai(self):
        self.assertEqual(select_auto(self.official.iloc[:12],[],self.start,self.end,self.clock)['mode'],'waiting')
    def test_overnight_requires_true_second_day_same_model_and_origin(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'fixture.json';p.write_text('fixture');a=self.item(p);b=self.item(p,self.start+pd.Timedelta(days=1))
            start=self.start+pd.Timedelta(hours=18);end=self.end+pd.Timedelta(hours=8)
            self.assertEqual(select_auto(self.none,[a],start,end,self.clock)['mode'],'waiting')
            self.assertEqual(select_auto(self.none,[a,b],start,end,self.clock)['mode'],'saved_ai')
            b=list(b);b[1]=dict(b[1],model_sha256='different');self.assertEqual(select_auto(self.none,[a,b],start,end,self.clock)['mode'],'waiting')
    def test_conflicting_overlapping_ai_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'fixture.json';p.write_text('fixture');a=self.item(p);b=copy.deepcopy(a);b[2]['predicted_smp']+=1
            self.assertEqual(select_auto(self.none,[a,b],self.start,self.end,self.clock)['mode'],'waiting')
    def test_zero_column_independent_check_and_not_absence(self):
        def payload(last):return ('<table><tr><th>hour</th><th>'+last+'</th></tr>'+''.join('<tr><td>'+str(h)+'h</td><td>0.0</td></tr>' for h in range(1,25))+'</table>').encode()
        r=grid_evidence(payload('10.20'),'2026-10-20')[0];self.assertTrue(r['all_zero']);self.assertTrue(r['last_column_matches_target'])
        self.assertFalse(grid_evidence(payload('10.19'),'2026-10-20')[0]['last_column_matches_target'])
    def test_zero_cross_source_conflict_and_later_change_preserved(self):
        base=dict(target_date='2026-10-19',market='land')
        rows=[dict(base,source='kpx_api',status='official_acquired',checked_at='2026-10-17T08:00+09:00'),dict(base,source='general_public',status='unverified_all_zero',checked_at='2026-10-17T09:00+09:00'),dict(base,source='general_public',status='official_acquired',checked_at='2026-10-17T10:00+09:00')]
        r=publication_audit(rows).query("source=='general_public'").iloc[0]
        self.assertTrue(r.zero_conflicts_with_already_acquired_other_source);self.assertTrue(r.zero_followed_by_numeric_receipt);self.assertEqual(r.zero_response_count,1)
    def test_slots_before23_are_not_due_not_success(self):
        self.assertTrue(all(r['status']=='not_due' for r in slots(pd.Timestamp('2026-10-17T13:00+09:00'))))
    def test_preserved_files_unchanged(self):self.assertEqual(preserve()['changed'],0)
    def test_web_comparison_ev_routing_status_and_save(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file('app_v20.py').run(timeout=60);self.assertFalse(app.exception);self.assertFalse(app.error)
        next(b for b in app.button if b.label=='비교 시뮬레이션 로컬 저장').click().run(timeout=60);self.assertFalse(app.exception)
        nav=next(r for r in app.radio if r.label=='화면');nav.set_value('EV 충전 추천').run(timeout=60);self.assertFalse(app.exception);self.assertFalse(app.error)
        next(w for w in app.date_input if w.label=='연결 시작일').set_value(pd.Timestamp('2026-10-20').date())
        next(w for w in app.date_input if w.label=='출발일').set_value(pd.Timestamp('2026-10-21').date());app.run(timeout=60)
        self.assertFalse(app.exception);self.assertTrue(any('수신 대기' in w.value for w in app.warning))
        next(r for r in app.radio if r.label=='화면').set_value('연구 결과·시스템 상태').run(timeout=60);self.assertFalse(app.exception);self.assertFalse(app.error)
        next(b for b in app.button if b.label=='저장 자료 새로 고침').click().run(timeout=60);self.assertFalse(app.exception)
    def test_web_ai_routing_uses_simulated_selection_without_live_issue(self):
        from unittest.mock import patch
        from streamlit.testing.v1 import AppTest
        selection=dict(mode='saved_ai',label='저장된 적격 AI · 테스트 fixture',prices=self.official,reason='테스트용 선택 · 실제 발급 아님',records=[])
        with patch('v20.core.select_auto',return_value=selection):
            app=AppTest.from_file('app_v20.py').run(timeout=60)
            next(r for r in app.radio if r.label=='화면').set_value('EV 충전 추천').run(timeout=60)
            next(w for w in app.date_input if w.label=='연결 시작일').set_value(self.start.date())
            next(w for w in app.time_input if w.label=='연결 시작시각').set_value(__import__('datetime').time(0))
            next(w for w in app.date_input if w.label=='출발일').set_value(self.end.date())
            next(w for w in app.time_input if w.label=='출발시각').set_value(__import__('datetime').time(0));app.run(timeout=60)
            self.assertFalse(app.exception);self.assertFalse(app.error);self.assertTrue(any('테스트 fixture' in w.value for w in app.success))


if __name__=='__main__':unittest.main()
