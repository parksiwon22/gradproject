"""Behavioral checks for the real 23:00/23:02 V20 publisher path."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

import v19.publisher as publisher
from v19.common import ROOT, read


SOURCES = ('kpx_api', 'general_public', 'epsis')


def temporary_directory():
    root = ROOT / 'artifacts/v20/test_temp'
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root)


def refresh_fixture(clock, overrides=None):
    overrides = overrides or {}
    days = ['2026-10-23', '2026-10-24']
    rows, ledger_days = [], {}
    for day in days:
        latest = {}
        ever_positive = False
        for source in SOURCES:
            status, age, legacy = overrides.get((day, source), ('normal_no_data', 0, False))
            checked = (clock - pd.Timedelta(seconds=age)).isoformat()
            row = dict(target_date=day, source=source, market='land', started_at=checked,
                       checked_at=checked, status=status, sha256=(source + day).encode().hex()[:64],
                       response_summary={'parsed_status': status}, legacy_parser=legacy)
            if source == 'kpx_api' and status == 'failed':
                row.update(reason='RuntimeError', response_error='API HTTP 403')
            rows.append(row)
            latest[source] = {k: row.get(k) for k in ('source', 'status', 'checked_at', 'legacy_parser', 'sha256')}
            ever_positive |= status == 'official_acquired'
        ledger_days[day] = dict(ever_positive=ever_positive, latest=latest)
    return rows, dict(days=ledger_days)


class V20RetryTests(unittest.TestCase):
    def test_2300_warm_path_does_not_run_retry_refresh(self):
        clock = pd.Timestamp('2026-10-21T23:00:02+09:00')
        bound = pd.Timestamp('2026-10-21T23:00:00+09:00')
        with temporary_directory() as tmp, \
             patch.object(publisher, 'HOME', Path(tmp)), \
             patch.object(publisher, 'origin', return_value=bound), \
             patch.object(publisher.pd.Timestamp, 'now', return_value=clock), \
             patch.object(publisher, 'write_once', side_effect=lambda p, d: None), \
             patch.object(publisher, 'refresh_publication') as refresh, \
             patch.object(publisher, 'publish_once', return_value={'ok': True}) as publish:
            self.assertEqual(publisher.run(warm=True), {'ok': True})
            refresh.assert_not_called()
            publish.assert_called_once_with(publication_refresh=None)

    def test_2302_warm_retry_refreshes_then_publishes_once(self):
        clock = pd.Timestamp('2026-10-21T23:02:02+09:00')
        bound = pd.Timestamp('2026-10-21T23:00:00+09:00')
        audit = {'status': 'probe_completed', 'network_calls': 6}
        with temporary_directory() as tmp, \
             patch.object(publisher, 'HOME', Path(tmp)), \
             patch.object(publisher, 'origin', return_value=bound), \
             patch.object(publisher.pd.Timestamp, 'now', return_value=clock), \
             patch.object(publisher, 'write_once', side_effect=lambda p, d: None), \
             patch.object(publisher, 'refresh_publication', return_value=audit) as refresh, \
             patch.object(publisher, 'publish_once', return_value={'ok': True}) as publish:
            self.assertEqual(publisher.run(warm=True), {'ok': True})
            refresh.assert_called_once_with(bound)
            publish.assert_called_once_with(publication_refresh=audit)

    def test_retry_refresh_records_freshness_and_decision_reason(self):
        clock = pd.Timestamp('2026-10-21T23:02:01+09:00')
        cases = [
            ('fresh normal nonobservations', {}, True, 'fresh_sampled_negative_checks_not_universal_release_proof'),
            ('uncertain homepage zero', {('2026-10-23', 'general_public'): ('unverified_all_zero', 0, False)}, False, 'unverified_publication_general_public_unverified_all_zero'),
            ('KPX API error', {('2026-10-23', 'kpx_api'): ('failed', 0, False)}, False, 'unverified_publication_kpx_api_failed'),
            ('expired KPX sample', {('2026-10-23', 'kpx_api'): ('normal_no_data', 181, False)}, False, 'publication_check_not_fresh_kpx_api'),
            ('official SMP already observed', {('2026-10-23', 'kpx_api'): ('official_acquired', 0, False)}, False, 'official_price_already_observed'),
        ]
        with temporary_directory() as tmp, patch.object(publisher, 'HOME', Path(tmp)), patch.object(publisher, 'ROOT', Path(tmp)):
            for index, (label, overrides, expected_ok, expected_reason) in enumerate(cases):
                with self.subTest(label=label):
                    case_clock = clock + pd.Timedelta(seconds=index)
                    rows, ledger = refresh_fixture(case_clock, overrides)
                    probe = Mock(return_value=rows)
                    result = publisher.refresh_publication(
                        pd.Timestamp('2026-10-21T23:00:00+09:00'), probe_fn=probe,
                        ledger_fn=lambda value=ledger: value, clock_fn=lambda value=case_clock: value)
                    probe.assert_called_once_with(days=['2026-10-23', '2026-10-24'], force=True, budget_seconds=45)
                    decision = next(d for d in result['decisions'] if d['target_date'] == '2026-10-23')
                    self.assertEqual(decision['eligible'], expected_ok)
                    self.assertEqual(decision['reason'], expected_reason)
                    gate_ok, gate_reason = publisher.refresh_gate(result, '2026-10-23')
                    self.assertEqual(gate_ok, expected_ok)
                    if not expected_ok:
                        self.assertEqual(gate_reason, expected_reason)
                    kpx = next(x for x in decision['freshness'] if x['source'] == 'kpx_api')
                    self.assertEqual(kpx['fresh'], not (label == 'expired KPX sample'))
                    saved = Path(tmp) / result['record_file']
                    self.assertEqual(read(saved)['reference_origin'], '2026-10-21T23:00:00+09:00')
                    if label == 'KPX API error':
                        self.assertEqual(next(x for x in result['checks'] if x['source'] == 'kpx_api' and x['target_date'] == '2026-10-23')['response_error'], 'API HTTP 403')
            old_ledger_must_not_be_used = {'status': 'refresh_failed', 'decisions': [{'target_date': '2026-10-23', 'eligible': True}]}
            self.assertEqual(publisher.refresh_gate(old_ledger_must_not_be_used, '2026-10-23'), (False, 'publication_refresh_not_completed'))

    def test_v20_status_page_separates_preparation_and_issuance(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file('app_v20.py').run(timeout=90)
        self.assertFalse(app.exception)
        next(x for x in app.radio if x.label == '화면').set_value('연구 결과·시스템 상태').run(timeout=90)
        self.assertFalse(app.exception)
        self.assertTrue(any('최근 준비본' in m.label for m in app.metric))
        self.assertTrue(any('최근 발급 확인' in m.label for m in app.metric))

    def test_kpx_probe_records_safe_response_code_and_http_failure(self):
        import v18.publication as publication
        from v20.core import write_audit_once
        payload = {'response': {'header': {'resultCode': '00'},
                                'body': {'totalCount': '0', 'items': {'item': []}}}}
        with temporary_directory() as tmp, \
             patch.object(publication, 'HOME', Path(tmp)), \
             patch.object(publication, 'all_rows', return_value=[]), \
             patch.object(publication, 'write_once', side_effect=write_audit_once), \
             patch.object(publication, 'key', return_value='fake-private-key'), \
             patch.object(publication, 'call', return_value=payload), \
             patch.object(publication.urllib.request, 'urlopen', side_effect=RuntimeError('offline')), \
             patch.object(publication, 'epsis_request', side_effect=RuntimeError('offline')):
            rows = publication.probe(days=['2026-10-23'], force=True, budget_seconds=45)
            kpx = next(r for r in rows if r['source'] == 'kpx_api')
            self.assertEqual(kpx['status'], 'normal_no_data')
            self.assertEqual(kpx['response_summary']['api_result_code'], '00')
            self.assertEqual(kpx['response_summary']['api_item_count'], 0)
            self.assertTrue((Path(tmp) / 'publication/raw' / (kpx['sha256'] + '.raw')).exists())
            self.assertNotIn('fake-private-key', json.dumps(rows))
        with temporary_directory() as tmp, \
             patch.object(publication, 'HOME', Path(tmp)), \
             patch.object(publication, 'all_rows', return_value=[]), \
             patch.object(publication, 'write_once', side_effect=write_audit_once), \
             patch.object(publication, 'key', return_value='fake-private-key'), \
             patch.object(publication, 'call', side_effect=RuntimeError('API HTTP 403')), \
             patch.object(publication.urllib.request, 'urlopen', side_effect=RuntimeError('offline')), \
             patch.object(publication, 'epsis_request', side_effect=RuntimeError('offline')):
            rows = publication.probe(days=['2026-10-23'], force=True, budget_seconds=45)
            kpx = next(r for r in rows if r['source'] == 'kpx_api')
            self.assertEqual(kpx['status'], 'failed')
            self.assertEqual(kpx['response_error'], 'API HTTP 403')
            self.assertNotIn('fake-private-key', json.dumps(rows))


if __name__ == '__main__':
    unittest.main()
