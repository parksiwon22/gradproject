import json
import tempfile
import unittest
from pathlib import Path

from automation.smp_auto.collect_weather import (
    LAND_GRIDS, collect, eligible_vintages, parse_asos_payload, parse_payload,
)


class WeatherVintageTests(unittest.TestCase):
    def test_parse_keeps_issue_receipt_target_and_source(self):
        raw = json.dumps({"response": {"header": {"resultCode": "00"}, "body": {
            "items": {"item": [{"fcstDate": "20261114", "fcstTime": "0300",
                                 "category": "TMP", "fcstValue": "8.2"}]}}}}).encode()
        row = parse_payload(raw, location="Seoul", grid=LAND_GRIDS["Seoul"],
            issued_at_kst="2026-11-13T20:00:00+09:00",
            received_at_utc="2026-11-13T11:03:00+00:00",
            forecast_origin_kst="2026-11-13T23:00:00+09:00")[0]
        self.assertEqual(row["record_type"], "forecast")
        self.assertEqual(row["target_at_kst"], "20261114T03:00:00+09:00")
        self.assertIn("KMA data.go.kr", row["source"])

    def test_observations_and_late_vintages_are_not_eligible(self):
        rows = [
            {"record_type": "forecast", "issued_at_kst": "2026-11-13T20:00:00+09:00",
             "received_at_utc": "2026-11-13T11:02:00+00:00"},
            {"record_type": "forecast", "issued_at_kst": "2026-11-13T23:00:00+09:00",
             "received_at_utc": "2026-11-13T14:01:00+00:00"},
            {"record_type": "observation", "issued_at_kst": "2026-11-13T20:00:00+09:00",
             "received_at_utc": "2026-11-13T11:02:00+00:00"},
        ]
        eligible = eligible_vintages(rows, "2026-11-13T23:00:00+09:00")
        self.assertEqual(eligible, rows[:1])

    def test_asos_is_stored_as_observation_not_forecast(self):
        raw = json.dumps({"response": {"header": {"resultCode": "00"}, "body": {
            "items": {"item": [{"tm": "2026-11-13 22:00", "ta": "4.1",
                                 "hm": "61", "ws": "1.8", "rn": "0"}]}}}}).encode()
        row = parse_asos_payload(raw, station_id=108, station_name="Seoul",
                                 received_at_utc="2026-11-13T13:01:00+00:00")[0]
        self.assertEqual(row["record_type"], "observation")
        self.assertIn("observed_at_kst", row)
        self.assertNotIn("issued_at_kst", row)

    def test_no_key_safely_defers_without_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weather.jsonl"
            result = collect("20261113", "2000", "2026-11-13T23:00:00+09:00",
                             path, env={})
            self.assertEqual(result["status"], "deferred")
            self.assertFalse(path.exists())

    def test_mocked_forecast_collection_writes_six_vintaged_locations_without_key(self):
        payload = json.dumps({"response": {"header": {"resultCode": "00"}, "body": {
            "items": {"item": [{"fcstDate": "20261201", "fcstTime": "0300",
                                 "category": "TMP", "fcstValue": "8.2"}]}}}}).encode()
        calls = []

        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return payload

        def opener(request, timeout=30):
            calls.append(request.full_url)
            return Response()

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weather.jsonl"
            result = collect("20261130", "2000", "2026-12-01T00:00:00+09:00",
                             path, env={"KMA_SERVICE_KEY": "test-secret"}, opener=opener)
            self.assertEqual(result["status"], "collected")
            self.assertEqual(result["request_count"], 6)
            self.assertEqual(len(calls), 6)
            self.assertNotIn("test-secret", path.read_text(encoding="utf-8"))
            rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(r["record_type"] == "forecast" for r in rows))
            self.assertTrue(all(r["received_at_utc"] for r in rows))


if __name__ == "__main__":
    unittest.main()
