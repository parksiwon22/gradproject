import json
import tempfile
import unittest
from pathlib import Path
from io import BytesIO
from urllib.error import HTTPError

from automation.smp_auto.collect_weather import (
    API_BASE, ASOS_PATH, FORECAST_PATH, LAND_GRIDS, collect,
    collect_asos_hourly, eligible_vintages, parse_asos_payload, parse_payload,
    _request,
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
        self.assertIn("KMA API Hub", row["source"])

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
        row = parse_asos_payload(raw, received_at_utc="2026-11-13T13:01:00+00:00")
        # JSON is intentionally not accepted for this API Hub endpoint: ASOS
        # Hourly observations use the fixed-width response format.
        self.assertEqual(row, [])

    def test_asos_api_hub_response_is_stored_as_observation(self):
        # Minimal tokenized row matching the API Hub response's columns used by
        # the V11 parser; no provider payload or credential is logged.
        cols = ["202611132200", "108", "180", "1.8", "0", "0", "0", "0", "0", "0", "0", "4.1", "0", "61", "0", "0"] + ["0"] * 24
        row = parse_asos_payload((" ".join(cols) + "\n").encode("cp949"),
                                 received_at_utc="2026-11-13T13:01:00+00:00")[0]
        self.assertEqual(row["record_type"], "observation")
        self.assertIn("observed_at_kst", row)
        self.assertNotIn("issued_at_kst", row)
        self.assertEqual(row["temperature_c"], 4.1)
        self.assertEqual(row["humidity_percent"], 61)

    def test_no_key_safely_defers_without_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weather.jsonl"
            result = collect("20261113", "2000", "2026-11-13T23:00:00+09:00",
                             path, env={})
            self.assertEqual(result["status"], "deferred")
            self.assertFalse(path.exists())

    def test_forecast_service_403_is_classified_without_echoing_key_or_body(self):
        secret = "hidden-test-credential"

        def denied(request, timeout=30):
            raise HTTPError(request.full_url, 403, "Forbidden", {},
                            BytesIO("서비스 신청 및 승인이 필요합니다".encode("utf-8")))

        with self.assertRaises(RuntimeError) as caught:
            _request(FORECAST_PATH, {"nx": 60, "ny": 127}, secret, denied)
        self.assertEqual(str(caught.exception), "service_application_or_approval")
        self.assertNotIn(secret, str(caught.exception))

    def test_mocked_api_hub_forecast_collection_writes_six_vintaged_locations(self):
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
            self.assertTrue(request.full_url.startswith(API_BASE + FORECAST_PATH + "?"))
            self.assertIn("authKey=test-secret", request.full_url)
            return Response()

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weather.jsonl"
            result = collect("20261130", "2000", "2026-12-01T00:00:00+09:00",
                             path, env={"KMA_APIHUB_AUTH_KEY": "test-secret"}, opener=opener)
            self.assertEqual(result["status"], "collected")
            self.assertEqual(result["request_count"], 6)
            self.assertEqual(len(calls), 6)
            self.assertNotIn("test-secret", path.read_text(encoding="utf-8"))
            rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(r["record_type"] == "forecast" for r in rows))
            self.assertTrue(all(r["received_at_utc"] for r in rows))
            self.assertTrue(all("data.go.kr" not in r["source"] for r in rows))

    def test_mocked_asos_uses_api_hub_and_separate_observation_file(self):
        cols = ["202611132200", "108", "180", "1.8", "0", "0", "0", "0", "0", "0", "0", "4.1", "0", "61", "0", "0"] + ["0"] * 24
        payload = (" ".join(cols) + "\n").encode("cp949")
        calls = []

        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return payload

        def opener(request, timeout=30):
            calls.append(request.full_url)
            self.assertTrue(request.full_url.startswith(API_BASE + ASOS_PATH + "?"))
            self.assertIn("authKey=test-secret", request.full_url)
            self.assertIn("stn=105%3A108%3A133%3A143%3A156%3A159", request.full_url)
            return Response()

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "observations.jsonl"
            result = collect_asos_hourly("2026-11-13", "2026-11-13", output=path,
                env={"KMA_APIHUB_AUTH_KEY": "test-secret"}, opener=opener)
            self.assertEqual(result["status"], "collected")
            self.assertEqual(result["request_count"], 1)
            self.assertEqual(len(calls), 1)
            content = path.read_text(encoding="utf-8")
            self.assertNotIn("test-secret", content)
            self.assertIn('"record_type":"observation"', content)


if __name__ == "__main__":
    unittest.main()
