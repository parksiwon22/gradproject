"""Credential-safe KMA API Hub weather vintage collector.

Forecasts and ASOS observations use separate endpoints and files. Only forecast
records issued and received by an SMP forecast origin are eligible model inputs.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

API_BASE = "https://apihub.kma.go.kr/api/"
FORECAST_PATH = "typ02/openApi/VilageFcstInfoService_2.0/getVilageFcst"
ASOS_PATH = "typ01/url/kma_sfctm3.php"
ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("KMA_WEATHER_DATA_DIR", ROOT / "automation/state/weather"))
FORECAST_OUTPUT = DATA_ROOT / "forecast_vintages.jsonl"
ASOS_OUTPUT = DATA_ROOT / "asos_observations.jsonl"
KST = ZoneInfo("Asia/Seoul")

# Geographically spread mainland sample. It is not asserted to be a load-
# weighted national representative; the station IDs are provenance references.
LAND_GRIDS = {
    "Gangneung": {"nx": 92, "ny": 131, "asos_station": 105},
    "Seoul": {"nx": 60, "ny": 127, "asos_station": 108},
    "Daejeon": {"nx": 67, "ny": 100, "asos_station": 133},
    "Daegu": {"nx": 89, "ny": 90, "asos_station": 143},
    "Gwangju": {"nx": 58, "ny": 74, "asos_station": 156},
    "Busan": {"nx": 98, "ny": 76, "asos_station": 159},
}


def _service_key(env: dict[str, str] | None = None) -> str | None:
    values = os.environ if env is None else env
    value = values.get("KMA_APIHUB_AUTH_KEY")
    return value.strip() if value and value.strip() else None


def _classify_http_error(exc: urllib.error.HTTPError) -> str:
    # Inspect only transiently to distinguish authorization from transport;
    # never serialize provider error text, which may echo request information.
    body = exc.read(4096).decode("utf-8", errors="ignore").lower()
    if exc.code == 401 or ("key" in body and any(x in body for x in ("invalid", "인증"))):
        return "authentication_rejected"
    if exc.code == 403:
        if any(word in body for word in ("신청", "승인", "application", "permission", "서비스")):
            return "service_application_or_approval"
        return "forbidden_service_or_account_scope"
    if exc.code == 429 or "limit" in body or "초과" in body:
        return "service_limit"
    if exc.code >= 500:
        return "upstream_service_failure"
    if exc.code == 400:
        return "request_parameter_rejected"
    return f"http_{exc.code}"


def _request(path: str, params: dict, key: str,
             opener: Callable = urllib.request.urlopen) -> bytes:
    query = urllib.parse.urlencode({**params, "authKey": key})
    request = urllib.request.Request(API_BASE + path + "?" + query,
                                     headers={"User-Agent": "smp-research-weather/1.0"})
    try:
        with opener(request, timeout=45) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(_classify_http_error(exc)) from None
    except Exception as exc:
        raise RuntimeError("transport_" + type(exc).__name__) from None
    # Do not persist a provider response if it unexpectedly echoes credentials.
    if key.encode() in payload or urllib.parse.quote(key).encode() in payload:
        raise RuntimeError("credential_echo_body_suppressed")
    return payload


def parse_payload(raw: bytes, *, location: str, grid: dict, issued_at_kst: str,
                  received_at_utc: str, forecast_origin_kst: str) -> list[dict]:
    """Normalize API Hub's compatible JSON forecast response, preserving vintage."""
    body = json.loads(raw.decode("utf-8"))
    header = body.get("response", {}).get("header", {})
    if str(header.get("resultCode", "00")) not in ("00", "0"):
        raise ValueError("api_result_" + str(header.get("resultCode", "unknown")))
    items = body.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    records = []
    for item in items:
        target_time = str(item["fcstTime"])
        records.append({
            "record_type": "forecast",
            "source": "KMA API Hub typ02/openApi/VilageFcstInfoService_2.0/getVilageFcst",
            "location": location, "grid_nx": int(grid["nx"]), "grid_ny": int(grid["ny"]),
            "asos_station_reference": int(grid["asos_station"]),
            "issued_at_kst": issued_at_kst, "received_at_utc": received_at_utc,
            "forecast_origin_kst": forecast_origin_kst,
            "target_at_kst": f"{item['fcstDate']}T{target_time[:2]}:{target_time[2:]}:00+09:00",
            "category": str(item["category"]), "value": str(item["fcstValue"]),
            "target_date": str(item["fcstDate"]), "target_time": target_time,
        })
    if not records:
        raise ValueError("empty_forecast_response")
    return records


def _append_jsonl(output: Path, records: list[dict]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def collect(base_date: str, base_time: str, forecast_origin_kst: str,
            output: Path = FORECAST_OUTPUT, env: dict[str, str] | None = None,
            opener: Callable = urllib.request.urlopen) -> dict:
    key = _service_key(env)
    if not key:
        return {"status": "deferred", "reason": "KMA_APIHUB_AUTH_KEY is not configured",
                "records_written": 0}
    issued = f"{base_date}T{base_time[:2]}:{base_time[2:]}:00+09:00"
    origin = datetime.fromisoformat(forecast_origin_kst.replace("Z", "+00:00"))
    issue_dt = datetime.fromisoformat(issued)
    if issue_dt > origin:
        return {"status": "deferred", "reason": "forecast vintage issued after forecast origin",
                "records_written": 0}
    records, received_times = [], []
    for location, grid in LAND_GRIDS.items():
        params = {"pageNo": 1, "numOfRows": 1000, "dataType": "JSON",
                  "base_date": base_date, "base_time": base_time,
                  "nx": grid["nx"], "ny": grid["ny"]}
        try:
            raw = _request(FORECAST_PATH, params, key, opener)
            received = datetime.now(timezone.utc).isoformat(timespec="seconds")
            received_times.append(received)
            records.extend(parse_payload(raw, location=location, grid=grid,
                issued_at_kst=issued, received_at_utc=received,
                forecast_origin_kst=forecast_origin_kst))
        except Exception as exc:
            return {"status": "failed", "reason": str(exc) if isinstance(exc, RuntimeError)
                    else type(exc).__name__, "records_written": 0,
                    "failed_location": location, "request_count": len(received_times) + 1}
    _append_jsonl(output, records)
    return {"status": "collected", "records_written": len(records),
            "locations": len(LAND_GRIDS), "request_count": len(LAND_GRIDS),
            "first_received_at_utc": min(received_times),
            "last_received_at_utc": max(received_times)}


def _parse_number(raw: str, missing: tuple[float, ...]) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return None if value in missing else value


def parse_asos_payload(raw: bytes, *, received_at_utc: str) -> list[dict]:
    """Parse API Hub's fixed-width/tokenized ASOS period response."""
    out = []
    for line in raw.decode("cp949", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 16 or len(fields[0]) != 12 or not fields[0].isdigit():
            continue
        try:
            station_id = int(fields[1])
            if station_id not in {g["asos_station"] for g in LAND_GRIDS.values()}:
                continue
            row = {
                "record_type": "observation",
                "source": "KMA API Hub typ01/url/kma_sfctm3.php",
                "station_id": station_id,
                "station_name": next(k for k, v in LAND_GRIDS.items()
                                      if v["asos_station"] == station_id),
                "observed_at_kst": f"{fields[0][:4]}-{fields[0][4:6]}-{fields[0][6:8]}T{fields[0][8:10]}:{fields[0][10:12]}:00+09:00",
                "received_at_utc": received_at_utc,
                "wind_speed_ms": _parse_number(fields[3], (-9, -99, -999)),
                "temperature_c": _parse_number(fields[11], (-99, -999)),
                "humidity_percent": _parse_number(fields[13], (-9, -99, -999)),
                "rainfall_mm": _parse_number(fields[15], (-9, -99, -999)),
            }
            out.append(row)
        except (IndexError, ValueError):
            continue
    return out


def collect_asos_hourly(start_date: str, end_date: str, *, output: Path = ASOS_OUTPUT,
                        env: dict[str, str] | None = None,
                        opener: Callable = urllib.request.urlopen) -> dict:
    """Collect ASOS observations as observations, never as forecast vintages."""
    key = _service_key(env)
    if not key:
        return {"status": "deferred", "reason": "KMA_APIHUB_AUTH_KEY is not configured",
                "records_written": 0}
    station_ids = ":".join(str(v["asos_station"]) for v in LAND_GRIDS.values())
    params = {"tm1": start_date.replace("-", "") + "0000",
              "tm2": end_date.replace("-", "") + "2300",
              "stn": station_ids, "help": 0}
    try:
        raw = _request(ASOS_PATH, params, key, opener)
        received = datetime.now(timezone.utc).isoformat(timespec="seconds")
        records = parse_asos_payload(raw, received_at_utc=received)
        if not records:
            return {"status": "failed", "reason": "empty_asos_response", "records_written": 0}
    except Exception as exc:
        return {"status": "failed", "reason": str(exc) if isinstance(exc, RuntimeError)
                else type(exc).__name__, "records_written": 0, "request_count": 1}
    _append_jsonl(output, records)
    return {"status": "collected", "records_written": len(records),
            "stations": len({row["station_id"] for row in records}), "request_count": 1,
            "received_at_utc": received}


def eligible_vintages(records: list[dict], origin_kst: str) -> list[dict]:
    """Select only forecast vintages issued and received before the origin."""
    origin = datetime.fromisoformat(origin_kst.replace("Z", "+00:00"))
    eligible = []
    for row in records:
        if row.get("record_type") != "forecast":
            continue
        received = datetime.fromisoformat(row["received_at_utc"].replace("Z", "+00:00"))
        issued = datetime.fromisoformat(row["issued_at_kst"].replace("Z", "+00:00"))
        if received <= origin and issued <= origin:
            eligible.append(row)
    return eligible


def _latest_issue(origin: datetime) -> tuple[str, str] | None:
    """Return latest standard short-forecast issue with a 10-minute release buffer."""
    issue_times = ("0200", "0500", "0800", "1100", "1400", "1700", "2000", "2300")
    safe_origin = origin.astimezone(KST) - timedelta(minutes=10)
    candidates = []
    for offset in range(2):
        day = safe_origin.date() - timedelta(days=offset)
        for hhmm in issue_times:
            issue = datetime.strptime(day.strftime("%Y%m%d") + hhmm, "%Y%m%d%H%M").replace(tzinfo=KST)
            if issue <= safe_origin:
                candidates.append(issue)
    if not candidates:
        return None
    issue = max(candidates)
    return issue.strftime("%Y%m%d"), issue.strftime("%H%M")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("forecast", "asos", "both"), default="both")
    parser.add_argument("--origin-kst", default=None)
    args = parser.parse_args()
    now = datetime.now(KST)
    origin = datetime.fromisoformat(args.origin_kst).replace(tzinfo=KST) if args.origin_kst else now
    result = {}
    if args.mode in ("forecast", "both"):
        issue = _latest_issue(origin)
        if issue is None:
            result["forecast"] = {"status": "deferred", "reason": "no released issue vintage before origin"}
        else:
            result["forecast"] = collect(*issue, origin.isoformat(), FORECAST_OUTPUT)
    if args.mode in ("asos", "both"):
        obs_day = (now.date() - timedelta(days=1)).isoformat()
        result["asos"] = collect_asos_hourly(obs_day, obs_day)
    # Results contain statuses/counts only; never print response bodies or keys.
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if all(v.get("status") in ("collected", "deferred") for v in result.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
