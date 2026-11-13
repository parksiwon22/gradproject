"""Credential-safe KMA short-range forecast vintage collector.

Only forecasts received before the issue origin are eligible model inputs.
ASOS observations are deliberately handled as a different data type and are
never substituted for historical forecast vintages.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

KMA_FORECAST_URL = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
KMA_ASOS_URL = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
OUTPUT = Path(__file__).resolve().parents[2] / "docs/live/weather/forecast_vintages.jsonl"
# A geographically spread, land-only exploratory set. These are not claimed
# to be load-weighted or official national representative stations.
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
    return values.get("KMA_DATA_GO_KR_SERVICE_KEY") or values.get("KMA_SERVICE_KEY")


def _request(params: dict, key: str, opener: Callable = urllib.request.urlopen) -> bytes:
    query = urllib.parse.urlencode({**params, "serviceKey": key})
    request = urllib.request.Request(KMA_FORECAST_URL + "?" + query,
                                     headers={"User-Agent": "smp-research-weather/1.0"})
    try:
        with opener(request, timeout=30) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        # Never serialize the URL or provider response: either can echo a key.
        raise RuntimeError(f"KMA forecast HTTP {exc.code}; response suppressed") from None


def parse_payload(raw: bytes, *, location: str, grid: dict, issued_at_kst: str,
                  received_at_utc: str, forecast_origin_kst: str) -> list[dict]:
    """Normalize official API JSON while retaining issue and valid times."""
    body = json.loads(raw.decode("utf-8"))
    header = body.get("response", {}).get("header", {})
    if str(header.get("resultCode", "00")) not in ("00", "0"):
        raise ValueError("KMA forecast API returned a non-success result")
    items = body.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    records = []
    for item in items:
        records.append({
            "record_type": "forecast",
            "source": "KMA data.go.kr VilageFcstInfoService_2.0/getVilageFcst",
            "location": location, "grid_nx": int(grid["nx"]), "grid_ny": int(grid["ny"]),
            "asos_station_reference": int(grid["asos_station"]),
            "issued_at_kst": issued_at_kst, "received_at_utc": received_at_utc,
            "forecast_origin_kst": forecast_origin_kst,
            "target_at_kst": f"{item['fcstDate']}T{item['fcstTime'][:2]}:{item['fcstTime'][2:]}:00+09:00",
            "category": str(item["category"]), "value": str(item["fcstValue"]),
            "target_date": str(item["fcstDate"]), "target_time": str(item["fcstTime"]),
        })
    return records


def collect(base_date: str, base_time: str, forecast_origin_kst: str,
            output: Path = OUTPUT, env: dict[str, str] | None = None,
            opener: Callable = urllib.request.urlopen) -> dict:
    key = _service_key(env)
    if not key:
        return {"status": "deferred", "reason": "KMA data.go.kr service key is not configured",
                "records_written": 0}
    issued = f"{base_date}T{base_time[:2]}:{base_time[2:]}:00+09:00"
    origin = datetime.fromisoformat(forecast_origin_kst.replace("Z", "+00:00"))
    issue_dt = datetime.fromisoformat(issued)
    if issue_dt > origin:
        return {"status": "deferred", "reason": "forecast vintage was issued after forecast origin",
                "records_written": 0}
    records = []
    received_times = []
    # One request per regional grid for one issuance vintage; six requests per
    # capture. The service-specific account quota must be checked after access approval.
    for location, grid in LAND_GRIDS.items():
        params = {"pageNo": 1, "numOfRows": 1000, "dataType": "JSON",
                  "base_date": base_date, "base_time": base_time,
                  "nx": grid["nx"], "ny": grid["ny"]}
        try:
            raw = _request(params, key, opener)
            received = datetime.now(timezone.utc).isoformat(timespec="seconds")
            received_times.append(received)
            records.extend(parse_payload(raw, location=location, grid=grid,
                issued_at_kst=issued, received_at_utc=received,
                forecast_origin_kst=forecast_origin_kst))
        except Exception as exc:
            # Do not store the URL, key, or exception text from the HTTP stack.
            return {"status": "failed", "reason": type(exc).__name__, "records_written": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {"status": "collected", "records_written": len(records),
            "locations": len(LAND_GRIDS), "request_count": len(LAND_GRIDS),
            "first_received_at_utc": min(received_times),
            "last_received_at_utc": max(received_times)}


def parse_asos_payload(raw: bytes, *, station_id: int, station_name: str,
                       received_at_utc: str) -> list[dict]:
    """Parse ASOS hourly observations into a distinct observation schema."""
    body = json.loads(raw.decode("utf-8"))
    header = body.get("response", {}).get("header", {})
    if str(header.get("resultCode", "00")) not in ("00", "0"):
        raise ValueError("KMA ASOS API returned a non-success result")
    items = body.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    out = []
    for item in items:
        out.append({
            "record_type": "observation", "source": "KMA data.go.kr AsosHourlyInfoService/getWthrDataList",
            "station_id": int(station_id), "station_name": station_name,
            "observed_at_kst": f"{item['tm']}:00+09:00", "received_at_utc": received_at_utc,
            "temperature_c": item.get("ta"), "humidity_percent": item.get("hm"),
            "wind_speed_ms": item.get("ws"), "rainfall_mm": item.get("rn"),
        })
    return out


def collect_asos_hourly(start_date: str, end_date: str, *, station_id: int,
                        station_name: str, output: Path | None = None,
                        env: dict[str, str] | None = None,
                        opener: Callable = urllib.request.urlopen) -> dict:
    """Fetch observed ASOS rows separately; never return these as a forecast."""
    key = _service_key(env)
    if not key:
        return {"status": "deferred", "reason": "KMA data.go.kr service key is not configured",
                "records_written": 0}
    params = {"dataType": "JSON", "dataCd": "ASOS", "dateCd": "HR",
              "startDt": start_date, "startHh": "00", "endDt": end_date,
              "endHh": "23", "stnIds": int(station_id), "pageNo": 1, "numOfRows": 999}
    try:
        raw = _request_asos(params, key, opener)
        rows = parse_asos_payload(raw, station_id=station_id, station_name=station_name,
                                  received_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    except Exception as exc:
        return {"status": "failed", "reason": type(exc).__name__, "records_written": 0}
    target = output or OUTPUT.with_name("asos_observations.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {"status": "collected", "records_written": len(rows), "request_count": 1}


def _request_asos(params: dict, key: str, opener: Callable = urllib.request.urlopen) -> bytes:
    query = urllib.parse.urlencode({**params, "serviceKey": key})
    request = urllib.request.Request(KMA_ASOS_URL + "?" + query,
                                     headers={"User-Agent": "smp-research-weather/1.0"})
    try:
        with opener(request, timeout=30) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"KMA ASOS HTTP {exc.code}; response suppressed") from None


def eligible_vintages(records: list[dict], origin_kst: str) -> list[dict]:
    """Select only weather forecast vintages received before issue origin."""
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


if __name__ == "__main__":
    now = datetime.now().astimezone()
    result = collect(now.strftime("%Y%m%d"), "0200", now.isoformat())
    print(json.dumps(result, ensure_ascii=False))
