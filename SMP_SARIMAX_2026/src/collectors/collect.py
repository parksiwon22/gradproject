from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from src.config import MONTHLY, RAW, external_dir, external_file
from src.excel_fallback import read_first_sheet

PERIOD_START, PERIOD_END = "2023-01-01", "2025-12-31"
FRED_SERIES = {
    "DCOILWTICO": ("WTI crude oil", "USD/barrel", "daily"),
    "DCOILBRENTEU": ("Brent crude oil", "USD/barrel", "daily"),
    "PNGASJPUSDM": ("Japan LNG import price", "USD/million Btu", "monthly"),
    "PCOALAUUSDM": ("Australia coal price", "USD/metric ton", "monthly"),
}


def _copy_sources(source: Path | None, label: str) -> list[dict]:
    records = []
    if source is None or not source.exists():
        return [{"dataset": label, "source": str(source or "not configured"), "status": "unavailable", "reason": "directory not configured or not found"}]
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            continue
        dest = RAW / f"{label}_{path.name}"
        if not dest.exists():
            shutil.copy2(path, dest)
        record = {"dataset": path.stem, "source": str(path), "copied_to": str(dest.relative_to(RAW.parent.parent)), "status": "copied_read_only", "bytes": path.stat().st_size}
        try:
            if path.suffix.lower() == ".xlsx":
                try: frame = pd.read_excel(path)
                except ImportError: frame = read_first_sheet(path)
            else:
                try: frame = pd.read_csv(path, encoding="utf-8-sig")
                except UnicodeDecodeError: frame = pd.read_csv(path, encoding="cp949")
            record["rows"] = int(len(frame))
            record["inspection_status"] = "readable"
            record["missing_rate"] = float(frame.isna().mean().mean()) if len(frame) and len(frame.columns) else None
            normalized = {re.sub(r"[^a-z0-9가-힣]", "", str(c).lower()): c for c in frame.columns}
            time_col = next((normalized[k] for k in normalized if k in {"timestamp", "datetime", "일시", "거래일시"}), None)
            date_col = next((normalized[k] for k in normalized if k in {"date", "tradedate", "tradeymd", "거래일자", "거래일", "기준일자", "일자", "기간", "날짜"}), None)
            hour_col = next((normalized[k] for k in normalized if k in {"hour", "tradeno", "거래시간", "거래시간대", "시간"}), None)
            wide_hours = [c for c in frame.columns if re.fullmatch(r"\s*\d{1,2}\s*시\s*", str(c))]
            stamps = None
            if time_col:
                stamps = pd.to_datetime(frame[time_col], errors="coerce")
            elif date_col:
                stamps = pd.to_datetime(frame[date_col].astype(str).str.replace(r"\.0$", "", regex=True), errors="coerce")
                if hour_col:
                    hour = pd.to_numeric(frame[hour_col], errors="coerce")
                    stamps = stamps + pd.to_timedelta(hour - 1, unit="h")
            if stamps is not None and stamps.notna().any():
                record["period_start"] = stamps.min().isoformat()
                record["period_end"] = stamps.max().isoformat()
                diffs = stamps.dropna().drop_duplicates().sort_values().diff().dropna()
                record["frequency"] = "hourly (wide date x 01–24 columns)" if wide_hours else (str(diffs.median()) if len(diffs) else "single/unknown")
            else:
                record["period_start"] = None; record["period_end"] = None; record["frequency"] = "unknown"
            smp_col = next((normalized[k] for k in normalized if k in {"smp", "육지smp", "계통한계가격", "육지계통한계가격"}), None)
            if smp_col:
                record["unit"] = "KRW/kWh (SMP)"
                record["smp_missing_rate"] = float(pd.to_numeric(frame[smp_col], errors="coerce").isna().mean()) if len(frame) else None
            elif wide_hours and "smp" in path.name.lower():
                record["unit"] = "KRW/kWh (hourly SMP)"
                record["smp_missing_rate"] = float(frame[wide_hours].apply(pd.to_numeric, errors="coerce").isna().mean().mean()) if len(frame) else None
            elif any("mwh" in str(c).lower() for c in frame.columns):
                record["unit"] = "MWh (source column labels)"
        except Exception as exc:
            record["inspection_status"] = "inspection_error"
            record["inspection_reason"] = type(exc).__name__
        records.append(record)
    if not records:
        records.append({"dataset": label, "source": str(source), "status": "unavailable", "reason": "no CSV/Excel files found"})
    return records


def collect_fred() -> list[dict]:
    key = os.getenv("FRED_API_KEY", "").strip()
    records = []
    if not key:
        return [{"dataset": sid, "source": "FRED", "status": "blocked", "reason": "FRED_API_KEY not set", "frequency": meta[2], "unit": meta[1], "series_page": f"https://fred.stlouisfed.org/series/{sid}"} for sid, meta in FRED_SERIES.items()]
    for sid, (name, unit, cadence) in FRED_SERIES.items():
        target_dir = MONTHLY if cadence == "monthly" else RAW
        target = target_dir / f"fred_{sid}_{PERIOD_START[:4]}_{PERIOD_END[:4]}.csv"
        if target.exists():
            df = pd.read_csv(target)
            status = "reused_cache"
        else:
            try:
                response = requests.get(
                    "https://api.stlouisfed.org/fred/series/observations",
                    params={"series_id": sid, "api_key": key, "file_type": "json", "observation_start": PERIOD_START, "observation_end": PERIOD_END}, timeout=(3, 10),
                )
                if response.status_code != 200:
                    records.append({"dataset": sid, "name": name, "source": "FRED API", "status": "http_error", "api_status": "not_confirmed", "http_status": response.status_code, "frequency": cadence, "unit": unit, "series_page": f"https://fred.stlouisfed.org/series/{sid}"})
                    continue
                payload = response.json()
                if "error_code" in payload:
                    records.append({"dataset": sid, "name": name, "source": "FRED API", "status": "api_error", "api_status": "api_error", "reason": payload.get("error_message", "FRED API error"), "frequency": cadence, "unit": unit, "series_page": f"https://fred.stlouisfed.org/series/{sid}"})
                    continue
                df = pd.DataFrame(payload.get("observations", []))
                if df.empty:
                    records.append({"dataset": sid, "name": name, "source": "FRED API", "status": "empty", "api_status": "success_empty", "reason": "No observations returned", "frequency": cadence, "unit": unit, "series_page": f"https://fred.stlouisfed.org/series/{sid}"})
                    continue
                df = df[["date", "value"]].rename(columns={"date": "observation_date", "value": sid})
                df[sid] = pd.to_numeric(df[sid].replace(".", pd.NA), errors="coerce")
                df.to_csv(target, index=False)
                status = "success"
            except requests.RequestException as exc:
                records.append({"dataset": sid, "name": name, "source": "FRED API", "status": "connection_error", "api_status": "unverified_network_unavailable", "reason": type(exc).__name__, "frequency": cadence, "unit": unit, "series_page": f"https://fred.stlouisfed.org/series/{sid}"})
                if isinstance(exc, requests.ConnectionError):
                    for remaining in list(FRED_SERIES)[list(FRED_SERIES).index(sid) + 1:]:
                        rem_name, rem_unit, rem_cadence = FRED_SERIES[remaining]
                        records.append({"dataset": remaining, "name": rem_name, "source": "FRED API", "status": "connection_error", "api_status": "unverified_network_unavailable", "reason": "network unavailable after first connection failure", "frequency": rem_cadence, "unit": rem_unit, "series_page": f"https://fred.stlouisfed.org/series/{remaining}"})
                    break
                continue
            except (ValueError, KeyError) as exc:
                records.append({"dataset": sid, "name": name, "source": "FRED API", "status": "response_error", "api_status": "invalid_response", "reason": type(exc).__name__, "frequency": cadence, "unit": unit, "series_page": f"https://fred.stlouisfed.org/series/{sid}"})
                continue
        dates = pd.to_datetime(df.get("observation_date", pd.Series(dtype=str)), errors="coerce")
        vals = pd.to_numeric(df.get(sid, pd.Series(dtype=float)), errors="coerce")
        records.append({"dataset": sid, "name": name, "source": "FRED API", "status": status, "api_status": "successful_call" if status == "success" else status, "official_series_metadata_confirmed": True, "series_page": f"https://fred.stlouisfed.org/series/{sid}", "period_start": dates.min().date().isoformat() if dates.notna().any() else None, "period_end": dates.max().date().isoformat() if dates.notna().any() else None, "rows": int(len(df)), "frequency": cadence, "unit": unit, "missing_rate": float(vals.isna().mean()) if len(vals) else None, "path": str(target.relative_to(target.parents[1]))})
    return records


def collect_kma_asos() -> list[dict]:
    """Fetch KMA's documented 31-day ASOS interval endpoint; store raw replies."""
    key = os.getenv("KMA_APIHUB_KEY", "").strip()
    if not key:
        return [{"dataset": "KMA ASOS hourly", "source": "KMA APIHub", "status": "blocked", "reason": "KMA_APIHUB_KEY not set"}]
    endpoint = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php"
    start, end = pd.Timestamp(PERIOD_START), pd.Timestamp(PERIOD_END)
    status_counts, raw_bodies, total_rows = [], [], 0
    current = start
    while current <= end:
        stop = min(current + pd.Timedelta(days=30, hours=23), end + pd.Timedelta(hours=23))
        target = RAW / f"kma_asos_108_{current:%Y%m%d}_{stop:%Y%m%d}.txt"
        if target.exists():
            body = target.read_text(encoding="utf-8", errors="replace")
            code = 200
            state = "reused_cache"
        else:
            try:
                response = requests.get(endpoint, params={"tm1": current.strftime("%Y%m%d%H%M"), "tm2": stop.strftime("%Y%m%d%H%M"), "stn": 108, "help": 1, "authKey": key}, timeout=(3, 20))
                code = response.status_code
                if code != 200:
                    status_counts.append({"period_start": current.date().isoformat(), "period_end": stop.date().isoformat(), "status": "http_error", "http_status": code})
                    current = stop + pd.Timedelta(hours=1); continue
                body = response.text
                if "인증" in body or "authKey" in body.lower() or "error" in body.lower()[:300]:
                    status_counts.append({"period_start": current.date().isoformat(), "period_end": stop.date().isoformat(), "status": "api_error", "reason": "KMA response indicates an authentication or service error"})
                    current = stop + pd.Timedelta(hours=1); continue
                target.write_text(body, encoding="utf-8")
                state = "success"
            except requests.RequestException as exc:
                status_counts.append({"period_start": current.date().isoformat(), "period_end": stop.date().isoformat(), "status": "connection_error", "reason": type(exc).__name__})
                if isinstance(exc, requests.ConnectionError):
                    break
                current = stop + pd.Timedelta(hours=1); continue
        raw_bodies.append(body)
        # KMA APIHub's text format includes comment/header lines followed by whitespace rows.
        lines = [line.split() for line in body.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        if lines and lines[0][0].upper() == "TM": lines = lines[1:]
        parsed = []
        for line in lines:
            if not line or not line[0][:8].isdigit(): continue
            if len(line) < 34: continue
            # Official field order in the APIHub guide: TM, STN, WD, WS, ..., TA, ..., SS, SI.
            parsed.append({"TM": line[0], "STN": line[1], "WS": line[3], "TA": line[11], "SS": line[32], "SI": line[33]})
        total_rows += len(parsed)
        status_counts.append({"period_start": current.date().isoformat(), "period_end": stop.date().isoformat(), "status": state, "http_status": code, "rows": len(parsed)})
        current = stop + pd.Timedelta(hours=1)
    if raw_bodies:
        obs_frames = []
        for body in raw_bodies:
            lines = [line.split() for line in body.splitlines() if line.strip() and not line.lstrip().startswith("#")]
            if lines and lines[0][0].upper() == "TM": lines = lines[1:]
            part = [{"TM": r[0], "STN": r[1], "WS": r[3], "TA": r[11], "SS": r[32], "SI": r[33]} for r in lines if r and r[0][:8].isdigit() and len(r) >= 34]
            if part: obs_frames.append(pd.DataFrame(part))
        if obs_frames:
            normalized = pd.concat(obs_frames, ignore_index=True)
            normalized["timestamp"] = pd.to_datetime(normalized.TM, format="%Y%m%d%H%M", errors="coerce")
            for field in ("WS", "TA", "SS", "SI"): normalized[field] = pd.to_numeric(normalized[field], errors="coerce")
            normalized = normalized.dropna(subset=["timestamp"])
            normalized.to_csv(RAW / "kma_asos_108_2023_2025.csv", index=False)
    if total_rows:
        overall = "success"
    elif any(x["status"] == "api_error" for x in status_counts):
        overall = "approval_or_api_error"
    elif any(x.get("http_status") == 429 for x in status_counts):
        overall = "rate_limited"
    elif any(x.get("http_status") in (401, 403) for x in status_counts):
        overall = "approval_or_auth_error"
    elif any(x["status"] == "connection_error" for x in status_counts):
        overall = "connection_error"
    elif any(x["status"] == "http_error" for x in status_counts):
        overall = "http_error"
    else:
        overall = "no_rows"
    first_failure = next((x for x in status_counts if x["status"] not in ("success", "reused_cache")), None)
    reason = f"First failed request: {first_failure['status']} ({first_failure.get('reason', first_failure.get('http_status', 'no rows'))})" if first_failure else None
    return [{"dataset": "KMA ASOS hourly at Seoul station 108", "source": "KMA APIHub kma_sfctm3.php (official 31-day max period)", "status": overall, "reason": reason, "period_start": PERIOD_START, "period_end": PERIOD_END, "rows": total_rows, "frequency": "hourly observations", "units": {"TA": "C", "SI": "MJ/m2", "WS": "m/s"}, "field_mapping": {"TA": "air temperature", "SI": "solar radiation", "WS": "wind speed"}, "window_results": status_counts, "note": "Observed weather is collected and archived only; it is excluded from forecast-time exogenous variables because future actuals are unavailable."}]


def _parse_kst(values: pd.Series) -> pd.Series:
    as_text = values.astype(str)
    has_offset = as_text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", case=False, regex=True).any()
    parsed = pd.to_datetime(values, errors="coerce", utc=has_offset)
    if has_offset:
        return parsed.dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    return parsed


def collect_existing_kpx_target() -> list[dict]:
    source = external_file("EXISTING_KPX_DATASET_FILE")
    official_source = external_file("EXISTING_KPX_OFFICIAL_FILE")
    if official_source and official_source.exists():
        official_copy = RAW / "existing_kpx_official_hourly.csv"
        if not official_copy.exists(): shutil.copy2(official_source, official_copy)
    if source is None or not source.exists():
        return [{"dataset": "Existing project land SMP actual series", "source": str(source or "not configured"), "status": "unavailable", "reason": "EXISTING_KPX_DATASET_FILE is not configured or does not exist"}]
    try:
        frame = pd.read_csv(source, usecols=lambda c: str(c).lower() in {"timestamp", "actual_smp", "smp"})
        time_col = next(c for c in frame.columns if str(c).lower() == "timestamp")
        value_col = next(c for c in frame.columns if str(c).lower() in {"actual_smp", "smp"})
        selected = pd.DataFrame({"timestamp": _parse_kst(frame[time_col]), "smp": pd.to_numeric(frame[value_col], errors="coerce")})
        selected = selected.dropna(subset=["timestamp"])
        selected = selected[(selected.timestamp >= PERIOD_START) & (selected.timestamp <= "2025-12-31 23:00:00")]
        dupe_counts = selected.groupby("timestamp").smp.nunique()
        if (dupe_counts > 1).any():
            return [{"dataset": "Existing project land SMP actual series", "source": str(source), "status": "conflicting_values", "reason": f"conflicting actual values at {int((dupe_counts > 1).sum())} timestamps"}]
        selected = selected.drop_duplicates("timestamp").sort_values("timestamp")
        if selected.smp.isna().any():
            return [{"dataset": "Existing project land SMP actual series", "source": str(source), "status": "missing_values", "reason": f"{int(selected.smp.isna().sum())} actual SMP values are missing"}]
        # Verify inherited values against independent read-only local target files.
        checks = []
        aligned = RAW / "existing_aligned_hourly_all_features_2023.csv"
        if aligned.exists():
            local = pd.read_csv(aligned, usecols=["timestamp", "smp"])
            local["timestamp"] = _parse_kst(local.timestamp)
            overlap = selected.merge(local, on="timestamp", suffixes=("_import", "_reference"))
            checks.append({"reference": "2023 read-only aligned source", "overlap_rows": int(len(overlap)), "exact_matches": int((overlap.smp_import == overlap.smp_reference).sum()), "matches": bool((overlap.smp_import == overlap.smp_reference).all())})
        official = RAW / "existing_kpx_official_hourly.csv"
        if official.exists():
            source_df = pd.read_csv(official, usecols=["timestamp", "smp"])
            source_df["timestamp"] = _parse_kst(source_df.timestamp)
            overlap = selected.merge(source_df, on="timestamp", suffixes=("_import", "_reference"))
            checks.append({"reference": "previously downloaded KPX official hourly source", "overlap_rows": int(len(overlap)), "exact_matches": int((overlap.smp_import == overlap.smp_reference).sum()), "matches": bool((overlap.smp_import == overlap.smp_reference).all())})
        if any(not check["matches"] for check in checks):
            return [{"dataset": "Existing project land SMP actual series", "source": str(source), "status": "overlap_validation_failed", "validation_checks": checks}]
        target = RAW / "existing_project_kpx_land_smp.csv"
        selected.to_csv(target, index=False)
        return [{"dataset": "Existing project land SMP actual series (actual_smp only)", "source": str(source), "copied_to": str(target.relative_to(RAW.parent.parent)), "status": "success", "period_start": selected.timestamp.min().isoformat(), "period_end": selected.timestamp.max().isoformat(), "rows": int(len(selected)), "frequency": "hourly", "unit": "KRW/kWh", "missing_rate": float(selected.smp.isna().mean()), "validation_checks": checks, "note": "Only timestamp and actual_smp were extracted; existing model features and predictions were not imported."}]
    except Exception as exc:
        return [{"dataset": "Existing project land SMP actual series", "source": str(source), "status": "inspection_error", "reason": type(exc).__name__}]


def collect() -> list[dict]:
    records = []
    records += _copy_sources(external_dir("EXISTING_SMP_DATA_DIR"), "existing")
    records += _copy_sources(external_dir("EPSIS_EXPORT_DIR"), "epsis")
    records += collect_existing_kpx_target()
    records += collect_fred()
    records += collect_kma_asos()
    # Official catalog confirms the base URLs but not operation paths/field schemas.
    # Keep these sources explicit; never construct a guessed request URL.
    records += [
        {"dataset": "KPX SMP and day-ahead demand", "source": "https://www.data.go.kr/data/15131225/openapi.do", "status": "manual_export_required", "reason": "Official page confirms service and interval semantics but omits operation path and schema; import EPSIS CSV/Excel until official detailed guide is available."},
        {"dataset": "KMA short-range forecast archive", "source": "https://apihub.kma.go.kr/apiList.do?seqApi=10", "status": "not_used", "reason": "Forecast API returns issue-time forecasts; historical vintage availability has not been verified, so it is excluded from historical model inputs."},
        {"dataset": "East-West day-ahead solar forecast", "source": "https://www.data.go.kr/data/15144541/openapi.do", "status": "credential_required", "reason": "Requires a separate data.go.kr credential, not supplied; historical forecast vintage availability also needs validation."},
    ]
    return records


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True); MONTHLY.mkdir(parents=True, exist_ok=True)
    records = collect()
    out = RAW / "collection_manifest.json"
    out.write_text(json.dumps({"collected_at": datetime.now().astimezone().isoformat(), "records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Collection manifest: {out}")
    for row in records:
        print(f"{row.get('dataset')}: {row.get('status')}")


if __name__ == "__main__":
    main()
