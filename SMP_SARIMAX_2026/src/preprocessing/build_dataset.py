from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.config import END, PROCESSED, RAW, START, TZ
from src.excel_fallback import read_first_sheet

DATE_NAMES = {"date", "tradedate", "tradeYmd", "거래일자", "거래일", "기준일자", "일자", "기간", "날짜"}
HOUR_NAMES = {"hour", "tradeNo", "거래시간", "거래시간대", "시간", "예측시각"}
SMP_NAMES = {"smp", "육지smp", "계통한계가격", "육지계통한계가격", "가격"}


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path)
        except ImportError:
            if path.suffix.lower() == ".xlsx": return read_first_sheet(path)
            raise
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1")


def _canonical(value: object) -> str:
    return re.sub(r"[^a-z0-9가-힣]", "", str(value).lower())


def find_column(columns, candidates: set[str]) -> str | None:
    normalized = {_canonical(c): c for c in columns}
    for candidate in candidates:
        if _canonical(candidate) in normalized:
            return normalized[_canonical(candidate)]
    for norm, col in normalized.items():
        if any(_canonical(candidate) in norm for candidate in candidates):
            return col
    return None


def extract_smp(path: Path) -> pd.DataFrame:
    df = read_table(path)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "smp"])
    time_col = find_column(df.columns, {"timestamp", "datetime", "일시", "거래일시"})
    smp_col = find_column(df.columns, SMP_NAMES)
    if time_col and smp_col:
        ts = pd.to_datetime(df[time_col], errors="coerce")
        # Source workbooks may contain explicit UTC offsets. Otherwise timestamps are KST.
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert(TZ).dt.tz_localize(None)
        values = pd.to_numeric(df[smp_col].astype(str).str.replace(",", "", regex=False), errors="coerce")
        return pd.DataFrame({"timestamp": ts, "smp": values, "source_file": path.name}).dropna(subset=["timestamp"])
    date_col = find_column(df.columns, DATE_NAMES)
    hour_col = find_column(df.columns, HOUR_NAMES)
    wide_hours = [c for c in df.columns if re.fullmatch(r"\s*\d{1,2}\s*시\s*", str(c))]
    if date_col and wide_hours and ("smp" in path.name.lower() or any("smp" in str(c).lower() for c in df.columns)):
        day = pd.to_datetime(df[date_col], errors="coerce")
        rows = []
        for col in wide_hours:
            hour = int(re.search(r"\d+", str(col)).group(0))
            if not 1 <= hour <= 24: continue
            rows.append(pd.DataFrame({"timestamp": day + pd.Timedelta(hours=hour - 1), "smp": pd.to_numeric(df[col], errors="coerce"), "source_file": path.name}))
        return pd.concat(rows, ignore_index=True).dropna(subset=["timestamp"]) if rows else pd.DataFrame(columns=["timestamp", "smp", "source_file"])
    if date_col and hour_col and smp_col:
        region = find_column(df.columns, {"areaName", "지역", "구분", "지역구분"})
        if region:
            mask = df[region].astype(str).str.contains("육지|land", case=False, regex=True)
            if mask.any():
                df = df.loc[mask].copy()
        day = pd.to_datetime(df[date_col].astype(str).str.replace(r"\.0$", "", regex=True), errors="coerce")
        hour = pd.to_numeric(df[hour_col], errors="coerce")
        # KPX day-ahead dataset defines each hour as the end of its interval: 01 marks 00-01.
        stamp = day + pd.to_timedelta(hour - 1, unit="h")
        vals = pd.to_numeric(df[smp_col].astype(str).str.replace(",", "", regex=False), errors="coerce")
        return pd.DataFrame({"timestamp": stamp, "smp": vals, "source_file": path.name}).dropna(subset=["timestamp"])
    return pd.DataFrame(columns=["timestamp", "smp", "source_file"])


def build() -> pd.DataFrame:
    if not RAW.exists():
        raise FileNotFoundError(f"No raw data directory: {RAW}. Run `python -m src.cli collect` first.")
    chunks = []
    for path in RAW.rglob("*"):
        if path.suffix.lower() not in {".csv", ".xlsx", ".xls"} or path.name == "collection_manifest.json":
            continue
        try:
            data = extract_smp(path)
            if not data.empty:
                chunks.append(data)
        except Exception as exc:
            print(f"Skipped unreadable source {path.name}: {type(exc).__name__}")
    if not chunks:
        raise ValueError("No recognizable hourly SMP column found. Import an official EPSIS SMP export or a CSV with timestamp,smp columns.")
    joined = pd.concat(chunks, ignore_index=True)
    joined = joined.dropna(subset=["smp"])
    joined["timestamp"] = pd.to_datetime(joined["timestamp"], errors="coerce")
    joined = joined.dropna(subset=["timestamp"])
    joined = joined[(joined.timestamp >= START) & (joined.timestamp <= END)]
    # Prefer the most recently copied source when duplicate sources overlap. Record duplicates separately.
    joined = joined.sort_values(["timestamp", "source_file"] if "source_file" in joined else ["timestamp"])
    conflicts = joined.groupby("timestamp").smp.nunique()
    if (conflicts > 1).any():
        raise ValueError(f"Conflicting SMP values for {int((conflicts > 1).sum())} timestamps. Resolve source conflict before modeling.")
    joined = joined.drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index()
    index = pd.date_range(START, END, freq="h", tz=TZ).tz_localize(None)
    y = joined["smp"].reindex(index).rename("smp")
    out = pd.DataFrame({"timestamp": index, "smp": y.to_numpy()})
    out["is_observed"] = out.smp.notna()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROCESSED / "hourly_smp_2023_2025.csv", index=False)
    counts = out.groupby(out.timestamp.dt.year).smp.agg(rows="count", min="min", max="max")
    print(counts.to_string())
    print(f"Hourly rows={len(out)}, missing_smp={int(out.smp.isna().sum())} ({out.smp.isna().mean():.2%})")
    return out


if __name__ == "__main__":
    build()
