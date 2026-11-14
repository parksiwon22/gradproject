from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    # Keep basic local runs functional if python-dotenv has not been installed yet.
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MONTHLY = ROOT / "data" / "monthly"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"
START = "2023-01-01 00:00:00"
END = "2025-12-31 23:00:00"
TZ = "Asia/Seoul"


def external_dir(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def external_file(name: str) -> Path | None:
    return external_dir(name)
