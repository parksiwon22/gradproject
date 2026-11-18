"""Render a saved 72-hour forecast for a user-selected issue time."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "SMP_XGBOOST_2026"
sys.path.insert(0, str(ROOT))

from SMP_XGBOOST_2026.src.models.train_xgboost_72h import make_72h_plot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-version", default="smp72h_common_v5_generation_v2_xgb_20261119")
    parser.add_argument("--forecast-origin", required=True, help="Issue time, for example: 2025-10-22 23:00:00")
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    prediction_path = args.predictions or (PROJECT / "outputs" / "predictions" / "72h" / args.dataset_version / "predictions_72h.csv")
    if not prediction_path.exists():
        raise FileNotFoundError(f"Saved predictions not found: {prediction_path}")
    wanted = pd.Timestamp(args.forecast_origin)
    data = pd.read_csv(prediction_path)
    data["forecast_origin"] = pd.to_datetime(data["forecast_origin"], errors="coerce")
    data["target_timestamp"] = pd.to_datetime(data["target_timestamp"], errors="coerce")
    sample = data[data["forecast_origin"] == wanted].sort_values("target_timestamp")
    if len(sample) != 72:
        available = data.loc[data["forecast_origin"] == wanted, "split"].unique().tolist()
        raise ValueError(f"Expected 72 saved forecast rows for {wanted}, found {len(sample)} (splits={available})")

    output_dir = args.output_dir or (PROJECT / "outputs" / "figures" / "72h" / "custom" / args.dataset_version)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sample_72h_{wanted:%Y%m%d_%H}"
    expected = [output_dir / f"{stem}.png", output_dir / f"{stem}.html"]
    existing = [str(path) for path in expected if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite selected-forecast graph(s): " + ", ".join(existing))
    make_72h_plot(sample, output_dir, str(wanted))
    print(f"png={expected[0]}")
    print(f"html={expected[1]}")


if __name__ == "__main__":
    main()
