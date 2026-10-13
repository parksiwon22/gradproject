"""Run saved V5 models using only history before the target day."""
import argparse
from pathlib import Path
import pandas as pd
from model.v5 import DATA_PATH, generate_day_ahead_forecast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=Path("artifacts/forecast.csv"))
    args = parser.parse_args()
    raw = pd.read_csv(args.data)
    target = pd.Timestamp(args.date).normalize()
    history = raw[pd.to_datetime(raw["timestamp"]) < target].copy()
    forecast = generate_day_ahead_forecast(history, target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(forecast)} rows to {args.output}")


if __name__ == "__main__":
    main()
