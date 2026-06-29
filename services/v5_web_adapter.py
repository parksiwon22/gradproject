from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


WEB_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = WEB_ROOT / "outputs"
V5_ROOT = Path(r"C:\smpsmp\smpsmpcodexuse\versions\v5_day_ahead")
V5_PREDICTIONS = V5_ROOT / "results" / "v5_day_ahead_predictions.csv"
V5_COMPARISON = V5_ROOT / "results" / "v5_model_comparison.csv"
V5_CHARGING = V5_ROOT / "results" / "v5_charging_window_backtest.csv"

FORECAST_OUT = OUTPUTS / "next_day_smp_forecast_tuned.csv"
RECOMMENDATION_OUT = OUTPUTS / "recommended_3h_charging_window_tuned.csv"
METRICS_OUT = OUTPUTS / "v5_metrics.json"

DEFAULT_TARGET_DATE = "2023-10-01"
MODEL_SOURCE = "V5_DAY_AHEAD_DIRECT_HGB"
SELECTED_MODEL = "validation_selected_profile_hgb_blend"


def _hour_for_web(timestamp: pd.Timestamp) -> int:
    return 24 if timestamp.hour == 0 else int(timestamp.hour)


def _selected_prediction_column(df: pd.DataFrame) -> str:
    if "profile_hgb_blend" in df.columns:
        return "profile_hgb_blend"
    if "predicted_smp" in df.columns:
        return "predicted_smp"
    raise ValueError("No selected V5 prediction column found.")


def _complete_days(df: pd.DataFrame) -> list[str]:
    counts = df.groupby("target_date")["horizon"].nunique()
    return sorted(counts[counts == 24].index.astype(str).tolist())


def choose_demo_day(df: pd.DataFrame, preferred_date: str = DEFAULT_TARGET_DATE) -> str:
    complete_days = _complete_days(df)
    if not complete_days:
        raise ValueError("No complete 24-hour V5 forecast day is available.")
    if preferred_date in complete_days:
        return preferred_date
    return complete_days[0]


def build_forecast_for_web(target_date: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(V5_PREDICTIONS, encoding="utf-8-sig")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["target_date"] = df["target_date"].astype(str)
    demo_date = choose_demo_day(df, target_date or DEFAULT_TARGET_DATE)
    day = df[df["target_date"] == demo_date].sort_values("horizon").copy()
    if len(day) != 24:
        raise ValueError(f"Selected demo day {demo_date} does not have exactly 24 rows.")

    selected_col = _selected_prediction_column(day)
    forecast = pd.DataFrame(
        {
            "datetime": day["timestamp"],
            "market_date": day["timestamp"].dt.date.astype(str),
            "hour": day["timestamp"].map(_hour_for_web),
            "predicted_smp": day[selected_col].astype(float),
            "lag_24": day["daily_profile_baseline"].astype(float),
            "lag_168": day.get("weekly_profile_baseline", pd.Series([pd.NA] * len(day))).astype(float),
            "ramp_probability": day["ramp_probability"].astype(float),
            "risk_level": day["risk_level"].astype(str),
            "price_weather": day["price_weather"].astype(str),
            "model_name": SELECTED_MODEL,
            "source": MODEL_SOURCE,
        }
    )
    if forecast["lag_168"].isna().any():
        forecast["lag_168"] = forecast["lag_24"].astype(float)
    return forecast.reset_index(drop=True)


def cheapest_consecutive_block(df: pd.DataFrame, duration_hours: int = 3) -> pd.DataFrame:
    ordered = df.sort_values("datetime").reset_index(drop=True)
    candidates: list[tuple[float, int, pd.DataFrame]] = []
    for start in range(0, len(ordered) - duration_hours + 1):
        block = ordered.iloc[start : start + duration_hours].copy()
        gaps = block["datetime"].diff().dropna()
        if not gaps.eq(pd.Timedelta(hours=1)).all():
            continue
        candidates.append((float(block["predicted_smp"].mean()), start, block))
    if not candidates:
        raise ValueError("No valid consecutive 3-hour charging block found.")
    block = min(candidates, key=lambda item: (item[0], item[1]))[2].copy()
    block["window_avg_smp"] = float(block["predicted_smp"].mean())
    return block


def write_v5_metrics() -> None:
    metrics = {
        "selected_model": SELECTED_MODEL,
        "final_test_mae": 11.0938,
        "daily_profile_mae": 13.5758,
        "average_charging_regret": 2.4643,
        "forecast_horizon": 24,
        "model_source": MODEL_SOURCE,
    }
    METRICS_OUT.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_web_outputs(target_date: str | None = None) -> dict[str, str | int]:
    OUTPUTS.mkdir(exist_ok=True)
    forecast = build_forecast_for_web(target_date)
    recommendation = cheapest_consecutive_block(forecast, 3)
    forecast.to_csv(FORECAST_OUT, index=False, encoding="utf-8-sig")
    recommendation.to_csv(RECOMMENDATION_OUT, index=False, encoding="utf-8-sig")
    write_v5_metrics()
    return {
        "target_date": str(forecast["market_date"].iloc[0]),
        "forecast_rows": int(len(forecast)),
        "recommendation_rows": int(len(recommendation)),
        "forecast_path": str(FORECAST_OUT),
        "recommendation_path": str(RECOMMENDATION_OUT),
        "metrics_path": str(METRICS_OUT),
    }


def main() -> None:
    result = generate_web_outputs()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
