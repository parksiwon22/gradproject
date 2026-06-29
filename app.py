from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
FORECAST_PATH = OUTPUTS / "next_day_smp_forecast_tuned.csv"
DEFAULT_RECOMMENDATION_PATH = OUTPUTS / "recommended_3h_charging_window_tuned.csv"
METRICS_PATH = OUTPUTS / "v5_metrics.json"
TITLE = "\uc804\ub825\uc608\ubcf4 | SMP \uae30\ubc18 EV \ucda9\uc804 \ucd94\ucc9c"
V5_MODE_CAPTION = "V5 \uc608\uce21 \ubaa8\ub4dc: \uc804\uc77c 23\uc2dc \uae30\uc900 \ub2e4\uc74c \ub0a0 24\uc2dc\uac04 SMP \uc608\uce21"
PRICE_WEATHER_NOTE = "\uac00\uaca9 \ub0a0\uc528\ub294 SMP \uc608\uce21 \uae30\ubc18 \uc704\ud5d8\ub3c4\uc774\uba70 \uc2e4\uc81c \uae30\uc0c1\uc608\ubcf4\uc640 \ub2e4\ub985\ub2c8\ub2e4."
LABEL_TIME = "\uc2dc\uac04"
LABEL_PREDICTED_SMP = "\uc608\uce21 SMP"
LABEL_LAG_24 = "\uc804\uc77c \ub3d9\uc77c\uc2dc\uac04 SMP"
LABEL_LAG_168 = "\uc804\uc8fc \ub3d9\uc77c\uc2dc\uac04 SMP"
LABEL_PRICE_WEATHER = "\uac00\uaca9 \ub0a0\uc528"
LABEL_RISK_LEVEL = "\uc704\ud5d8\ub3c4"
LABEL_RAMP_PROBABILITY = "\uae09\ub4f1 \uac00\ub2a5\uc131"


st.set_page_config(
    page_title=TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_forecast() -> pd.DataFrame:
    df = pd.read_csv(FORECAST_PATH, encoding="utf-8-sig")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["market_date"] = pd.to_datetime(df["market_date"])
    df["time_label"] = df["datetime"].dt.strftime("%H:%M")
    df["ramp_probability"] = pd.to_numeric(df.get("ramp_probability", 0.0), errors="coerce").fillna(0.0)
    return df.sort_values("datetime").reset_index(drop=True)


@st.cache_data
def load_default_recommendation() -> pd.DataFrame:
    df = pd.read_csv(DEFAULT_RECOMMENDATION_PATH, encoding="utf-8-sig")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["ramp_probability"] = pd.to_numeric(df.get("ramp_probability", 0.0), errors="coerce").fillna(0.0)
    return df


@st.cache_data
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def required_charge_hours(
    battery_kwh: float,
    current_soc: float,
    target_soc: float,
    charger_kw: float,
) -> tuple[float, int, float]:
    needed_kwh = battery_kwh * max(target_soc - current_soc, 0) / 100
    raw_hours = needed_kwh / charger_kw if charger_kw > 0 else 0
    rounded_hours = max(1, math.ceil(raw_hours)) if needed_kwh > 0 else 0
    return needed_kwh, rounded_hours, raw_hours


def filter_available_hours(df: pd.DataFrame, start_hour: int, end_hour: int) -> pd.DataFrame:
    hours = df["hour"].replace({24: 0})
    if start_hour == end_hour:
        mask = pd.Series(True, index=df.index)
    elif start_hour < end_hour:
        mask = (hours >= start_hour) & (hours < end_hour)
    else:
        mask = (hours >= start_hour) | (hours < end_hour)
    return df.loc[mask].copy()


def cheapest_block(df: pd.DataFrame, required_hours: int) -> pd.DataFrame:
    if required_hours <= 0:
        return pd.DataFrame()

    candidates = []
    ordered = df.sort_values("datetime").reset_index(drop=True)
    for start in range(0, len(ordered) - required_hours + 1):
        block = ordered.iloc[start : start + required_hours].copy()
        hour_gap = block["datetime"].diff().dropna()
        if not hour_gap.eq(pd.Timedelta(hours=1)).all():
            continue
        candidates.append((block["predicted_smp"].mean(), start, block))

    if not candidates:
        return pd.DataFrame()

    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def recommendation_summary(block: pd.DataFrame) -> tuple[str, str, float]:
    start = block["datetime"].iloc[0]
    end = block["datetime"].iloc[-1] + pd.Timedelta(hours=1)
    return start.strftime("%Y-%m-%d %H:%M"), end.strftime("%Y-%m-%d %H:%M"), block["predicted_smp"].mean()


def style_metric(label: str, value: str, help_text: str | None = None):
    st.metric(label, value, help=help_text)


def format_probability(value: float) -> str:
    return f"{value * 100:.1f}%"


missing_files = [path for path in [FORECAST_PATH, DEFAULT_RECOMMENDATION_PATH] if not path.exists()]
if missing_files:
    st.error("\uc0dd\uc131\ub41c V5 \uc608\uce21 \ud30c\uc77c\uc744 \ucc3e\uc744 \uc218 \uc5c6\uc2b5\ub2c8\ub2e4.")
    for path in missing_files:
        st.code(str(path.relative_to(ROOT)))
    st.stop()

forecast = load_forecast()
default_recommendation = load_default_recommendation()
metrics = load_metrics()
forecast_date = forecast["market_date"].iloc[0].strftime("%Y-%m-%d")
max_risk_row = forecast.sort_values(["ramp_probability", "predicted_smp"], ascending=False).iloc[0]
default_start, default_end, default_avg = recommendation_summary(default_recommendation)

st.title(TITLE)
st.caption(V5_MODE_CAPTION)
st.info(PRICE_WEATHER_NOTE)
price_weather_labels = " \u00b7 ".join(sorted(forecast["price_weather"].dropna().astype(str).unique()))
risk_level_labels = " \u00b7 ".join(sorted(forecast["risk_level"].dropna().astype(str).unique()))
st.caption(f"{LABEL_PRICE_WEATHER}: {price_weather_labels} | {LABEL_RISK_LEVEL}: {risk_level_labels}")

if metrics:
    with st.expander("V5 \uc608\uce21 \uc131\ub2a5 \uc694\uc57d"):
        metric_cols = st.columns(4)
        metric_cols[0].metric("\ubaa8\ub378", metrics.get("selected_model", "V5"))
        metric_cols[1].metric("\ucd5c\uc885 MAE", f"{metrics.get('final_test_mae', 0):.4f}")
        metric_cols[2].metric("\uc77c\ubcc4 \ud504\ub85c\ud30c\uc77c MAE", f"{metrics.get('daily_profile_mae', 0):.4f}")
        metric_cols[3].metric("\ucda9\uc804 regret", f"{metrics.get('average_charging_regret', 0):.4f}")

with st.sidebar:
    st.header("EV \uc785\ub825")
    battery_kwh = st.number_input("\ubc30\ud130\ub9ac \uc6a9\ub7c9 (kWh)", min_value=20.0, max_value=150.0, value=60.0, step=1.0)
    current_soc = st.slider("\ud604\uc7ac \ucda9\uc804\uc728 (%)", min_value=0, max_value=100, value=35, step=1)
    target_soc = st.slider("\ubaa9\ud45c \ucda9\uc804\uc728 (%)", min_value=0, max_value=100, value=80, step=1)
    charger_kw = st.number_input("\ucda9\uc804\uae30 \ucd9c\ub825 (kW)", min_value=1.0, max_value=350.0, value=7.0, step=0.5)
    st.divider()
    st.subheader("\ucda9\uc804 \uac00\ub2a5 \uc2dc\uac04")
    start_hour = st.selectbox("\uc2dc\uc791 \uc2dc\uac04", options=list(range(0, 24)), index=22, format_func=lambda x: f"{x:02d}:00")
    end_hour = st.selectbox("\uc885\ub8cc \uc2dc\uac04", options=list(range(0, 24)), index=8, format_func=lambda x: f"{x:02d}:00")

if target_soc < current_soc:
    st.warning("\ubaa9\ud45c \ucda9\uc804\uc728\uc774 \ud604\uc7ac \ucda9\uc804\uc728\ubcf4\ub2e4 \ub0ae\uc2b5\ub2c8\ub2e4. \ud544\uc694 \ucda9\uc804\ub7c9\uc744 0\uc73c\ub85c \uacc4\uc0b0\ud569\ub2c8\ub2e4.")

needed_kwh, needed_hours, raw_hours = required_charge_hours(
    battery_kwh=battery_kwh,
    current_soc=current_soc,
    target_soc=target_soc,
    charger_kw=charger_kw,
)

available = filter_available_hours(forecast, start_hour, end_hour)
ev_recommendation = cheapest_block(available, needed_hours)

summary_cols = st.columns(6)
with summary_cols[0]:
    style_metric("\uc608\uce21 \ub0a0\uc9dc", forecast_date)
with summary_cols[1]:
    style_metric("\uae30\ubcf8 \ucd94\ucc9c", f"{default_start[-5:]}~{default_end[-5:]}", "V5 24\uc2dc\uac04 \uc608\uce21 \uae30\uc900 \ucd5c\uc800\uac00 3\uc2dc\uac04")
with summary_cols[2]:
    style_metric("\ud3c9\uade0 \uc608\uce21 SMP", f"{default_avg:.2f}")
with summary_cols[3]:
    style_metric("\ud544\uc694 \ucda9\uc804 \uc2dc\uac04", f"{needed_hours}\uc2dc\uac04", f"\uc608\uc0c1 \ucda9\uc804 {raw_hours:.2f}\uc2dc\uac04")
with summary_cols[4]:
    style_metric(LABEL_PRICE_WEATHER, str(max_risk_row.get("price_weather", "-")), "\uac00\uaca9 \uc608\uce21 \uae30\uc900 \ucd5c\ub300 \uc704\ud5d8")
with summary_cols[5]:
    style_metric(f"{LABEL_RISK_LEVEL} / {LABEL_RAMP_PROBABILITY}", f"{max_risk_row.get('risk_level', '-')} \u00b7 {format_probability(float(max_risk_row['ramp_probability']))}")

left, right = st.columns([1.05, 1.35], gap="large")

with left:
    st.subheader("\ucda9\uc804 \ucd94\ucc9c \uacb0\uacfc")
    st.write(
        f"\ucd94\uac00 \ucda9\uc804\ub7c9\uc740 **{needed_kwh:.1f} kWh**\uc774\uace0, "
        f"\uc785\ub825\ud55c \ucda9\uc804\uae30 \uae30\uc900 **{raw_hours:.2f}\uc2dc\uac04**\uc774 \ud544\uc694\ud569\ub2c8\ub2e4."
    )

    if needed_hours == 0:
        st.success("\ucda9\uc804\uc774 \uc774\ubbf8 \ucda9\ubd84\ud569\ub2c8\ub2e4. \ucd94\uac00 \ucda9\uc804\uc774 \ud544\uc694\ud558\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4.")
    elif ev_recommendation.empty:
        st.error("\uc120\ud0dd\ud55c \uc2dc\uac04 \uc548\uc5d0\uc11c \ud544\uc694\ud55c \uc5f0\uc18d \ucda9\uc804 \uc2dc\uac04\uc744 \ucc3e\uc9c0 \ubabb\ud588\uc2b5\ub2c8\ub2e4.")
        st.info("\ucda9\uc804 \uac00\ub2a5 \uc2dc\uac04\uc744 \ub298\ub9ac\uac70\ub098 \ubaa9\ud45c \ucda9\uc804\uc728\uc744 \ub0ae\ucdb0\ubcf4\uc138\uc694.")
    else:
        ev_start, ev_end, ev_avg = recommendation_summary(ev_recommendation)
        avg_ramp = ev_recommendation["ramp_probability"].mean()
        st.success(f"\ucd94\ucc9c \ucda9\uc804 \uad6c\uac04: {ev_start} ~ {ev_end}")
        rec_cols = st.columns(3)
        rec_cols[0].metric("\ud3c9\uade0 \uc608\uce21 SMP", f"{ev_avg:.2f}")
        rec_cols[1].metric("\ucda9\uc804 \uc2dc\uac04", f"{len(ev_recommendation)}\uc2dc\uac04")
        rec_cols[2].metric(LABEL_RAMP_PROBABILITY, format_probability(avg_ramp))

        display = pd.DataFrame(
            {
                LABEL_TIME: ev_recommendation["time_label"],
                LABEL_PREDICTED_SMP: ev_recommendation["predicted_smp"],
                LABEL_RISK_LEVEL: ev_recommendation["risk_level"],
                LABEL_PRICE_WEATHER: ev_recommendation["price_weather"],
                LABEL_RAMP_PROBABILITY: ev_recommendation["ramp_probability"].map(format_probability),
            }
        )
        st.dataframe(display, width="stretch", hide_index=True)

with right:
    st.subheader("\ub2e4\uc74c \ub0a0 24\uc2dc\uac04 SMP \uc608\uce21")
    chart_df = forecast[["time_label", "predicted_smp"]].rename(
        columns={"time_label": LABEL_TIME, "predicted_smp": LABEL_PREDICTED_SMP}
    )
    st.line_chart(chart_df, x=LABEL_TIME, y=LABEL_PREDICTED_SMP, height=320)

st.subheader("\uc2dc\uac04\ubcc4 \uc608\uce21\ud45c")
table_df = pd.DataFrame(
    {
        LABEL_TIME: forecast["time_label"],
        LABEL_PREDICTED_SMP: forecast["predicted_smp"],
        LABEL_LAG_24: forecast["lag_24"],
        LABEL_LAG_168: forecast["lag_168"],
        LABEL_PRICE_WEATHER: forecast["price_weather"],
        LABEL_RISK_LEVEL: forecast["risk_level"],
        LABEL_RAMP_PROBABILITY: forecast["ramp_probability"].map(format_probability),
    }
)
if not table_df.columns.is_unique:
    raise ValueError("Display table columns must be unique.")
st.dataframe(table_df, width="stretch", hide_index=True)

st.caption(f"\uc774 \ud654\uba74\uc740 V5 day-ahead SMP \uc608\uce21 CSV\ub97c \ud45c\uc2dc\ud569\ub2c8\ub2e4. {PRICE_WEATHER_NOTE}")
