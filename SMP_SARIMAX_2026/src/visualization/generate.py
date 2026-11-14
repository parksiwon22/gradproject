from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import OUT

COLORS = {"actual_smp": "#202124", "D+1": "#2878B5", "D+2": "#E07A1F", "D+3": "#3A9D5D"}


def _read(split: str) -> pd.DataFrame:
    df = pd.read_csv(OUT / "predictions" / f"{split}_predictions.csv", parse_dates=["forecast_origin", "target_timestamp"])
    df["forecast_date"] = pd.to_datetime(df["target_timestamp"]).dt.date.astype(str)
    df["target_month"] = pd.to_datetime(df["target_timestamp"]).dt.to_period("M").astype(str)
    return df


def _actual_month(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates("target_timestamp").sort_values("target_timestamp")


def _month_chart(df: pd.DataFrame, split: str, month: str) -> None:
    subset = df[df.target_month == month]
    if subset.empty: return
    actual = _actual_month(subset)
    mae = subset.absolute_error.mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual.target_timestamp, y=actual.actual_smp, name="Actual SMP", line={"color": COLORS["actual_smp"], "width": 1.4}))
    for day in ("D+1", "D+2", "D+3"):
        part = subset[subset.forecast_day == day].sort_values("target_timestamp")
        fig.add_trace(go.Scatter(x=part.target_timestamp, y=part.predicted_smp, name=f"SARIMAX Predicted SMP {day}", line={"color": COLORS[day], "width": 1.1}))
    fig.update_layout(title=f"{month} SMP: Actual vs SARIMAX — {split.replace('_',' ').title()} (overall MAE {mae:.2f})", xaxis_title="Date / time (KST)", yaxis_title="SMP (KRW/kWh)", hovermode="x unified", template="plotly_white")
    dest = OUT / "figures" / "monthly"; dest.mkdir(parents=True, exist_ok=True)
    fig.write_html(dest / f"{month}_{split}.html", include_plotlyjs="cdn")
    plt.figure(figsize=(14, 5))
    plt.plot(actual.target_timestamp, actual.actual_smp, label="Actual SMP", color=COLORS["actual_smp"], lw=1.3)
    for day in ("D+1", "D+2", "D+3"):
        part = subset[subset.forecast_day == day].sort_values("target_timestamp")
        plt.plot(part.target_timestamp, part.predicted_smp, label=f"SARIMAX Predicted SMP {day}", color=COLORS[day], lw=1.0, alpha=.9)
    plt.title(f"{month} SMP: Actual vs SARIMAX ({split}; MAE {mae:.2f})")
    plt.xlabel("Date / time (KST)"); plt.ylabel("SMP (KRW/kWh)"); plt.legend(ncol=2); plt.tight_layout()
    plt.savefig(dest / f"{month}_{split}.png", dpi=160); plt.close()


def _six_month_chart(df: pd.DataFrame, split: str) -> None:
    if df.empty: return
    actual = _actual_month(df)
    mae = df.absolute_error.mean()
    dest = OUT / "figures" / "six_months"; dest.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=actual.target_timestamp, y=actual.actual_smp, name="Actual SMP", line={"color": COLORS["actual_smp"], "width": 1.2}))
    for day in ("D+1", "D+2", "D+3"):
        part = df[df.forecast_day == day].sort_values("target_timestamp")
        fig.add_trace(go.Scattergl(x=part.target_timestamp, y=part.predicted_smp, name=f"SARIMAX Predicted SMP {day}", line={"color": COLORS[day], "width": 1.0}))
    fig.update_layout(title=f"2025 {split.replace('_',' ').title()} SMP hourly comparison (MAE {mae:.2f})", xaxis_title="Date / time (KST)", yaxis_title="SMP (KRW/kWh)", hovermode="x unified", template="plotly_white")
    fig.write_html(dest / f"{split}_hourly.html", include_plotlyjs="cdn")
    # Static six-month overview uses hourly points; no date-by-date graph is emitted.
    plt.figure(figsize=(16, 6)); plt.plot(actual.target_timestamp, actual.actual_smp, label="Actual SMP", color=COLORS["actual_smp"], lw=1.3)
    for day in ("D+1", "D+2", "D+3"):
        part = df[df.forecast_day == day].sort_values("target_timestamp")
        plt.plot(part.target_timestamp, part.predicted_smp, label=f"SARIMAX Predicted SMP {day}", color=COLORS[day], lw=0.8, alpha=.85)
    plt.title(f"2025 {split.replace('_',' ').title()} hourly SMP overview (MAE {mae:.2f})")
    plt.xlabel("Date / time (KST)"); plt.ylabel("SMP (KRW/kWh)"); plt.legend(ncol=2); plt.tight_layout()
    plt.savefig(dest / f"{split}_hourly.png", dpi=160); plt.close()


def _error_charts(all_df: pd.DataFrame) -> None:
    dest = OUT / "figures" / "errors"; dest.mkdir(parents=True, exist_ok=True)
    df = all_df.dropna(subset=["actual_smp", "predicted_smp"]).copy()
    if df.empty: return
    month = df.groupby(["split", "target_month"], as_index=False).absolute_error.mean()
    fig = go.Figure()
    for split, part in month.groupby("split"):
        fig.add_trace(go.Bar(x=part.target_month, y=part.absolute_error, name=split.replace("_", " ")))
    fig.update_layout(barmode="group", title="Monthly MAE by split", xaxis_title="Month", yaxis_title="MAE (KRW/kWh)", template="plotly_white")
    fig.write_html(dest / "monthly_mae.html", include_plotlyjs="cdn")
    pivot = month.pivot(index="target_month", columns="split", values="absolute_error")
    ax = pivot.plot(kind="bar", figsize=(13, 5), color=["#2878B5", "#E07A1F"][:len(pivot.columns)])
    ax.set_title("Monthly MAE by split"); ax.set_xlabel("Month"); ax.set_ylabel("MAE (KRW/kWh)"); plt.tight_layout(); plt.savefig(dest / "monthly_mae.png", dpi=160); plt.close()
    by_day = df.groupby(["split", "forecast_day"], as_index=False).absolute_error.mean()
    fig2 = go.Figure()
    for split, part in by_day.groupby("split"):
        fig2.add_trace(go.Bar(x=part.forecast_day, y=part.absolute_error, name=split.replace("_", " ")))
    fig2.update_layout(barmode="group", title="MAE by forecast day", xaxis_title="Forecast day", yaxis_title="MAE (KRW/kWh)", template="plotly_white")
    fig2.write_html(dest / "mae_by_forecast_day.html", include_plotlyjs="cdn")
    pivot_day = by_day.pivot(index="forecast_day", columns="split", values="absolute_error").reindex(["D+1", "D+2", "D+3"])
    ax = pivot_day.plot(kind="bar", figsize=(8, 5), color=["#2878B5", "#E07A1F"][:len(pivot_day.columns)])
    ax.set_title("MAE by forecast day"); ax.set_xlabel("Forecast day"); ax.set_ylabel("MAE (KRW/kWh)"); plt.tight_layout(); plt.savefig(dest / "mae_by_forecast_day.png", dpi=160); plt.close()
    horizon = df.groupby(["split", "horizon"], as_index=False).absolute_error.mean()
    fig3 = go.Figure()
    for split, part in horizon.groupby("split"):
        fig3.add_trace(go.Scatter(x=part.horizon, y=part.absolute_error, mode="lines", name=split.replace("_", " ")))
    fig3.update_layout(title="MAE by forecast horizon", xaxis_title="Forecast horizon (hours)", yaxis_title="MAE (KRW/kWh)", template="plotly_white")
    fig3.write_html(dest / "mae_by_horizon.html", include_plotlyjs="cdn")
    plt.figure(figsize=(10, 5))
    for split, part in horizon.groupby("split"):
        plt.plot(part.horizon, part.absolute_error, label=split.replace("_", " "))
    plt.title("MAE by forecast horizon"); plt.xlabel("Forecast horizon (hours)"); plt.ylabel("MAE (KRW/kWh)"); plt.legend(); plt.tight_layout(); plt.savefig(dest / "mae_by_horizon.png", dpi=160); plt.close()
    fig4 = go.Figure(go.Scattergl(x=df.actual_smp, y=df.predicted_smp, mode="markers", marker={"size": 3, "opacity": .35}, name="Forecast rows"))
    lo = min(df.actual_smp.min(), df.predicted_smp.min()); hi = max(df.actual_smp.max(), df.predicted_smp.max())
    fig4.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction"))
    fig4.update_layout(title="Actual vs predicted SMP", xaxis_title="Actual SMP (KRW/kWh)", yaxis_title="SARIMAX Predicted SMP (KRW/kWh)", template="plotly_white")
    fig4.write_html(dest / "actual_vs_predicted.html", include_plotlyjs="cdn")
    plt.figure(figsize=(6, 6)); plt.scatter(df.actual_smp, df.predicted_smp, s=6, alpha=.25); plt.plot([lo, hi], [lo, hi], color="#D04A3A", lw=1)
    plt.title("Actual vs predicted SMP"); plt.xlabel("Actual SMP (KRW/kWh)"); plt.ylabel("SARIMAX Predicted SMP (KRW/kWh)"); plt.tight_layout(); plt.savefig(dest / "actual_vs_predicted.png", dpi=160); plt.close()


def generate() -> None:
    sets = {}
    for split in ("validation", "final_test"):
        path = OUT / "predictions" / f"{split}_predictions.csv"
        if not path.exists(): continue
        df = _read(split); sets[split] = df
        for month in sorted(df.target_month.unique()): _month_chart(df, split, month)
        _six_month_chart(df, split)
    if sets: _error_charts(pd.concat(sets.values(), ignore_index=True))
