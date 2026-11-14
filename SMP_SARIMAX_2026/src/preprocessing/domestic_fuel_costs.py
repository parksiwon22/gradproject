from __future__ import annotations

import pandas as pd


RAW_MONTHS = pd.period_range("2022-11", "2025-12", freq="M")
FORECAST_MONTHS = pd.period_range("2023-01", "2025-12", freq="M")
KPX_FUEL_TYPES = {"원자력", "유연탄", "무연탄", "LNG", "유류"}


def _month_period(values: pd.Series, name: str) -> pd.Series:
    text = values.astype("string").str.replace(r"\D", "", regex=True).str.slice(0, 6)
    parsed = pd.to_datetime(text, format="%Y%m", errors="coerce")
    if parsed.isna().any():
        bad = values[parsed.isna()].astype(str).head(3).tolist()
        raise ValueError(f"{name} contains invalid YYYYMM values: {bad}")
    return parsed.dt.to_period("M")


def map_domestic_fuel_costs(
    epsis_monthly: pd.DataFrame,
    kpx_monthly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map monthly domestic fuel costs to origin month M using source month M-2.

    Input frames must already be normalized from the official EPSIS export and
    KPX API/CSV. No values are filled, inferred, or substituted. The returned
    tables are indexed by forecast-origin month, so callers can keep the
    feature fixed throughout all 72 forecast hours, including month crossings.
    """
    epsis_required = {"month", "lng_fuel_cost_rate_krw_per_kwh"}
    if not epsis_required.issubset(epsis_monthly.columns):
        raise ValueError("EPSIS normalized data must contain month and lng_fuel_cost_rate_krw_per_kwh.")
    kpx_required = {"fuelType", "day", "untpc", "unit"}
    if not kpx_required.issubset(kpx_monthly.columns):
        raise ValueError("KPX data must retain fuelType, day, untpc, and unit fields.")

    epsis = epsis_monthly[["month", "lng_fuel_cost_rate_krw_per_kwh"]].copy()
    epsis["source_month"] = _month_period(epsis["month"], "EPSIS month")
    epsis["lng_fuel_cost_rate_krw_per_kwh"] = pd.to_numeric(epsis["lng_fuel_cost_rate_krw_per_kwh"], errors="coerce")
    if epsis.source_month.duplicated().any():
        raise ValueError("EPSIS export has duplicate monthly LNG fuel-cost rows.")
    if epsis.lng_fuel_cost_rate_krw_per_kwh.isna().any():
        raise ValueError("EPSIS LNG 연료비단가 contains missing or non-numeric values.")

    kpx = kpx_monthly[["fuelType", "day", "untpc", "unit"]].copy()
    kpx["fuelType"] = kpx.fuelType.astype("string").str.strip()
    kpx["source_month"] = _month_period(kpx.day, "KPX day")
    kpx["untpc"] = pd.to_numeric(kpx.untpc, errors="coerce")
    if kpx.untpc.isna().any() or kpx.unit.isna().any():
        raise ValueError("KPX untpc/unit contains missing or invalid values.")
    if kpx.duplicated(["source_month", "fuelType"]).any():
        raise ValueError("KPX has duplicate source_month/fuelType rows; preserve and resolve source duplicates explicitly.")
    unexpected = set(kpx.fuelType.dropna().unique()) - KPX_FUEL_TYPES
    if unexpected:
        raise ValueError(f"Unexpected KPX fuelType values require review: {sorted(unexpected)}")

    month_map = pd.DataFrame({"forecast_month": FORECAST_MONTHS})
    month_map["source_month"] = month_map.forecast_month - 2
    expected_raw = set(RAW_MONTHS)
    epsis_missing = sorted(expected_raw - set(epsis.source_month))
    if epsis_missing:
        raise ValueError("EPSIS monthly raw coverage missing: " + ", ".join(map(str, epsis_missing)))
    kpx_missing_months = sorted(expected_raw - set(kpx.source_month))
    if kpx_missing_months:
        raise ValueError("KPX monthly raw coverage missing: " + ", ".join(map(str, kpx_missing_months)))
    missing_categories = [
        f"{month}/{fuel}" for month in RAW_MONTHS for fuel in sorted(KPX_FUEL_TYPES)
        if not ((kpx.source_month == month) & (kpx.fuelType == fuel)).any()
    ]
    if missing_categories:
        raise ValueError("KPX monthly fuel category coverage missing: " + ", ".join(missing_categories[:24]))

    epsis_model = month_map.merge(epsis[["source_month", "lng_fuel_cost_rate_krw_per_kwh"]], on="source_month", how="left", validate="one_to_one")
    kpx_model = month_map.merge(kpx[["source_month", "fuelType", "day", "untpc", "unit"]], on="source_month", how="left", validate="one_to_many")
    if epsis_model.lng_fuel_cost_rate_krw_per_kwh.isna().any() or kpx_model.untpc.isna().any():
        raise ValueError("Two-month lag mapping produced missing model input rows.")
    epsis_model["forecast_month"] = epsis_model.forecast_month.astype(str)
    epsis_model["source_month"] = epsis_model.source_month.astype(str)
    kpx_model["forecast_month"] = kpx_model.forecast_month.astype(str)
    kpx_model["source_month"] = kpx_model.source_month.astype(str)
    return epsis_model, kpx_model
