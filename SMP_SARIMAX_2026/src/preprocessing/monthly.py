from __future__ import annotations

import pandas as pd


def merge_monthly_with_conservative_lag(hourly: pd.DataFrame, monthly: pd.DataFrame, date_col: str, value_cols: list[str], lag_months: int = 1, monthly_date_col: str | None = None) -> pd.DataFrame:
    """As-of join monthly values at their first assumed availability timestamp.

    A conservative one-month publication delay means January's observation becomes
    available at the start of March. Monthly observations remain monthly until this join.
    """
    if lag_months < 0:
        raise ValueError("lag_months must be non-negative")
    left = hourly.copy()
    monthly_date_col = monthly_date_col or date_col
    right = monthly[[monthly_date_col, *value_cols]].copy()
    left[date_col] = pd.to_datetime(left[date_col])
    right[monthly_date_col] = pd.to_datetime(right[monthly_date_col]).dt.to_period("M").dt.to_timestamp()
    right["available_at"] = right[monthly_date_col] + pd.offsets.MonthBegin(lag_months + 1)
    right = right.drop(columns=monthly_date_col).sort_values("available_at").rename(columns={"available_at": date_col})
    left = left.sort_values(date_col)
    return pd.merge_asof(left, right, on=date_col, direction="backward", allow_exact_matches=True)
