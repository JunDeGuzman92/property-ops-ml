import numpy as np
import pandas as pd

from .categories import (
    MAINTENANCE_KEYWORDS,
    standardize_maintenance_category,
    standardize_status,
)
from .cleaning import coerce_numeric, parse_date_series, parse_money_series, parse_percent_series


def _optional_series(df, column, default):
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)


def _days_between(later, earlier):
    delta = later - earlier
    return delta.dt.total_seconds().div(86400).clip(lower=0)


def build_maintenance_features(df, today=None):
    """Build ML-ready features from a work-order/service-request table."""
    today = pd.Timestamp(today or pd.Timestamp.today().date())
    created = parse_date_series(_optional_series(df, "created_date", pd.NaT))
    closed = parse_date_series(_optional_series(df, "closed_date", pd.NaT))
    category = standardize_maintenance_category(_optional_series(df, "category", "other"))
    status = standardize_status(_optional_series(df, "status", "unknown"))

    features = pd.DataFrame(index=df.index)
    features["age_days"] = _days_between(pd.Series(today, index=df.index), created).fillna(0)
    features["closure_days"] = _days_between(closed, created).fillna(0)
    features["is_open"] = status.eq("open").astype(int)
    features["is_closed"] = status.eq("closed").astype(int)
    features["is_winter"] = created.dt.month.isin([11, 12, 1, 2, 3]).astype(int).fillna(0)
    features["occupied_unit"] = coerce_numeric(_optional_series(df, "occupied_unit", 0))
    features["recurrence_count"] = coerce_numeric(_optional_series(df, "recurrence_count", 0))
    features["asset_age_years"] = coerce_numeric(_optional_series(df, "asset_age_years", 0))

    for label in sorted(MAINTENANCE_KEYWORDS):
        features[f"category_{label}"] = category.eq(label).astype(int)
    features["category_other"] = category.eq("other").astype(int)
    return features


def build_renewal_features(df):
    """Build ML-ready features from a renewal tracker or rent-roll extract."""
    lease_start = parse_date_series(_optional_series(df, "lease_start_date", pd.NaT))
    lease_end = parse_date_series(_optional_series(df, "lease_end_date", pd.NaT))
    current_rent = parse_money_series(_optional_series(df, "current_rent", np.nan))
    proposed_rent = parse_money_series(_optional_series(df, "proposed_rent", np.nan))
    market_rent = parse_money_series(_optional_series(df, "market_rent", np.nan))

    features = pd.DataFrame(index=df.index)
    features["tenure_months"] = (_days_between(lease_end, lease_start) / 30.44).fillna(0)
    features["current_rent"] = current_rent.fillna(current_rent.median()).fillna(0)
    features["rent_increase_pct"] = ((proposed_rent - current_rent) / current_rent * 100).replace([np.inf, -np.inf], np.nan).fillna(0)
    features["market_gap_pct"] = ((market_rent - current_rent) / current_rent * 100).replace([np.inf, -np.inf], np.nan).fillna(0)
    features["maintenance_tickets_12m"] = coerce_numeric(_optional_series(df, "maintenance_tickets_12m", 0))
    features["on_time_payment_rate"] = parse_percent_series(_optional_series(df, "on_time_payment_rate", 100)).fillna(100)
    features["satisfaction_score"] = coerce_numeric(_optional_series(df, "satisfaction_score", 3))
    return features


def build_market_rent_features(df):
    """Build features for market rent review and comp benchmarking."""
    current_rent = parse_money_series(_optional_series(df, "current_rent", np.nan))
    market_rent = parse_money_series(_optional_series(df, "market_rent", np.nan))
    prior_market_rent = parse_money_series(_optional_series(df, "prior_market_rent", np.nan))

    features = pd.DataFrame(index=df.index)
    features["current_rent"] = current_rent.fillna(0)
    features["market_rent"] = market_rent.fillna(0)
    features["rent_gap_pct"] = ((market_rent - current_rent) / current_rent * 100).replace([np.inf, -np.inf], np.nan).fillna(0)
    features["market_growth_pct"] = ((market_rent - prior_market_rent) / prior_market_rent * 100).replace([np.inf, -np.inf], np.nan).fillna(0)
    features["comp_count"] = coerce_numeric(_optional_series(df, "comp_count", 0))
    features["amenity_score"] = coerce_numeric(_optional_series(df, "amenity_score", 0.5))
    return features

