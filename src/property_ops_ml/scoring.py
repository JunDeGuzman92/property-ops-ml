import numpy as np
import pandas as pd


def _clip_score(values):
    return pd.Series(values).clip(0, 100).round(2)


def score_maintenance_priority(features):
    """Create a transparent 0-100 maintenance-priority review score."""
    score = (
        18 * features.get("is_open", 0)
        + 12 * features.get("occupied_unit", 0)
        + 8 * np.log1p(features.get("age_days", 0))
        + 10 * np.log1p(features.get("recurrence_count", 0))
        + 12 * features.get("category_life_safety", 0)
        + 10 * features.get("category_hvac", 0) * features.get("is_winter", 0)
        + 8 * features.get("category_plumbing", 0)
        + 6 * features.get("category_electrical", 0)
    )
    return _clip_score(score)


def score_renewal_review(features):
    """Create a transparent 0-100 renewal-review score."""
    score = (
        8 * np.log1p(features.get("maintenance_tickets_12m", 0))
        + 1.8 * features.get("rent_increase_pct", 0).clip(lower=0)
        + 1.3 * features.get("market_gap_pct", 0).clip(lower=0)
        + 18 * (1 - (features.get("on_time_payment_rate", 100) / 100)).clip(lower=0)
        + 7 * (3 - features.get("satisfaction_score", 3)).clip(lower=0)
    )
    return _clip_score(score)


def score_market_rent_review(features):
    """Create a transparent 0-100 market-rent review score."""
    score = (
        2.0 * features.get("rent_gap_pct", 0).abs()
        + 1.5 * features.get("market_growth_pct", 0).clip(lower=0)
        + 4 * (features.get("comp_count", 0) < 3).astype(int)
        + 5 * (features.get("amenity_score", 0.5) < 0.4).astype(int)
    )
    return _clip_score(score)


def add_review_band(score, low=25, high=60):
    """Convert numeric review scores into simple review bands."""
    return pd.cut(
        score,
        bins=[-0.01, low, high, 100],
        labels=["standard", "review", "high_review"],
    ).astype(str)

