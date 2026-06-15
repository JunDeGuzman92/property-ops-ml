import pandas as pd

from .features import (
    build_maintenance_features,
    build_market_rent_features,
    build_renewal_features,
)
from .schemas import apply_schema_mapping
from .scoring import (
    add_review_band,
    score_maintenance_priority,
    score_market_rent_review,
    score_renewal_review,
)


def maintenance_priority_pipeline(df, schema_mapping=None):
    """Return source rows with maintenance features and review scores."""
    working = apply_schema_mapping(df, schema_mapping)
    features = build_maintenance_features(working)
    output = pd.concat([working.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    output["maintenance_priority_score"] = score_maintenance_priority(features)
    output["maintenance_review_band"] = add_review_band(output["maintenance_priority_score"])
    return output


def renewal_review_pipeline(df, schema_mapping=None):
    """Return source rows with renewal features and review scores."""
    working = apply_schema_mapping(df, schema_mapping)
    features = build_renewal_features(working)
    output = pd.concat([working.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    output["renewal_review_score"] = score_renewal_review(features)
    output["renewal_review_band"] = add_review_band(output["renewal_review_score"])
    return output


def market_rent_review_pipeline(df, schema_mapping=None):
    """Return source rows with market-rent features and review scores."""
    working = apply_schema_mapping(df, schema_mapping)
    features = build_market_rent_features(working)
    output = pd.concat([working.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    output["market_review_score"] = score_market_rent_review(features)
    output["market_review_band"] = add_review_band(output["market_review_score"])
    return output

