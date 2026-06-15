import re

import pandas as pd


MAINTENANCE_KEYWORDS = {
    "hvac": ["hvac", "heat", "hot water", "air conditioning", "ac", "boiler"],
    "plumbing": ["plumbing", "leak", "water", "toilet", "sink", "drain", "pipe"],
    "electrical": ["electric", "electrical", "outlet", "power", "breaker", "light"],
    "appliance": ["appliance", "fridge", "refrigerator", "stove", "dishwasher", "washer", "dryer"],
    "life_safety": ["smoke", "carbon", "alarm", "fire", "safety", "lock", "security"],
    "pest": ["pest", "rodent", "mouse", "mice", "bug", "cockroach", "bedbug"],
    "cleaning": ["trash", "garbage", "clean", "sanitation", "unsanitary"],
    "cosmetic": ["paint", "plaster", "floor", "wall", "door", "window", "cosmetic"],
}

STATUS_KEYWORDS = {
    "open": ["open", "new", "assigned", "in progress", "pending"],
    "closed": ["closed", "complete", "completed", "resolved", "done"],
    "cancelled": ["cancel", "cancelled", "void", "duplicate"],
}


def _standardize_one(value, keyword_map, default):
    text = "" if pd.isna(value) else str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    for label, keywords in keyword_map.items():
        if any(keyword in text for keyword in keywords):
            return label
    return default


def standardize_maintenance_category(series, default="other"):
    """Map messy maintenance text into broad operational categories."""
    return series.apply(lambda value: _standardize_one(value, MAINTENANCE_KEYWORDS, default))


def standardize_status(series, default="unknown"):
    """Map messy status labels into open/closed/cancelled buckets."""
    return series.apply(lambda value: _standardize_one(value, STATUS_KEYWORDS, default))


def standardize_unit_type(series):
    """Normalize common unit-type labels like studio, 1 bed, and 2BR."""
    text = series.astype("string").str.strip().str.lower()
    text = text.str.replace(r"\s+", " ", regex=True)
    replacements = {
        "bachelor": "studio",
        "0 bed": "studio",
        "0br": "studio",
        "1 bedroom": "1br",
        "one bedroom": "1br",
        "1 bed": "1br",
        "2 bedroom": "2br",
        "two bedroom": "2br",
        "2 bed": "2br",
        "3 bedroom": "3br",
        "three bedroom": "3br",
        "3 bed": "3br",
    }
    return text.replace(replacements)

