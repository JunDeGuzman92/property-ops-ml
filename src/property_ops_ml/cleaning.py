import re

import pandas as pd


def normalize_column_names(df):
    """Return a copy with predictable snake_case column names."""
    cleaned = df.copy()
    cleaned.columns = [
        re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]+", "_", str(col).strip().lower())).strip("_")
        for col in cleaned.columns
    ]
    return cleaned


def clean_text_series(series):
    """Normalize text values without changing their business meaning."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace("\u00a0", " ", regex=False)
    )


def parse_money_series(series):
    """Convert currency-like values such as '$1,250.50' into floats."""
    cleaned = (
        series.astype("string")
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def parse_percent_series(series):
    """Convert values like '4.5%' or '0.045' into percentage points."""
    text = series.astype("string").str.strip()
    has_percent = text.str.contains("%", regex=False, na=False)
    values = pd.to_numeric(text.str.replace("%", "", regex=False), errors="coerce")
    values.loc[~has_percent & values.between(-1, 1, inclusive="both")] = (
        values.loc[~has_percent & values.between(-1, 1, inclusive="both")] * 100
    )
    return values


def parse_date_series(series):
    """Parse common spreadsheet/API date values into pandas datetime values."""
    return pd.to_datetime(series, errors="coerce", utc=False)


def coerce_numeric(series, default=0.0):
    """Convert a series to numeric and fill missing values."""
    return pd.to_numeric(series, errors="coerce").fillna(default)


def data_quality_report(df, key_columns=None):
    """Produce a compact data-quality report for a DataFrame."""
    total_rows = len(df)
    duplicate_count = int(df.duplicated(subset=key_columns).sum()) if key_columns else int(df.duplicated().sum())
    rows = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_count": missing,
                "missing_pct": round((missing / total_rows * 100) if total_rows else 0, 2),
                "unique_count": int(df[col].nunique(dropna=True)),
                "duplicate_rows_in_dataset": duplicate_count,
            }
        )
    return pd.DataFrame(rows)

