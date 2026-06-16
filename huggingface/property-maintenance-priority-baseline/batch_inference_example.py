import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_PATH = Path("sample_maintenance_model.json")
DEFAULT_INPUT = Path("sample_work_orders.csv")
DEFAULT_OUTPUT = Path("scored_work_orders.csv")

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


def sigmoid(value):
    value = np.clip(value, -30, 30)
    return 1.0 / (1.0 + np.exp(-value))


def parse_args():
    parser = argparse.ArgumentParser(description="Batch-score a work-order CSV.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def standardize_category(value):
    text = "" if pd.isna(value) else str(value).lower()
    for label, keywords in MAINTENANCE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return label
    return "other"


def build_features(df, feature_names):
    if set(feature_names).issubset(df.columns):
        return df[feature_names].copy()

    working = df.copy()
    rename_map = {
        "opened": "created_date",
        "closed": "closed_date",
        "issue": "category",
        "occupied": "occupied_unit",
        "repeat_count": "recurrence_count",
    }
    working = working.rename(columns={k: v for k, v in rename_map.items() if k in working.columns})

    created = pd.to_datetime(working.get("created_date"), errors="coerce")
    closed = pd.to_datetime(working.get("closed_date"), errors="coerce")
    today = pd.Timestamp.today().normalize()
    status = working.get("status", pd.Series("", index=working.index)).astype(str).str.lower()
    category = working.get("category", pd.Series("", index=working.index)).apply(standardize_category)

    features = pd.DataFrame(index=working.index)
    features["age_days"] = ((today - created).dt.total_seconds() / 86400).clip(lower=0).fillna(0)
    features["closure_days"] = ((closed - created).dt.total_seconds() / 86400).clip(lower=0).fillna(0)
    features["is_open"] = status.str.contains("open|progress|pending|assigned", regex=True).astype(int)
    features["is_closed"] = status.str.contains("closed|complete|resolved|done", regex=True).astype(int)
    features["is_winter"] = created.dt.month.isin([11, 12, 1, 2, 3]).astype(int).fillna(0)
    features["occupied_unit"] = pd.to_numeric(working.get("occupied_unit", 0), errors="coerce").fillna(0)
    features["recurrence_count"] = pd.to_numeric(working.get("recurrence_count", 0), errors="coerce").fillna(0)
    features["asset_age_years"] = pd.to_numeric(working.get("asset_age_years", 0), errors="coerce").fillna(0)

    for label in MAINTENANCE_KEYWORDS:
        features[f"category_{label}"] = category.eq(label).astype(int)
    features["category_other"] = category.eq("other").astype(int)

    for name in feature_names:
        if name not in features.columns:
            features[name] = 0.0
    return features[feature_names]


def score_features(features, payload, threshold):
    values = features.to_numpy(dtype=float)
    mean = np.array(payload["scaler"]["mean"], dtype=float)
    scale = np.array(payload["scaler"]["scale"], dtype=float)
    values = (values - mean) / scale
    weights = np.array(payload["weights"], dtype=float)
    probabilities = sigmoid(values @ weights + payload["bias"])
    return pd.DataFrame(
        {
            "score": np.round(probabilities, 4),
            "label": np.where(probabilities >= threshold, "review", "standard"),
        },
        index=features.index,
    )


def main():
    args = parse_args()
    payload = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    source = pd.read_csv(args.input)
    features = build_features(source, payload["feature_names"])
    scored = score_features(features, payload, threshold=args.threshold)
    output = pd.concat([source.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
    output.to_csv(args.output, index=False)
    print(output.to_string(index=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
