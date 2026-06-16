import json
from pathlib import Path

import numpy as np


MODEL_PATH = Path("sample_maintenance_model.json")
RECORD_PATH = Path("sample_record.json")


def sigmoid(value):
    value = np.clip(value, -30, 30)
    return 1.0 / (1.0 + np.exp(-value))


def score_record(record, payload, threshold=0.5):
    feature_names = payload["feature_names"]
    values = np.array([[float(record.get(name, 0.0)) for name in feature_names]], dtype=float)

    scaler = payload.get("scaler")
    if scaler:
        mean = np.array(scaler["mean"], dtype=float)
        scale = np.array(scaler["scale"], dtype=float)
        values = (values - mean) / scale

    weights = np.array(payload["weights"], dtype=float)
    probability = float(sigmoid((values @ weights) + payload["bias"])[0])
    return {
        "score": round(probability, 4),
        "label": "review" if probability >= threshold else "standard",
        "threshold": threshold,
    }


def main():
    payload = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    record = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    print(json.dumps(score_record(record, payload), indent=2))


if __name__ == "__main__":
    main()
