import numpy as np
import pandas as pd

from .models import load_model_artifact


def score_record_from_artifact(record, artifact_or_path, threshold=0.5):
    """Score one dictionary-like record from a JSON model artifact."""
    payload = (
        load_model_artifact(artifact_or_path)
        if isinstance(artifact_or_path, (str, bytes))
        else artifact_or_path
    )
    feature_names = payload["feature_names"]
    defaults = {}
    if "scaler" in payload:
        defaults = dict(zip(feature_names, payload["scaler"]["mean"]))
    values = np.array([[float(record.get(name, defaults.get(name, 0.0))) for name in feature_names]])
    if "scaler" in payload:
        mean = np.array(payload["scaler"]["mean"], dtype=float)
        scale = np.array(payload["scaler"]["scale"], dtype=float)
        values = (values - mean) / scale
    weights = np.array(payload["weights"], dtype=float)
    probability = 1 / (1 + np.exp(-np.clip(values @ weights + payload["bias"], -30, 30)))
    score = float(probability[0])
    return {
        "score": round(score, 4),
        "label": "review" if score >= threshold else "standard",
        "threshold": threshold,
    }


def score_frame_from_artifact(features, artifact_or_path, threshold=0.5):
    """Score a feature DataFrame with a JSON model artifact."""
    payload = (
        load_model_artifact(artifact_or_path)
        if isinstance(artifact_or_path, (str, bytes))
        else artifact_or_path
    )
    feature_names = payload["feature_names"]
    frame = pd.DataFrame(index=features.index)
    for name in feature_names:
        if name in features.columns:
            frame[name] = pd.to_numeric(features[name], errors="coerce")
        elif "scaler" in payload:
            defaults = dict(zip(feature_names, payload["scaler"]["mean"]))
            frame[name] = defaults.get(name, 0.0)
        else:
            frame[name] = 0.0

    values = frame[feature_names].fillna(0.0).to_numpy(dtype=float)
    if "scaler" in payload:
        mean = np.array(payload["scaler"]["mean"], dtype=float)
        scale = np.array(payload["scaler"]["scale"], dtype=float)
        values = (values - mean) / scale

    weights = np.array(payload["weights"], dtype=float)
    probabilities = 1 / (1 + np.exp(-np.clip(values @ weights + payload["bias"], -30, 30)))
    return pd.DataFrame(
        {
            "score": np.round(probabilities, 4),
            "label": np.where(probabilities >= threshold, "review", "standard"),
            "threshold": threshold,
        },
        index=features.index,
    )
