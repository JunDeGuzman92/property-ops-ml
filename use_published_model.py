import json
import numpy as np
from huggingface_hub import hf_hub_download

REPO_ID = "JunDG92/property-maintenance-priority-baseline"

model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="sample_maintenance_model.json"
)

payload = json.load(open(model_path, "r", encoding="utf-8"))

record = {
    "age_days": 4,
    "closure_days": 0,
    "is_open": 1,
    "is_closed": 0,
    "is_winter": 1,
    "occupied_unit": 1,
    "recurrence_count": 2,
    "asset_age_years": 14,
    "category_hvac": 1,
}

feature_names = payload["feature_names"]
values = np.array([[float(record.get(name, 0.0)) for name in feature_names]])

mean = np.array(payload["scaler"]["mean"])
scale = np.array(payload["scaler"]["scale"])
values = (values - mean) / scale

weights = np.array(payload["weights"])
probability = 1 / (1 + np.exp(-np.clip(values @ weights + payload["bias"], -30, 30)))

score = float(probability[0])
print({
    "score": round(score, 4),
    "label": "review" if score >= 0.5 else "standard"
})