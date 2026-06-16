import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HF_DIR = ROOT / "huggingface" / "property-maintenance-priority-baseline"
ARTIFACT_PATH = ROOT / "artifacts" / "sample_maintenance_model.json"


README = """---
license: mit
tags:
  - property-management
  - tabular-classification
  - maintenance
  - operations-analytics
  - logistic-regression
library_name: property-ops-ml
pipeline_tag: tabular-classification
---

# Property Maintenance Priority Baseline

This repository contains a small baseline model artifact created with
[`property-ops-ml`](https://github.com/JunDeGuzman92/property-ops-ml), a reusable
Python toolkit for property-management analytics workflows.

## Model Summary

The model is a transparent logistic-regression baseline for maintenance-priority
review. It scores work-order-like records and returns a review-support label.

This is not a universal production model. It is a reproducible example showing how
property-management teams can package a model artifact, document it, and load it
for inference.

## Intended Use

Appropriate uses:

- workflow demonstration
- reproducible ML packaging
- review-support prototyping
- data-cleaning and feature-engineering practice
- baseline modeling before retraining on approved internal data

Out-of-scope uses:

- automated maintenance decisions
- tenant/resident decisions
- production dispatching without validation
- use as a substitute for property-management judgment

## Training Data

The included artifact was trained on a tiny example file bundled with the
`property-ops-ml` GitHub repo. It is intentionally small and exists only to
demonstrate the workflow.

Teams should retrain and validate the model using approved internal work-order
labels before operational use.

## Files

- `sample_maintenance_model.json`: portable model artifact
- `inference_example.py`: first-use inference script
- `sample_record.json`: sample work-order-like record
- `requirements.txt`: minimal runtime dependencies

## First Use

```bash
pip install -r requirements.txt
python inference_example.py
```

Expected output:

```json
{
  "score": 0.8,
  "label": "review",
  "threshold": 0.5
}
```

Exact score may vary if the artifact is regenerated.

## Responsible Use

This model is for review support and learning. Any real deployment should include
data governance, validation on approved local data, threshold calibration, drift
monitoring, and human review.
"""


REQUIREMENTS = """numpy>=1.24
pandas>=2.0
huggingface_hub>=0.32
"""


SAMPLE_RECORD = """{
  "age_days": 4,
  "closure_days": 0,
  "is_open": 1,
  "is_closed": 0,
  "is_winter": 1,
  "occupied_unit": 1,
  "recurrence_count": 2,
  "asset_age_years": 14,
  "category_appliance": 0,
  "category_cleaning": 0,
  "category_cosmetic": 0,
  "category_electrical": 0,
  "category_hvac": 1,
  "category_life_safety": 0,
  "category_pest": 0,
  "category_plumbing": 0,
  "category_other": 0
}
"""


INFERENCE_EXAMPLE = '''import json
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
'''


def main():
    if not ARTIFACT_PATH.exists():
        raise SystemExit(
            "Model artifact is missing. Run `python scripts/train_sample_model.py` first."
        )

    HF_DIR.mkdir(parents=True, exist_ok=True)
    (HF_DIR / "README.md").write_text(README, encoding="utf-8")
    (HF_DIR / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (HF_DIR / "sample_record.json").write_text(SAMPLE_RECORD, encoding="utf-8")
    (HF_DIR / "inference_example.py").write_text(INFERENCE_EXAMPLE, encoding="utf-8")
    shutil.copy2(ARTIFACT_PATH, HF_DIR / "sample_maintenance_model.json")
    print(f"Prepared Hugging Face model folder: {HF_DIR}")


if __name__ == "__main__":
    main()
