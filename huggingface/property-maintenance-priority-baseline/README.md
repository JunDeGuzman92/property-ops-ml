---
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

## Score A Dataset

The model can score a CSV dataset when the file has either:

1. model-ready feature columns, or
2. common work-order columns such as `created_date`, `closed_date`, `category`, `status`, `occupied_unit`, `recurrence_count`, and `asset_age_years`.

Run the bundled batch example:

```bash
python batch_inference_example.py
```

This reads:

```text
sample_work_orders.csv
```

and writes:

```text
scored_work_orders.csv
```

To score your own CSV:

```bash
python batch_inference_example.py --input your_work_orders.csv --output scored_your_work_orders.csv
```

## Responsible Use

This model is for review support and learning. Any real deployment should include
data governance, validation on approved local data, threshold calibration, drift
monitoring, and human review.
