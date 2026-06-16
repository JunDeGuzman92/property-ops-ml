# property-ops-ml

Reusable Python toolkit for property-management analytics and ML workflows.

`property-ops-ml` is designed for teams that work with messy operational files: work orders, renewal trackers, rent rolls, market surveys, inspection logs, and workbook exports. It is not tied to one public dataset or one property-management platform. The package gives analysts a repeatable way to clean data, map columns, generate ML-ready features, score review queues, train transparent baseline models, and export auditable artifacts.

## What This Library Does

- normalizes messy spreadsheet column names
- maps source-system fields into standard property-operations schemas
- cleans money, percentages, dates, text, and status fields
- standardizes common maintenance categories and unit types
- generates reusable features for maintenance, renewal, and market-rent workflows
- creates deterministic review scores when labels are not available
- trains transparent binary classifiers when labels are available
- saves portable JSON model artifacts for reproducible inference
- includes tests and examples that can run in VS Code, terminal, or CI

## Core Workflows

### Maintenance Priority

Use this for work orders, service tickets, resident requests, and maintenance logs.

Example outputs:

- standardized maintenance category
- age and closure-time features
- recurrence and occupancy flags
- priority review score
- optional trainable urgency model

### Renewal Review

Use this for renewal trackers, rent-roll extracts, payment summaries, and resident-experience indicators.

Example outputs:

- tenure features
- rent-increase pressure
- market-gap pressure
- maintenance friction indicators
- renewal review score

### Market Rent Review

Use this for rent comp files, market surveys, rent-roll comparisons, and public market-index tables.

Example outputs:

- rent-gap percentage
- market-growth features
- comparable-count checks
- market review score

## Quick Start

From the repo folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/demo_workflows.py
python scripts/train_sample_model.py
python -m unittest discover -s tests
```

If `python` is not recognized, try `py -3` instead.

## Basic Usage

```python
import pandas as pd
from property_ops_ml.pipelines import maintenance_priority_pipeline

df = pd.read_csv("examples/sample_data/work_orders.csv")

result = maintenance_priority_pipeline(
    df,
    schema_mapping={
        "ticket_id": "work_order_id",
        "opened": "created_date",
        "closed": "closed_date",
        "issue": "category",
        "status": "status",
        "occupied": "occupied_unit",
        "repeat_count": "recurrence_count",
    },
)

print(result.head())
```

## Train A Baseline Model

The package includes a small transparent logistic-regression implementation. It is useful for baseline modeling when a team has an approved label, such as `urgent_followup`, `renewal_review_required`, or `market_review_required`.

```powershell
python scripts/train_sample_model.py
```

This writes:

```text
artifacts/sample_maintenance_model.json
```

The sample model is not meant to be production-grade. It shows the model-artifact structure that can later be published to GitHub or Hugging Face.

## Hugging Face Publishing Direction

This package can support a Hugging Face model repo, but the best first public artifact is a baseline model card plus JSON model artifact, not a claim of a universal production model.

Recommended Hugging Face repo contents:

```text
README.md                 # model card
sample_maintenance_model.json
requirements.txt
inference_example.py
```

See `docs/HUGGING_FACE_PUBLISHING_GUIDE.md` for a step-by-step guide.

## Using A Published Model On A Dataset

After publishing the maintenance baseline to Hugging Face, score a CSV from this repo with:

```powershell
python scripts/score_dataset_with_published_model.py
```

This downloads the model artifact from Hugging Face, converts a work-order CSV into the expected feature matrix, appends `score` and `label`, and writes:

```text
outputs/scored_work_orders.csv
```

You can pass another CSV:

```powershell
python scripts/score_dataset_with_published_model.py --input path\to\your_work_orders.csv --output outputs\your_scored_file.csv
```

## Responsible Use

This library supports review, triage, analytics, and decision preparation. It should not be used for automated resident, tenant, or pricing decisions without approved data, validation, governance, and human oversight.

## Project Structure

```text
property-ops-ml/
  src/property_ops_ml/
    categories.py
    cleaning.py
    evaluation.py
    features.py
    inference.py
    models.py
    pipelines.py
    schemas.py
    scoring.py
  examples/sample_data/
  scripts/
  tests/
  docs/
```
