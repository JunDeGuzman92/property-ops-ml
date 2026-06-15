# VS Code Walkthrough

Open this repo in VS Code and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/demo_workflows.py
python scripts/train_sample_model.py
python -m unittest discover -s tests
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## What To Inspect

- `src/property_ops_ml/cleaning.py`: text, money, percentage, date, and quality-report utilities
- `src/property_ops_ml/features.py`: universal maintenance, renewal, and market-rent feature builders
- `src/property_ops_ml/scoring.py`: transparent review scoring rules
- `src/property_ops_ml/models.py`: dependency-light logistic regression and model artifact export
- `scripts/demo_workflows.py`: end-to-end scoring demo
- `scripts/train_sample_model.py`: baseline model training and JSON artifact export

## How To Adapt To Company Data

1. Export an approved file from the source system.
2. Create a `schema_mapping` dictionary from source columns to standard columns.
3. Run the appropriate pipeline.
4. Review data-quality output.
5. Train only if you have a reliable approved label.
6. Validate with business users before operational use.

