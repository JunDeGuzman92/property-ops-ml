# Hugging Face Publishing Guide

This guide explains how `property-ops-ml` can be published as a model artifact or demo on Hugging Face.

## Best First Artifact

Publish a baseline model repo for a specific workflow, such as:

```text
property-maintenance-priority-baseline
```

Recommended files:

```text
README.md
sample_maintenance_model.json
requirements.txt
inference_example.py
```

## Why Not Publish One Universal Model Immediately?

Property-management data is not universal. Each company has different:

- work-order categories
- status values
- rent-roll columns
- renewal workflows
- escalation rules
- resident data governance requirements

The universal artifact should be the **library**. Individual models should be trained against approved local data and published with a clear model card.

## Steps

1. Create a Hugging Face account.
2. Install the CLI:

```powershell
pip install huggingface_hub
```

3. Log in:

```powershell
huggingface-cli login
```

4. Train or export a model artifact:

```powershell
python scripts/train_sample_model.py
```

5. Create a Hugging Face model repo:

```powershell
huggingface-cli repo create property-maintenance-priority-baseline --type model
```

6. Clone the new repo and copy files:

```powershell
git clone https://huggingface.co/YOUR_USERNAME/property-maintenance-priority-baseline
copy artifacts\sample_maintenance_model.json property-maintenance-priority-baseline\
copy requirements.txt property-maintenance-priority-baseline\
```

7. Add a model card as `README.md`.

8. Commit and push:

```powershell
cd property-maintenance-priority-baseline
git add .
git commit -m "Add baseline property operations model"
git push
```

## Model Card Language

Use careful wording:

> This model is a baseline review-support model for property-management maintenance triage. It is intended for reproducible experimentation and should be retrained and validated on approved internal data before operational use.

Avoid claiming:

- automated tenant decisions
- universal accuracy across all companies
- production readiness without validation
- use of private company data

