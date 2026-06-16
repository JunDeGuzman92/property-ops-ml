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

This repo can generate that folder for you.

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
hf auth login
```

Do not paste your token into a public place. Create a token in Hugging Face with write access and paste it only into the local login prompt.

4. Train or export a model artifact:

```powershell
python scripts/train_sample_model.py
```

5. Prepare the Hugging Face upload folder:

```powershell
python scripts/prepare_hf_model_repo.py
```

This creates:

```text
huggingface/property-maintenance-priority-baseline/
```

6. Upload the folder to Hugging Face:

Replace `YOUR_HF_USERNAME` with your Hugging Face username.

```powershell
hf upload YOUR_HF_USERNAME/property-maintenance-priority-baseline huggingface/property-maintenance-priority-baseline .
```

According to Hugging Face's CLI docs, `hf upload` can upload a whole folder and will create the repo automatically if it does not exist.

7. Open the model page:

```text
https://huggingface.co/YOUR_HF_USERNAME/property-maintenance-priority-baseline
```

8. Test the model locally from the upload folder:

```powershell
cd huggingface/property-maintenance-priority-baseline
pip install -r requirements.txt
python inference_example.py
```

## Model Card Language

Use careful wording:

> This model is a baseline review-support model for property-management maintenance triage. It is intended for reproducible experimentation and should be retrained and validated on approved internal data before operational use.

Avoid claiming:

- automated tenant decisions
- universal accuracy across all companies
- production readiness without validation
- use of private company data
