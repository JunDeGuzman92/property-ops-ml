# First-Time Hugging Face Publishing And Use

This is the shortest path to publish and use the first `property-ops-ml` model artifact.

## 1. Open The Repo In VS Code

Open:

```text
C:\Users\JosefinoJrDeGuzman\Documents\Internship Assignment\property-ops-ml
```

Open a terminal in VS Code.

## 2. Install Hugging Face Tools

```powershell
pip install huggingface_hub
```

If you are using the local virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install huggingface_hub
```

If the `.venv` Python gives an error like `Unable to create process`, recreate the virtual environment:

```powershell
Remove-Item -Recurse -Force .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install huggingface_hub
```

## 3. Log In

Create a Hugging Face account, then create a token with write access.

Run:

```powershell
hf auth login
```

Paste the token only into the local terminal prompt.

## 4. Train The Sample Model

```powershell
python scripts/train_sample_model.py
```

or:

```powershell
.\.venv\Scripts\python.exe scripts\train_sample_model.py
```

This creates:

```text
artifacts/sample_maintenance_model.json
```

## 5. Prepare The Hugging Face Folder

```powershell
python scripts/prepare_hf_model_repo.py
```

or:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_hf_model_repo.py
```

This creates:

```text
huggingface/property-maintenance-priority-baseline/
```

## 6. Test Before Uploading

```powershell
cd huggingface/property-maintenance-priority-baseline
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

Exact values may differ if the sample model artifact is regenerated.

Return to the repo root:

```powershell
cd ..\..
```

## 7. Upload To Hugging Face

Replace `YOUR_HF_USERNAME` with your Hugging Face username:

```powershell
hf upload YOUR_HF_USERNAME/property-maintenance-priority-baseline huggingface/property-maintenance-priority-baseline .
```

After upload, open:

```text
https://huggingface.co/YOUR_HF_USERNAME/property-maintenance-priority-baseline
```

## 8. Use The Published Model From A Fresh Script

Install the download helper:

```powershell
pip install huggingface_hub numpy
```

Create a file called `use_published_model.py`:

```python
import json
import numpy as np
from huggingface_hub import hf_hub_download

REPO_ID = "YOUR_HF_USERNAME/property-maintenance-priority-baseline"

model_path = hf_hub_download(repo_id=REPO_ID, filename="sample_maintenance_model.json")

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
print({"score": round(score, 4), "label": "review" if score >= 0.5 else "standard"})
```

Run:

```powershell
python use_published_model.py
```

That confirms you can download your model from Hugging Face and use it locally.
