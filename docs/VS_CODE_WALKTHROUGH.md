# VS Code Walkthrough

## Important: Open The Folder First

Before running any commands, open the actual repo folder in VS Code:

```text
C:\Users\JosefinoJrDeGuzman\Documents\Internship Assignment\property-ops-ml
```

In VS Code, use:

```text
File > Open Folder
```

If the left sidebar says **NO FOLDER OPENED**, stop and open the repo folder first.

## Important: Run Commands In The Terminal

The commands below are **PowerShell terminal commands**, not Python code.

Do not type them into an untitled editor tab and click the Python **Run** button. If you do that, VS Code will try to run the file as Python and may show an error like:

```text
can't open file 'C:\Users\...\Untitled-1': [Errno 2] No such file or directory
```

Instead:

1. Open the repo folder.
2. Go to **Terminal > New Terminal**.
3. Paste the commands into the terminal panel at the bottom.
4. Press Enter.

## Setup And Test

Run these commands in the VS Code terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/demo_workflows.py
python scripts/train_sample_model.py
python -m unittest discover -s tests
```

If `python` is not recognized, use:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3 -m pip install --upgrade pip
py -3 -m pip install -r requirements.txt
py -3 scripts/demo_workflows.py
py -3 scripts/train_sample_model.py
py -3 -m unittest discover -s tests
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## What The Correct Terminal Should Look Like

After opening the folder, your terminal prompt should look similar to this:

```powershell
PS C:\Users\JosefinoJrDeGuzman\Documents\Internship Assignment\property-ops-ml>
```

If it only shows:

```powershell
PS C:\Users\JosefinoJrDeGuzman>
```

then you are not inside the repo folder yet. Run:

```powershell
cd "C:\Users\JosefinoJrDeGuzman\Documents\Internship Assignment\property-ops-ml"
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
