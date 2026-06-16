import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from property_ops_ml.features import build_maintenance_features
from property_ops_ml.inference import score_frame_from_artifact
from property_ops_ml.schemas import apply_schema_mapping


DEFAULT_REPO_ID = "JunDG92/property-maintenance-priority-baseline"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score a work-order CSV using the published Hugging Face baseline model."
    )
    parser.add_argument(
        "--input",
        default=str(ROOT / "examples" / "sample_data" / "work_orders.csv"),
        help="Input CSV. Can contain raw work-order columns or model-ready feature columns.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "scored_work_orders.csv"),
        help="Output CSV with score and label columns.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", default="sample_maintenance_model.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def download_model(repo_id, filename):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        local_fallback = ROOT / "huggingface" / "property-maintenance-priority-baseline" / filename
        if local_fallback.exists():
            print("huggingface_hub is not installed; using local bundled model artifact.")
            return local_fallback
        raise SystemExit("Install Hugging Face Hub first: pip install huggingface_hub") from exc
    return hf_hub_download(repo_id=repo_id, filename=filename)


def infer_schema_mapping(columns):
    mapping = {}
    if "ticket_id" in columns:
        mapping["ticket_id"] = "work_order_id"
    if "opened" in columns:
        mapping["opened"] = "created_date"
    if "closed" in columns:
        mapping["closed"] = "closed_date"
    if "issue" in columns:
        mapping["issue"] = "category"
    if "occupied" in columns:
        mapping["occupied"] = "occupied_unit"
    if "repeat_count" in columns:
        mapping["repeat_count"] = "recurrence_count"
    return mapping


def build_features_if_needed(df, model_feature_names):
    if set(model_feature_names).issubset(df.columns):
        return df[model_feature_names].copy()
    mapped = apply_schema_mapping(df, infer_schema_mapping(df.columns))
    return build_maintenance_features(mapped)


def main():
    args = parse_args()
    model_path = download_model(args.repo_id, args.filename)
    payload = json.loads(Path(model_path).read_text(encoding="utf-8"))

    source = pd.read_csv(args.input)
    features = build_features_if_needed(source, payload["feature_names"])
    scored = score_frame_from_artifact(features, payload, threshold=args.threshold)
    output = pd.concat([source.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    print(output.head().to_string(index=False))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
