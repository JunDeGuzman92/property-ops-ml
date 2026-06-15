import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from property_ops_ml.evaluation import binary_classification_metrics, stratified_train_test_split_indices
from property_ops_ml.features import build_maintenance_features
from property_ops_ml.models import (
    LogisticRegressionGD,
    StandardScaler,
    save_model_artifact,
)
from property_ops_ml.schemas import apply_schema_mapping


def main():
    data_path = ROOT / "examples" / "sample_data" / "work_orders.csv"
    df = pd.read_csv(data_path)
    mapped = apply_schema_mapping(
        df,
        {
            "ticket_id": "work_order_id",
            "opened": "created_date",
            "closed": "closed_date",
            "issue": "category",
            "occupied": "occupied_unit",
            "repeat_count": "recurrence_count",
        },
    )
    features = build_maintenance_features(mapped)
    target = mapped["urgent_followup"].astype(int).to_numpy()
    train_idx, test_idx = stratified_train_test_split_indices(target, test_size=0.25, random_state=7)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(features.iloc[train_idx].to_numpy())
    x_test = scaler.transform(features.iloc[test_idx].to_numpy())
    y_train = target[train_idx]
    y_test = target[test_idx]

    model = LogisticRegressionGD(epochs=1000, learning_rate=0.08, l2=0.01)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    scores = model.predict_proba(x_test)
    metrics = binary_classification_metrics(y_test, predictions, scores)

    payload = model.to_dict(
        features.columns,
        scaler=scaler,
        metrics=metrics,
        metadata={
            "workflow": "maintenance_priority",
            "data_boundary": "small example data only; replace with approved internal labels",
            "responsible_use": "review support, not automated maintenance decisioning",
        },
    )
    out_path = ROOT / "artifacts" / "sample_maintenance_model.json"
    save_model_artifact(payload, out_path)
    print(metrics)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
