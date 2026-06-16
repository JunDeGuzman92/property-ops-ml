import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from property_ops_ml.evaluation import binary_classification_metrics, stratified_train_test_split_indices
from property_ops_ml.inference import score_frame_from_artifact, score_record_from_artifact
from property_ops_ml.models import LogisticRegressionGD, StandardScaler, load_model_artifact, save_model_artifact


class ModelTests(unittest.TestCase):
    def test_logistic_model_learns_simple_pattern(self):
        x = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1])
        scaler = StandardScaler()
        model = LogisticRegressionGD(learning_rate=0.2, epochs=1200, l2=0.0)
        model.fit(scaler.fit_transform(x), y)
        preds = model.predict(scaler.transform(x))
        self.assertGreaterEqual((preds == y).mean(), 0.83)

    def test_metrics_keys(self):
        metrics = binary_classification_metrics([0, 1, 1], [0, 1, 0], [0.1, 0.9, 0.4])
        self.assertIn("f1", metrics)
        self.assertIn("roc_auc", metrics)

    def test_stratified_split_keeps_class_coverage(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        train_idx, test_idx = stratified_train_test_split_indices(y, test_size=0.25, random_state=1)
        self.assertEqual(set(y[test_idx]), {0, 1})
        self.assertEqual(set(y[train_idx]), {0, 1})

    def test_artifact_roundtrip_and_inference(self):
        payload = {
            "feature_names": ["x"],
            "weights": [2.0],
            "bias": -0.5,
            "scaler": {"mean": [0.0], "scale": [1.0]},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            save_model_artifact(payload, path)
            loaded = load_model_artifact(path)
            result = score_record_from_artifact({"x": 1.0}, loaded)
        self.assertEqual(result["label"], "review")

    def test_frame_inference_scores_multiple_rows(self):
        payload = {
            "feature_names": ["x"],
            "weights": [2.0],
            "bias": -0.5,
            "scaler": {"mean": [0.0], "scale": [1.0]},
        }
        import pandas as pd

        result = score_frame_from_artifact(pd.DataFrame({"x": [1.0, -1.0]}), payload)
        self.assertEqual(list(result["label"]), ["review", "standard"])


if __name__ == "__main__":
    unittest.main()
