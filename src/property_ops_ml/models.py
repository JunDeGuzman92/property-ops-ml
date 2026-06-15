import json
from pathlib import Path

import numpy as np


class StandardScaler:
    """Small dependency-light scaler for tabular features."""

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, x):
        return (np.asarray(x, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def to_dict(self):
        return {
            "mean": self.mean_.round(8).tolist(),
            "scale": self.scale_.round(8).tolist(),
        }


class LogisticRegressionGD:
    """Transparent logistic regression trained with gradient descent."""

    def __init__(self, learning_rate=0.08, epochs=2000, l2=0.01):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2

    @staticmethod
    def _sigmoid(values):
        values = np.clip(values, -30, 30)
        return 1.0 / (1.0 + np.exp(-values))

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_rows, n_features = x.shape
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        for _ in range(self.epochs):
            probabilities = self._sigmoid(x @ self.weights_ + self.bias_)
            error = probabilities - y
            grad_w = (x.T @ error) / n_rows + self.l2 * self.weights_
            grad_b = error.mean()
            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b
        return self

    def predict_proba(self, x):
        return self._sigmoid(np.asarray(x, dtype=float) @ self.weights_ + self.bias_)

    def predict(self, x, threshold=0.5):
        return (self.predict_proba(x) >= threshold).astype(int)

    def to_dict(self, feature_names, scaler=None, metrics=None, metadata=None):
        payload = {
            "model_type": "logistic_regression_gradient_descent",
            "feature_names": list(feature_names),
            "weights": self.weights_.round(8).tolist(),
            "bias": round(float(self.bias_), 8),
        }
        if scaler is not None:
            payload["scaler"] = scaler.to_dict()
        if metrics is not None:
            payload["metrics"] = metrics
        if metadata is not None:
            payload["metadata"] = metadata
        return payload


def save_model_artifact(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_model_artifact(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

