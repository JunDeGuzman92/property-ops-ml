import numpy as np


def train_test_split_indices(n_rows, test_size=0.25, random_state=42):
    """Return deterministic train/test index arrays."""
    rng = np.random.default_rng(random_state)
    indices = np.arange(n_rows)
    rng.shuffle(indices)
    test_count = max(1, int(round(n_rows * test_size)))
    return indices[test_count:], indices[:test_count]


def stratified_train_test_split_indices(y, test_size=0.25, random_state=42):
    """Return train/test indices while preserving binary class coverage when possible."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(random_state)
    train_parts = []
    test_parts = []
    for label in np.unique(y):
        class_indices = np.where(y == label)[0]
        rng.shuffle(class_indices)
        test_count = max(1, int(round(len(class_indices) * test_size)))
        if test_count >= len(class_indices) and len(class_indices) > 1:
            test_count = len(class_indices) - 1
        test_parts.append(class_indices[:test_count])
        train_parts.append(class_indices[test_count:])
    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def binary_classification_metrics(y_true, y_pred, y_score=None):
    """Compute compact binary classification metrics without sklearn."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    if y_score is not None:
        metrics["roc_auc"] = round(float(_roc_auc(y_true, np.asarray(y_score))), 4)
    return metrics


def _roc_auc(y_true, y_score):
    positives = y_score[y_true == 1]
    negatives = y_score[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.5
    wins = 0.0
    for pos in positives:
        wins += (pos > negatives).sum()
        wins += 0.5 * (pos == negatives).sum()
    return wins / (len(positives) * len(negatives))
