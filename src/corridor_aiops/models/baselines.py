from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


def isolation_forest_scores(train: np.ndarray, test: np.ndarray, seed: int) -> np.ndarray:
    model = IsolationForest(random_state=seed, n_estimators=100, contamination="auto")
    model.fit(train)
    return -model.score_samples(test)


def one_class_svm_scores(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    model = OneClassSVM(gamma="scale", nu=0.05)
    model.fit(train)
    return -model.decision_function(test).reshape(-1)


def rule_scores(metrics: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Count telemetry fields beyond the normal-data robust z-score threshold."""
    values = np.asarray(metrics, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("metrics must be a non-empty 2D feature matrix")
    return np.sum(np.abs(values) > threshold, axis=1).astype(float)
