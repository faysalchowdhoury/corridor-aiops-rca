from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np


def rank_root_causes(
    metric_names: Sequence[str],
    values: np.ndarray,
    event_sources: Sequence[str],
    *,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """Rank observed nodes by absolute deviation from the replay median.

    This is an explanatory anomaly ranking, not an identified causal effect.
    """
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(metric_names):
        raise ValueError("values must be a 2D matrix aligned with metric_names")
    if len(event_sources) != matrix.shape[0]:
        raise ValueError("event_sources must align with matrix rows")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    center = np.nanmedian(matrix, axis=0)
    spread = np.nanmedian(np.abs(matrix - center), axis=0)
    spread[spread == 0] = 1.0
    node_scores: dict[str, float] = defaultdict(float)
    for row, source in zip(matrix, event_sources, strict=True):
        score = float(np.nansum(np.abs((row - center) / spread)))
        node_scores[source] += score
    return sorted(node_scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]


def perturbation_attribution(
    predict_score,
    features: dict[str, float],
) -> list[tuple[str, float]]:
    """Return finite-difference feature effects for one event score function."""
    baseline = float(predict_score(features))
    effects = []
    for name, _value in features.items():
        perturbed = dict(features)
        perturbed[name] = 0.0
        effects.append((name, baseline - float(predict_score(perturbed))))
    return sorted(effects, key=lambda item: abs(item[1]), reverse=True)
