from __future__ import annotations

import numpy as np
import pytest

from corridor_aiops.evaluation.sequence_models import (
    SequenceSplit,
    fit_sequence_baselines,
)


def _split(labels: list[int], seed: int) -> SequenceSplit:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(len(labels), 4, 3)).astype(np.float32)
    feature_mask = np.ones_like(values, dtype=bool)
    step_mask = np.ones(values.shape[:2], dtype=bool)
    return SequenceSplit(values, feature_mask, step_mask, np.asarray(labels, dtype=int))


def test_sequence_baselines_train_and_score_official_split_shapes() -> None:
    pytest.importorskip("torch")
    train = _split([0] * 8 + [1] * 8, 1)
    validation = _split([0] * 4 + [1] * 4, 2)
    test = _split([0] * 3 + [1] * 3, 3)

    results = fit_sequence_baselines(train, validation, test, epochs=1, seed=7)

    assert set(results) == {"lstm_autoencoder", "tabular_transformer"}
    for result in results.values():
        assert result["test_scores"].shape == (6,)
        assert result["validation_scores"].shape == (8,)
        assert np.isfinite(result["test_scores"]).all()
        assert result["training_seconds"] >= 0.0
        assert result["selected_epoch"] == 1
