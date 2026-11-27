from __future__ import annotations

import copy

from corridor_aiops.evaluation.clean_checkout import HEADLINE_METRICS, compare_headline


def _summary() -> dict:
    row = {key: 0.5 for key in HEADLINE_METRICS}
    row["checkpoint_sha256"] = "a" * 64
    return {
        "models": {"m": {"per_seed": [dict(row), dict(row)]}},
        "uncertainty": {
            "sender_cluster_bootstrap": {"intervals_95pct": {"m": {"auroc": [0.4, 0.6]}}}
        },
    }


def test_identical_summaries_reproduce() -> None:
    result = compare_headline(_summary(), _summary())

    assert result["status"] == "REPRODUCED_WITHIN_TOLERANCE"
    assert result["per_seed_values_compared"] == 2 * len(HEADLINE_METRICS)
    assert result["checkpoint_hashes_matched"] == 2


def test_difference_above_tolerance_is_reported() -> None:
    candidate = copy.deepcopy(_summary())
    candidate["models"]["m"]["per_seed"][1]["auroc"] = 0.51

    result = compare_headline(_summary(), candidate)

    assert result["status"] == "DIFFERS"
    assert result["values_over_tolerance"] == 1


def test_missing_model_is_not_reported_as_reproduced() -> None:
    candidate = copy.deepcopy(_summary())
    candidate["models"] = {}

    result = compare_headline(_summary(), candidate)

    assert result["status"] == "DIFFERS"
    assert result["models_missing"] == ["m"]
