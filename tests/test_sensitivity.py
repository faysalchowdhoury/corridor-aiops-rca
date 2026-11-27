from __future__ import annotations

import numpy as np
import pytest

from corridor_aiops.evaluation.heldout import scenario_of
from corridor_aiops.evaluation.sensitivity import ci_verdict, confusion


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        ([0.01, 0.05], "reference_better"),
        ([-0.05, -0.01], "reference_worse"),
        ([-0.01, 0.02], "inconclusive"),
        ([0.0, 0.02], "inconclusive"),
    ],
)
def test_ci_verdict_requires_the_whole_interval_on_one_side_of_zero(interval, expected) -> None:
    assert ci_verdict(interval) == expected


def test_confusion_counts_alerts_at_or_above_threshold() -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.5, 0.9, 0.5, 0.8, 0.3])

    result = confusion(labels, scores, 0.5)

    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (2, 2, 1, 2)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["false_alarm_rate"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(2 * 2 / (2 * 2 + 2 + 1))


def test_confusion_without_alerts_reports_zero_precision_not_nan() -> None:
    result = confusion(np.asarray([0, 1]), np.asarray([0.1, 0.2]), 1.0)

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0


def test_scenario_is_parsed_from_composite_corridor_id() -> None:
    corridors = np.asarray(["InTAS_highway_2_dataReplay-train", "InTAS_urban_2_dosAttack-test"])

    assert scenario_of(corridors).tolist() == ["InTAS_highway_2", "InTAS_urban_2"]
