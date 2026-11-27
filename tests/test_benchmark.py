from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from corridor_aiops.evaluation.benchmark import (
    ABLATION_CONFIGS,
    FULL_CONFIG,
    EventTable,
    _best_f1_threshold,
    _simulated_mttd,
    evaluate_operating_points,
    far_budget_threshold,
    sender_disjoint_partition,
    split_eligibility,
    summarize_seeds,
)
from corridor_aiops.schema import CorridorEvent, EventKind


def _bsm(event_id: str, timestamp_ns: int, label: str) -> CorridorEvent:
    return CorridorEvent(
        event_id=event_id,
        timestamp=datetime.fromtimestamp(timestamp_ns / 1e9, tz=timezone.utc),
        receive_time_ns=timestamp_ns,
        corridor_id="scenario-test",
        source_id="vehicle-1",
        kind=EventKind.BSM,
        label=label,
    )


def test_validation_threshold_is_selected_without_using_test_labels() -> None:
    threshold = _best_f1_threshold(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.1, 0.2, 0.7, 0.8]),
    )

    assert threshold == 0.7


def test_far_budget_threshold_never_exceeds_budget_on_calibration_scores() -> None:
    benign = np.asarray([0.1, 0.2, 0.3, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    for budget in (0.0, 0.1, 0.15, 0.25):
        threshold = far_budget_threshold(benign, budget)
        assert np.mean(benign >= threshold) <= budget + 1e-12


def test_operating_points_report_both_far_definitions() -> None:
    validation_labels = np.asarray([0] * 100 + [1] * 20)
    validation_scores = np.concatenate((np.linspace(0, 1, 100), np.full(20, 2.0)))
    test_labels = np.asarray([0] * 100 + [1] * 10)
    test_scores = np.concatenate((np.linspace(0, 1, 100), np.full(10, 2.0)))

    metrics = evaluate_operating_points(
        validation_labels, validation_scores, test_labels, test_scores
    )

    assert metrics["test_roc_tpr_at_far_1.0pct"] == 1.0
    assert metrics["validation_calibrated_recall_at_far_1.0pct"] == 1.0
    assert metrics["validation_calibrated_realized_far_at_far_1.0pct"] <= 0.01
    assert metrics["validation_calibrated_realized_far_at_far_5.0pct"] <= 0.05


def test_seed_summary_uses_sample_standard_deviation() -> None:
    summary = summarize_seeds([{"auroc": 0.9, "label": "x"}, {"auroc": 0.8, "label": "y"}])

    assert summary["auroc"]["mean"] == pytest.approx(0.85)
    assert summary["auroc"]["std"] == pytest.approx(np.std([0.9, 0.8], ddof=1))
    assert "label" not in summary


def _table(sources: list[str], corridors: list[str], labels: list[int]) -> EventTable:
    count = len(sources)
    return EventTable(
        events=[],
        values=np.zeros((count, 1, 1), dtype=np.float32),
        feature_mask=np.ones((count, 1, 1), dtype=bool),
        step_mask=np.ones((count, 1), dtype=bool),
        labels=np.asarray(labels),
        corridor=np.asarray(corridors),
        source=np.asarray(sources),
        archive=np.asarray([corridor.split("-")[0] for corridor in corridors]),
        split=np.asarray([corridor.split("-")[1] for corridor in corridors]),
        feature_names=("f",),
    )


def test_sender_disjoint_partition_has_zero_bare_and_composite_overlap() -> None:
    sources = [f"veh_{index % 40}" for index in range(400)]
    corridors = [
        ("replay" if index % 2 else "dos") + "-" + ("train", "validation", "test")[index % 3]
        for index in range(400)
    ]
    labels = [int(index % 40 < 12) for index in range(400)]

    split_rows, meta = sender_disjoint_partition(_table(sources, corridors, labels))

    bare = {split: set(np.asarray(sources)[rows]) for split, rows in split_rows.items()}
    assert not bare["train"] & bare["test"]
    assert not bare["train"] & bare["validation"]
    assert not bare["validation"] & bare["test"]
    assert meta["overlap"]["composite_train_test"] == 0
    assert sum(len(rows) for rows in split_rows.values()) == 400
    assert split_eligibility(np.asarray(labels), split_rows) is None


def test_split_eligibility_marks_single_class_partition_not_evaluable() -> None:
    labels = np.asarray([0, 1, 0, 0])
    reason = split_eligibility(
        labels,
        {"train": np.asarray([0, 1]), "validation": np.asarray([0, 1]), "test": np.asarray([2, 3])},
    )

    assert reason is not None and "test partition lacks both attacker classes" in reason


def test_ablation_configs_change_one_factor_each() -> None:
    families = dict(ABLATION_CONFIGS)

    assert set(families) == {
        "window_length",
        "coordinate_representation",
        "timing_and_jitter",
        "local_density",
        "validity_masks",
    }
    assert families["window_length"].window == 1
    assert families["window_length"].feature_names == FULL_CONFIG.feature_names
    assert "position_x_m" in families["coordinate_representation"].feature_names
    assert "displacement_xy_m" not in families["coordinate_representation"].feature_names
    assert "causal_jitter_s" not in families["timing_and_jitter"].feature_names
    assert "local_density_150m" not in families["local_density"].feature_names
    assert families["validity_masks"].use_masks is False
    for name, config in families.items():
        if name != "validity_masks":
            assert config.use_masks


def test_simulated_mttd_uses_integer_simulator_receive_times() -> None:
    events = [
        _bsm("benign", 1_000_000_000, "benign"),
        _bsm("onset", 2_000_000_000, "cyber"),
        _bsm("detected", 3_000_000_000, "cyber"),
    ]

    result = _simulated_mttd(events, np.asarray([0.1, 0.2, 0.8]), threshold=0.5)

    assert result["attacker_label_onsets"] == 1
    assert result["detected_onsets"] == 1
    assert result["mean_detected_onset_delay_seconds"] == 1.0
    assert "not field-incident MTTD" in result["time_basis"]
