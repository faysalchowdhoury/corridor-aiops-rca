from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from corridor_aiops.evaluation.benchmark import (
    _best_f1_threshold,
    _simulated_mttd,
    _source_cluster_bootstrap,
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


def test_source_cluster_bootstrap_returns_paired_resamples() -> None:
    labels = np.asarray([0, 1, 0, 1])
    source_ids = np.asarray(["a", "a", "b", "b"])
    scores = {
        "first": np.asarray([0.1, 0.9, 0.2, 0.8]),
        "second": np.asarray([0.2, 0.8, 0.1, 0.9]),
    }

    intervals, rows = _source_cluster_bootstrap(
        labels,
        source_ids,
        scores,
        {"first": 0.5, "second": 0.5},
        seed=17,
        samples=20,
        seed_rows=5,
    )

    assert set(intervals) == {"first", "second"}
    assert len(rows) == 10
    assert {row["resampling_unit"] for row in rows} == {"test_sender_id_cluster"}
    assert {row["seed"] for row in rows} == set(range(5))


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
