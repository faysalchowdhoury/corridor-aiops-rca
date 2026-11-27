from __future__ import annotations

import io
import json
import math
import zipfile
from datetime import datetime, timedelta, timezone

import pytest

from corridor_aiops.data.parse import veremi_events
from corridor_aiops.data.trajectory import (
    FEATURE_NAMES,
    extract_trajectory_windows,
)
from corridor_aiops.schema import CorridorEvent, EventKind


def _bsm(
    event_id: str,
    source_id: str,
    second: float,
    x: float,
    y: float,
    z: float = 0.0,
    *,
    speed: float = 10.0,
    acceleration: float = 0.0,
    heading: float = 0.0,
    position_valid: bool = True,
) -> CorridorEvent:
    return CorridorEvent(
        event_id=event_id,
        timestamp=datetime(2022, 1, 1, tzinfo=timezone.utc)
        + timedelta(seconds=second),
        corridor_id="trace",
        source_id=source_id,
        kind=EventKind.BSM,
        label="benign",
        metrics={
            "position_x": x,
            "position_y": y,
            "position_z": z,
            "position_xy_valid": float(position_valid),
            "position_z_valid": 1.0,
            "speed": speed,
            "speed_valid": 1.0,
            "acceleration": acceleration,
            "acceleration_valid": 1.0,
            "heading": heading,
            "heading_valid": 1.0,
        },
    )


def _last_features(window) -> dict[str, float]:
    return dict(zip(FEATURE_NAMES, window.values[-1], strict=True))


def test_trajectory_features_are_translation_invariant_in_2d_and_3d() -> None:
    events = [
        _bsm("a0", "a", 0.0, 0.0, 0.0, 0.0),
        _bsm("a1", "a", 1.0, 10.0, 0.0, 1.0, speed=10.0, heading=359.0),
        _bsm("b0", "b", 0.5, 12.0, 4.0, 2.0),
        _bsm("a2", "a", 2.0, 20.0, 0.0, 1.0, speed=10.0, heading=1.0),
    ]
    offset = (12345.25, -6789.5, 42.0)
    translated = [
        _bsm(
            event.event_id,
            event.source_id,
            (event.timestamp - datetime(2022, 1, 1, tzinfo=timezone.utc)).total_seconds(),
            event.metrics["position_x"] + offset[0],
            event.metrics["position_y"] + offset[1],
            event.metrics["position_z"] + offset[2],
            speed=event.metrics["speed"],
            acceleration=event.metrics["acceleration"],
            heading=event.metrics["heading"],
        )
        for event in events
    ]

    original = extract_trajectory_windows(events, window_size=4)
    shifted = extract_trajectory_windows(translated, window_size=4)

    for first, second in zip(original, shifted, strict=True):
        assert first.time_mask == second.time_mask
        assert first.feature_mask == second.feature_mask
        max_difference = max(
            abs(left - right)
            for left_row, right_row in zip(first.values, second.values, strict=True)
            for left, right in zip(left_row, right_row, strict=True)
        )
        assert max_difference < 1e-6


def test_observed_speed_and_wrapped_heading_changes_use_differences_not_motion_models() -> None:
    events = [
        _bsm("a0", "a", 0.0, 0.0, 0.0, speed=10.0, heading=359.0),
        _bsm("a1", "a", 1.0, 2.0, 0.0, speed=12.0, heading=1.0),
    ]

    features = _last_features(extract_trajectory_windows(events, window_size=2)[1])

    assert features["speed_change_mps"] == pytest.approx(2.0)
    assert features["heading_change_rad"] == pytest.approx(math.radians(2.0))
    assert "position_residual_m" not in FEATURE_NAMES
    assert "speed_residual_mps" not in FEATURE_NAMES
    assert "heading_rate_rad_s" not in FEATURE_NAMES


def test_jitter_uses_only_prior_intervals_and_is_not_changed_by_future_events() -> None:
    events = [
        _bsm("a0", "a", 0.0, 0.0, 0.0),
        _bsm("a1", "a", 1.0, 10.0, 0.0),
        _bsm("a2", "a", 3.0, 30.0, 0.0),
        _bsm("a3", "a", 4.0, 40.0, 0.0),
    ]
    prefix = extract_trajectory_windows(events[:3], window_size=1)
    complete = extract_trajectory_windows(events, window_size=1)
    jitter_index = FEATURE_NAMES.index("causal_jitter_s")

    assert _last_features(prefix[1])["causal_jitter_s"] == 0.0
    assert prefix[1].feature_mask[-1][jitter_index] is False
    assert _last_features(prefix[2])["causal_jitter_s"] == pytest.approx(1.0)
    assert prefix[2].feature_mask[-1][jitter_index] is True
    assert [window.values for window in prefix] == [
        window.values for window in complete[:3]
    ]


def test_nanosecond_time_is_used_when_python_datetimes_are_equal() -> None:
    first = _bsm("a0", "a", 0.0, 0.0, 0.0).model_copy(
        update={"receive_time_ns": 1_000_000_000}
    )
    second = _bsm("a1", "a", 0.0, 1.0, 0.0).model_copy(
        update={"receive_time_ns": 1_000_000_100}
    )

    windows = extract_trajectory_windows([first, second], window_size=2)

    assert windows[0].event.timestamp == windows[1].event.timestamp
    assert _last_features(windows[1])["delta_t_s"] == pytest.approx(1e-7)
    assert _last_features(windows[1])["duplicate_timestamp"] == 0.0


def test_parser_marks_missing_and_non_finite_telemetry_fields_invalid(tmp_path) -> None:
    record = {
        "rcvTime": 1_000_000_000,
        "sender_id": "veh-1",
        "messageID": 3,
        "attacker": 0,
        "sender": {
            "spd": "nan",
            "acl": 0.1,
            "hed": 90,
            "pos": "1.0,inf",
        },
    }
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr("messages/veh-1.json", json.dumps([record]))
    path = tmp_path / "published.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("InTAS/Train/InTAS-train.zip", inner.getvalue())

    event = next(veremi_events(path))
    window = extract_trajectory_windows([event], window_size=1)[0]

    assert event.metrics["speed"] == 0.0
    assert event.metrics["speed_valid"] == 0.0
    assert event.metrics["position_xy_valid"] == 0.0
    assert window.feature_mask[0][FEATURE_NAMES.index("speed_mps")] is False
    assert window.feature_mask[0][FEATURE_NAMES.index("local_density_150m")] is False


def test_parser_deduplicates_receiver_copies_and_keeps_earliest_raw_receive_time(
    tmp_path,
) -> None:
    record = {
        "rcvTime": 1_000_000_010,
        "sendTime": 900_000_000,
        "sender_id": "veh-1",
        "messageID": 3,
        "attacker": 1,
        "sender": {"spd": 10, "acl": 0.1, "hed": 90, "pos": "1.0,2.0,0.0"},
    }
    earlier = {**record, "rcvTime": 1_000_000_000}
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr("messages/veh-1.json", json.dumps([record]))
        archive.writestr("messages/veh-2.json", json.dumps([earlier]))
    test_inner = io.BytesIO()
    with zipfile.ZipFile(test_inner, "w") as archive:
        archive.writestr("messages/veh-3.json", json.dumps([record]))
    path = tmp_path / "published.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("InTAS/Train/InTAS-train.zip", inner.getvalue())
        archive.writestr("InTAS/Test/InTAS-test.zip", test_inner.getvalue())

    events = list(veremi_events(path))

    assert len(events) == 2
    train_event = next(event for event in events if event.corridor_id.endswith("-train"))
    test_event = next(event for event in events if event.corridor_id.endswith("-test"))
    assert train_event.broadcast_id == "veh-1:3:900000000"
    assert train_event.receive_time_ns == 1_000_000_000
    assert train_event.timestamp.timestamp() == 1.0
    assert train_event.corridor_id.endswith("-train")
    assert test_event.corridor_id.endswith("-test")


def test_duplicate_timestamps_large_gaps_and_missing_positions_are_masked() -> None:
    events = [
        _bsm("a0", "a", 0.0, 0.0, 0.0),
        _bsm("a1", "a", 0.0, 1.0, 0.0),
        _bsm("a2", "a", 10.0, 100.0, 0.0, position_valid=False),
    ]
    windows = extract_trajectory_windows(events, window_size=4, max_gap_seconds=5.0)
    duplicate = _last_features(windows[1])
    gap = _last_features(windows[2])

    assert duplicate["duplicate_timestamp"] == 1.0
    assert windows[1].feature_mask[-1][FEATURE_NAMES.index("delta_t_s")] is False
    assert gap["implausible_gap"] == 1.0
    assert windows[2].time_mask == (False, False, False, True)
    assert windows[2].feature_mask[-1][FEATURE_NAMES.index("local_density_150m")] is False


def test_local_density_uses_distinct_sources_within_radius_and_causal_order() -> None:
    events = [
        _bsm("a0", "a", 0.0, 0.0, 0.0),
        _bsm("b0", "b", 0.25, 100.0, 0.0),
        _bsm("c0", "c", 0.5, 300.0, 0.0),
    ]
    windows = extract_trajectory_windows(events, window_size=2, local_radius_meters=150.0)

    densities = [_last_features(window)["local_density_150m"] for window in windows]
    assert densities == [1.0, 2.0, 1.0]


def test_extractor_rejects_non_bsm_events_and_invalid_configuration() -> None:
    detector = CorridorEvent(
        event_id="det",
        timestamp=datetime(2022, 1, 1, tzinfo=timezone.utc),
        corridor_id="trace",
        source_id="detector-1",
        kind=EventKind.DETECTOR,
    )
    with pytest.raises(ValueError, match="BSM"):
        extract_trajectory_windows([detector])
    with pytest.raises(ValueError, match="window_size"):
        extract_trajectory_windows([], window_size=0)
