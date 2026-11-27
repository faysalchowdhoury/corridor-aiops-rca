from __future__ import annotations

import heapq
import math
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from corridor_aiops.schema import CorridorEvent, EventKind

FEATURE_NAMES = (
    "speed_mps",
    "acceleration_mps2",
    "delta_t_s",
    "causal_jitter_s",
    "displacement_xy_m",
    "vertical_step_m",
    "speed_change_mps",
    "heading_change_rad",
    "local_density_150m",
    "duplicate_timestamp",
    "implausible_gap",
)
RAW_COORDINATE_FEATURE_NAMES = (
    "position_x_m",
    "position_y_m",
    "position_z_m",
    "heading_deg",
)
TRANSLATION_INVARIANT_KINEMATIC_FEATURES = (
    "displacement_xy_m",
    "vertical_step_m",
    "speed_change_mps",
    "heading_change_rad",
)
TIMING_FEATURES = (
    "delta_t_s",
    "causal_jitter_s",
    "duplicate_timestamp",
    "implausible_gap",
)
CAUSAL_JITTER_DEFINITION = (
    "causal_jitter_s[i] = |dt[i] - mean(dt[j] for j in 1..i-1)|, where dt[j] is the "
    "positive inter-arrival time (receive_time_ns difference / 1e9) of the same sender "
    "within the current official split; the running mean is cumulative since the last "
    "history reset (a gap > max_gap_seconds) and the feature is masked until one prior "
    "interval exists"
)
LOCAL_DENSITY_DEFINITION = (
    "number of distinct senders, including the focal sender itself, whose latest "
    "position observed within the trailing local_window_seconds lies within "
    "local_radius_meters of the focal position (so the minimum valid value is 1)"
)

DEFAULT_WINDOW_SIZE = 10
DEFAULT_MAX_GAP_SECONDS = 5.0
DEFAULT_LOCAL_RADIUS_METERS = 150.0
DEFAULT_LOCAL_WINDOW_SECONDS = 1.0
_NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class TrajectoryWindow:
    event: CorridorEvent
    values: tuple[tuple[float, ...], ...]
    time_mask: tuple[bool, ...]
    feature_mask: tuple[tuple[bool, ...], ...]


def _metric_valid(event: CorridorEvent, name: str, valid_name: str) -> bool:
    if valid_name in event.metrics:
        return event.metrics[valid_name] == 1.0
    value = event.metrics.get(name)
    return value is not None and math.isfinite(value)


def _position_xy(event: CorridorEvent) -> tuple[float, float] | None:
    valid = event.metrics.get("position_xy_valid")
    if valid is not None and valid != 1.0:
        return None
    if not _metric_valid(event, "position_x", "position_xy_valid") or not _metric_valid(
        event, "position_y", "position_xy_valid"
    ):
        return None
    x = event.metrics.get("position_x")
    y = event.metrics.get("position_y")
    if x is None or y is None:
        return None
    return x, y


def _position_z(event: CorridorEvent) -> float | None:
    if not _metric_valid(event, "position_z", "position_z_valid"):
        return None
    return event.metrics.get("position_z")


def _position_time_ns(timestamp: datetime) -> int:
    return int(round(timestamp.timestamp() * _NANOSECONDS_PER_SECOND))


def _event_time_ns(event: CorridorEvent) -> int:
    if event.receive_time_ns is not None:
        return event.receive_time_ns
    return _position_time_ns(event.timestamp)


def _local_densities(
    events: list[CorridorEvent],
    *,
    radius_meters: float,
    window_seconds: float,
) -> tuple[list[float], list[bool]]:
    radius_ns = int(window_seconds * _NANOSECONDS_PER_SECOND)
    cell_width = radius_meters
    densities = [0.0] * len(events)
    valid = [False] * len(events)
    ordered = sorted(
        enumerate(events),
        key=lambda item: (
            item[1].corridor_id,
            _event_time_ns(item[1]),
            item[1].event_id,
            item[0],
        ),
    )
    corridor_id: str | None = None
    active: dict[str, tuple[int, float, float, tuple[int, int]]] = {}
    cells: dict[tuple[int, int], set[str]] = defaultdict(set)
    expiry: list[tuple[int, str]] = []

    for index, event in ordered:
        if event.corridor_id != corridor_id:
            corridor_id = event.corridor_id
            active.clear()
            cells.clear()
            expiry.clear()

        timestamp_ns = _event_time_ns(event)
        cutoff_ns = timestamp_ns - radius_ns
        while expiry and expiry[0][0] < cutoff_ns:
            expired_at, source_id = heapq.heappop(expiry)
            current = active.get(source_id)
            if current is None or current[0] != expired_at:
                continue
            cells[current[3]].discard(source_id)
            del active[source_id]

        position = _position_xy(event)
        if position is None:
            continue

        source_id = event.source_id
        previous = active.get(source_id)
        if previous is not None:
            cells[previous[3]].discard(source_id)
        x, y = position
        cell = (math.floor(x / cell_width), math.floor(y / cell_width))
        active[source_id] = (timestamp_ns, x, y, cell)
        cells[cell].add(source_id)
        heapq.heappush(expiry, (timestamp_ns, source_id))

        count = 0
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                for candidate in cells.get((cell[0] + offset_x, cell[1] + offset_y), ()):
                    _, candidate_x, candidate_y, _ = active[candidate]
                    if math.hypot(x - candidate_x, y - candidate_y) <= radius_meters:
                        count += 1
        densities[index] = float(count)
        valid[index] = True
    return densities, valid


def _pad_window(
    history: deque[tuple[tuple[float, ...], tuple[bool, ...]]],
    window_size: int,
    event: CorridorEvent,
    feature_count: int = len(FEATURE_NAMES),
) -> TrajectoryWindow:
    missing = window_size - len(history)
    empty_values = (0.0,) * feature_count
    empty_mask = (False,) * feature_count
    rows = [(empty_values, empty_mask)] * missing + list(history)
    return TrajectoryWindow(
        event=event,
        values=tuple(row[0] for row in rows),
        time_mask=(False,) * missing + (True,) * len(history),
        feature_mask=tuple(row[1] for row in rows),
    )


def extract_trajectory_windows(
    events: Iterable[CorridorEvent],
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
    local_radius_meters: float = DEFAULT_LOCAL_RADIUS_METERS,
    local_window_seconds: float = DEFAULT_LOCAL_WINDOW_SECONDS,
    include_raw_coordinates: bool = False,
) -> list[TrajectoryWindow]:
    """Build one causal trailing window per event.

    With ``include_raw_coordinates`` the per-step vector is ``FEATURE_NAMES`` followed
    by ``RAW_COORDINATE_FEATURE_NAMES`` (absolute position and heading); this exists
    only for the raw-versus-translation-invariant ablation.
    """
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if not math.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be finite and positive")
    if not math.isfinite(local_radius_meters) or local_radius_meters <= 0:
        raise ValueError("local_radius_meters must be finite and positive")
    if not math.isfinite(local_window_seconds) or local_window_seconds <= 0:
        raise ValueError("local_window_seconds must be finite and positive")

    event_list = list(events)
    if any(event.kind != EventKind.BSM for event in event_list):
        raise ValueError("trajectory extraction accepts BSM events only")
    densities, density_valid = _local_densities(
        event_list,
        radius_meters=local_radius_meters,
        window_seconds=local_window_seconds,
    )

    names = FEATURE_NAMES + (RAW_COORDINATE_FEATURE_NAMES if include_raw_coordinates else ())
    tracks: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, event in enumerate(event_list):
        tracks[(event.corridor_id, event.source_id)].append(index)

    output: list[TrajectoryWindow | None] = [None] * len(event_list)
    max_gap_seconds = float(max_gap_seconds)
    for indices in tracks.values():
        indices.sort(
            key=lambda index: (
                _event_time_ns(event_list[index]),
                event_list[index].event_id,
                index,
            )
        )
        history: deque[tuple[tuple[float, ...], tuple[bool, ...]]] = deque(
            maxlen=window_size
        )
        prior_interval_sum = 0.0
        prior_interval_count = 0
        previous_event: CorridorEvent | None = None

        for index in indices:
            event = event_list[index]
            row = {name: 0.0 for name in names}
            mask = {name: False for name in names}
            row["duplicate_timestamp"] = 0.0
            row["implausible_gap"] = 0.0
            mask["duplicate_timestamp"] = True
            mask["implausible_gap"] = True

            speed_valid = _metric_valid(event, "speed", "speed_valid")
            acceleration_valid = _metric_valid(
                event, "acceleration", "acceleration_valid"
            )
            heading_valid = _metric_valid(event, "heading", "heading_valid")
            current_speed = event.metrics.get("speed")
            current_acceleration = event.metrics.get("acceleration")
            current_heading = event.metrics.get("heading")
            if speed_valid and current_speed is not None:
                row["speed_mps"] = current_speed
                mask["speed_mps"] = True
            if acceleration_valid and current_acceleration is not None:
                row["acceleration_mps2"] = current_acceleration
                mask["acceleration_mps2"] = True

            previous_position = _position_xy(previous_event) if previous_event else None
            current_position = _position_xy(event)
            previous_z = _position_z(previous_event) if previous_event else None
            current_z = _position_z(event)

            if previous_event is not None:
                delta_t = (
                    _event_time_ns(event) - _event_time_ns(previous_event)
                ) / _NANOSECONDS_PER_SECOND
                if delta_t <= 0:
                    row["duplicate_timestamp"] = 1.0
                elif delta_t > max_gap_seconds:
                    row["implausible_gap"] = 1.0
                    history.clear()
                    prior_interval_sum = 0.0
                    prior_interval_count = 0
                else:
                    row["delta_t_s"] = delta_t
                    mask["delta_t_s"] = True
                    if prior_interval_count:
                        mean_prior_interval = prior_interval_sum / prior_interval_count
                        row["causal_jitter_s"] = abs(delta_t - mean_prior_interval)
                        mask["causal_jitter_s"] = True
                    prior_interval_sum += delta_t
                    prior_interval_count += 1

                    if previous_position is not None and current_position is not None:
                        dx = current_position[0] - previous_position[0]
                        dy = current_position[1] - previous_position[1]
                        displacement = math.hypot(dx, dy)
                        row["displacement_xy_m"] = displacement
                        mask["displacement_xy_m"] = True

                    if previous_z is not None and current_z is not None:
                        row["vertical_step_m"] = abs(current_z - previous_z)
                        mask["vertical_step_m"] = True

                    if (
                        speed_valid
                        and current_speed is not None
                        and _metric_valid(previous_event, "speed", "speed_valid")
                    ):
                        previous_speed = previous_event.metrics.get("speed")
                        if previous_speed is not None:
                            row["speed_change_mps"] = abs(current_speed - previous_speed)
                            mask["speed_change_mps"] = True

                    if (
                        heading_valid
                        and current_heading is not None
                        and _metric_valid(previous_event, "heading", "heading_valid")
                    ):
                        previous_heading = previous_event.metrics.get("heading")
                        if previous_heading is not None:
                            angle_delta = math.radians(current_heading - previous_heading)
                            wrapped_delta = math.atan2(
                                math.sin(angle_delta), math.cos(angle_delta)
                            )
                            row["heading_change_rad"] = abs(wrapped_delta)
                            mask["heading_change_rad"] = True

            if density_valid[index]:
                row["local_density_150m"] = densities[index]
                mask["local_density_150m"] = True

            if include_raw_coordinates:
                if current_position is not None:
                    row["position_x_m"], row["position_y_m"] = current_position
                    mask["position_x_m"] = mask["position_y_m"] = True
                if current_z is not None:
                    row["position_z_m"] = current_z
                    mask["position_z_m"] = True
                if heading_valid and current_heading is not None:
                    row["heading_deg"] = current_heading
                    mask["heading_deg"] = True
            values = tuple(row[name] for name in names)
            feature_mask = tuple(mask[name] for name in names)
            history.append((values, feature_mask))
            output[index] = _pad_window(history, window_size, event, len(names))
            previous_event = event

    if any(window is None for window in output):
        raise RuntimeError("Trajectory extraction did not produce one window per input event")
    return [window for window in output if window is not None]
