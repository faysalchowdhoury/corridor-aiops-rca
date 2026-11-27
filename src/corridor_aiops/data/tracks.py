from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from corridor_aiops.data.trajectory import (
    TrajectoryWindow,
    _event_time_ns,
    _position_xy,
    extract_trajectory_windows,
)
from corridor_aiops.schema import CorridorEvent, EventKind

_NANOSECONDS_PER_SECOND = 1_000_000_000
_MODEL_DEFINITIONS = (
    {
        "name": "isolation_forest",
        "implementation_status": "implemented_baseline",
        "input": "causal_trajectory_features",
        "inputs_available_in_veremi": True,
        "emitter_scoring": "aggregate_event_anomaly_scores_by_sender",
    },
    {
        "name": "one_class_svm",
        "implementation_status": "implemented_baseline",
        "input": "causal_trajectory_features",
        "inputs_available_in_veremi": True,
        "emitter_scoring": "aggregate_event_anomaly_scores_by_sender",
    },
    {
        "name": "robust_threshold_rules",
        "implementation_status": "implemented_baseline",
        "input": "causal_trajectory_features",
        "inputs_available_in_veremi": True,
        "emitter_scoring": "aggregate_event_anomaly_scores_by_sender",
    },
    {
        "name": "lstm_autoencoder",
        "implementation_status": "architecture_only",
        "input": "causal_trajectory_sequences",
        "inputs_available_in_veremi": True,
        "emitter_scoring": "aggregate_event_anomaly_scores_by_sender",
    },
    {
        "name": "tabular_transformer",
        "implementation_status": "architecture_only",
        "input": "causal_trajectory_sequences",
        "inputs_available_in_veremi": True,
        "emitter_scoring": "aggregate_event_anomaly_scores_by_sender",
    },
    {
        "name": "corridor_log_transformer",
        "implementation_status": "architecture_only",
        "input": "trajectory_sequences_and_controller_text_logs",
        "inputs_available_in_veremi": False,
        "emitter_scoring": "aggregate_event_anomaly_scores_by_sender",
    },
    {
        "name": "static_gcn",
        "implementation_status": "architecture_only",
        "input": "trajectory_features_and_dynamic_v2x_graph",
        "inputs_available_in_veremi": True,
        "emitter_scoring": "node_localization_head",
    },
    {
        "name": "async_st_gnn",
        "implementation_status": "architecture_only",
        "input": "causal_trajectory_sequences_and_dynamic_v2x_graph",
        "inputs_available_in_veremi": True,
        "emitter_scoring": "node_localization_head",
    },
)


@dataclass(frozen=True)
class V2XGraphSlice:
    corridor_id: str
    slice_start_ns: int
    source_ids: tuple[str, ...]
    windows: tuple[TrajectoryWindow, ...]
    attacker_labels: tuple[bool, ...]
    adjacency: tuple[tuple[bool, ...], ...]


def build_v2x_graph_slices(
    events: list[CorridorEvent],
    *,
    radius_meters: float = 300.0,
    time_slice_seconds: float = 1.0,
) -> list[V2XGraphSlice]:
    """Create time-local communication-neighborhood graphs from labeled BSM records."""
    if not math.isfinite(radius_meters) or radius_meters <= 0:
        raise ValueError("radius_meters must be finite and positive")
    if not math.isfinite(time_slice_seconds) or time_slice_seconds <= 0:
        raise ValueError("time_slice_seconds must be finite and positive")
    slice_width_ns = int(time_slice_seconds * _NANOSECONDS_PER_SECOND)
    if slice_width_ns < 1:
        raise ValueError("time_slice_seconds must be at least one nanosecond")
    if any(event.kind != EventKind.BSM for event in events):
        raise ValueError("V2X graph construction accepts BSM events only")

    windows = extract_trajectory_windows(events)
    grouped: dict[tuple[str, int], dict[str, tuple[CorridorEvent, TrajectoryWindow, bool]]] = (
        defaultdict(dict)
    )
    for event, window in zip(events, windows, strict=True):
        if event.label not in {"benign", "cyber"}:
            raise ValueError(f"BSM event {event.event_id!r} has no supported VeReMi label")
        timestamp_ns = _event_time_ns(event)
        group_key = (event.corridor_id, timestamp_ns // slice_width_ns)
        sender = grouped[group_key].get(event.source_id)
        attacker = event.label == "cyber"
        if sender is None:
            grouped[group_key][event.source_id] = (event, window, attacker)
            continue
        previous_event, _, previous_attacker = sender
        if (_event_time_ns(event), event.event_id) > (
            _event_time_ns(previous_event),
            previous_event.event_id,
        ):
            grouped[group_key][event.source_id] = (
                event,
                window,
                attacker or previous_attacker,
            )
        elif attacker and not previous_attacker:
            grouped[group_key][event.source_id] = (
                previous_event,
                sender[1],
                True,
            )

    graph_slices = []
    for (corridor_id, time_bin), senders in sorted(grouped.items()):
        source_ids = tuple(sorted(senders))
        nodes = [senders[source_id] for source_id in source_ids]
        adjacency = [[False] * len(nodes) for _ in nodes]
        positions = [_position_xy(event) for event, _, _ in nodes]
        for left, left_position in enumerate(positions):
            if left_position is None:
                continue
            for right in range(left + 1, len(positions)):
                right_position = positions[right]
                if right_position is None:
                    continue
                distance = math.hypot(
                    left_position[0] - right_position[0],
                    left_position[1] - right_position[1],
                )
                if distance <= radius_meters:
                    adjacency[left][right] = True
                    adjacency[right][left] = True
        graph_slices.append(
            V2XGraphSlice(
                corridor_id=corridor_id,
                slice_start_ns=time_bin * slice_width_ns,
                source_ids=source_ids,
                windows=tuple(window for _, window, _ in nodes),
                attacker_labels=tuple(label for _, _, label in nodes),
                adjacency=tuple(tuple(row) for row in adjacency),
            )
        )
    return graph_slices


def _track_status(capability: dict[str, Any], dataset: str, metric: str) -> str:
    try:
        status = capability["datasets"][dataset]["metrics"][metric]["status"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Capability matrix is missing {dataset}.{metric}") from error
    if status not in {"EVALUABLE", "NOT_EVALUABLE"}:
        raise ValueError(f"Invalid capability status for {dataset}.{metric}: {status}")
    return status


def build_model_eligibility_matrix(capability: dict[str, Any]) -> dict[str, Any]:
    """Separate evidence eligibility from model implementation and evaluation status."""
    tasks = (
        ("track_a1_detection", "veremi", "detection"),
        ("track_a1_attacker_emitter_localization", "veremi", "emitter_or_node_localization"),
        ("track_a1_simulated_message_latency", "veremi", "mttd"),
        ("track_a2_labeled_anomaly_detection", "pems_bay", "detection"),
        ("track_b_cross_modal_fusion", "tampa_cv", "cross_modal_fusion"),
        ("causal_root_cause_localization", "veremi", "causal_rca"),
        ("work_zone_false_alarm_rate", "caltrans_lcs", "work_zone_far"),
    )
    model_rows = []
    for definition in _MODEL_DEFINITIONS:
        eligibility = {}
        for task_name, dataset, metric in tasks:
            status = _track_status(capability, dataset, metric)
            eligibility[task_name] = (
                "DATA_ELIGIBLE"
                if status == "EVALUABLE" and definition["inputs_available_in_veremi"]
                else "NOT_EVALUABLE"
            )
        model_rows.append(
            {
                **definition,
                "training_status": (
                    "not_applicable_for_fixed_rules"
                    if definition["name"] == "robust_threshold_rules"
                    else "not_trained"
                ),
                "evaluation_status": "not_evaluated_under_track_a1_protocol",
                "data_task_eligibility": eligibility,
            }
        )
    return {
        "schema_version": 1,
        "evidence_source": "results/dataset_capability_matrix.json",
        "dataset_capability_matrix_sha256": None,
        "scope": (
            "Eligibility reflects available labels and input modalities only. It does "
            "not imply that a model is trained, benchmarked, or validated."
        ),
        "models": model_rows,
        "task_notes": {
            "track_a1_detection": (
                "Simulated VeReMi misbehavior detection using official attacker labels."
            ),
            "track_a1_attacker_emitter_localization": (
                "Emitter identification from per-message labels and sender IDs; this "
                "is not causal corridor root-cause localization."
            ),
            "track_a1_simulated_message_latency": (
                "Only latency from a labeled simulator attacker transition is eligible; "
                "it is not field-incident MTTD."
            ),
            "causal_root_cause_localization": (
                "Not evaluable without causal ground-truth root-cause identifiers."
            ),
            "work_zone_false_alarm_rate": (
                "Not evaluable without time-aligned work-zone and traffic observations."
            ),
            "track_b_cross_modal_fusion": (
                "Not evaluable without temporally and spatially aligned modalities."
            ),
        },
    }


def write_model_eligibility_matrix(
    capability_path: Path = Path("results/dataset_capability_matrix.json"),
    output_path: Path = Path("results/model_eligibility_matrix.json"),
) -> dict[str, Any]:
    capability_path = Path(capability_path)
    output_path = Path(output_path)
    capability = json.loads(capability_path.read_text(encoding="utf-8"))
    if not isinstance(capability, dict):
        raise ValueError("Dataset capability matrix must contain a JSON object")
    matrix = build_model_eligibility_matrix(capability)
    matrix["dataset_capability_matrix_sha256"] = hashlib.sha256(
        capability_path.read_bytes()
    ).hexdigest()
    benchmark_path = capability_path.parent / "benchmark_summary.json"
    if benchmark_path.exists():
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        if not isinstance(benchmark, dict) or not isinstance(benchmark.get("models"), dict):
            raise ValueError("Benchmark summary must contain a models object")
        benchmark_models = benchmark["models"]
        training = benchmark.get("training", {})
        if not isinstance(training, dict):
            raise ValueError("Benchmark training metadata must be an object")
        training_statuses = training.get("sequence_models", {})
        if not isinstance(training_statuses, dict):
            raise ValueError("Benchmark sequence_models metadata must be an object")
        matrix["benchmark_summary_sha256"] = hashlib.sha256(
            benchmark_path.read_bytes()
        ).hexdigest()
        for model in matrix["models"]:
            name = model["name"]
            if name in benchmark_models:
                model["evaluation_status"] = "evaluated_on_official_veremi_splits"
                model["evaluation_reference"] = "results/benchmark_summary.json"
                status_record = training_statuses.get(name)
                if isinstance(status_record, dict):
                    model["implementation_status"] = "trained_sequence_baseline"
                    training_status = status_record.get("training_status")
                elif isinstance(status_record, str):
                    model["implementation_status"] = "trained_sequence_baseline"
                    training_status = status_record
                elif status_record is None:
                    training_status = None
                else:
                    raise ValueError(f"Invalid sequence model training status for {name}")
                model["training_status"] = (
                    "fixed_rule_scored"
                    if name == "robust_threshold_rules"
                    else "fit_on_official_benign_training_split"
                )
                if training_status is not None:
                    model["training_status"] = training_status
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(matrix, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return matrix
