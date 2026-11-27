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
_EVENT_SCORING = "aggregate_event_anomaly_scores_by_sender"


def _definition(
    name: str,
    regime: str,
    status: str,
    model_input: str,
    *,
    available: bool = True,
    scoring: str = _EVENT_SCORING,
) -> dict[str, Any]:
    return {
        "name": name,
        "supervision_regime": regime,
        "implementation_status": status,
        "input": model_input,
        "inputs_available_in_veremi": available,
        "emitter_scoring": scoring,
    }


_UNSUP = "unsupervised_or_one_class"
_SUP = "supervised"
_MODEL_DEFINITIONS = (
    _definition("robust_threshold_rules", _UNSUP, "implemented_baseline", "current_step_features"),
    _definition("isolation_forest", _UNSUP, "implemented_baseline", "current_step_features"),
    _definition("one_class_svm", _UNSUP, "implemented_baseline", "current_step_features"),
    _definition("lstm_autoencoder", _UNSUP, "implemented_baseline", "causal_sequences"),
    _definition("logistic_regression", _SUP, "implemented_baseline", "flattened_causal_window"),
    _definition("random_forest", _SUP, "implemented_baseline", "flattened_causal_window"),
    _definition("hist_gradient_boosting", _SUP, "implemented_baseline", "flattened_causal_window"),
    _definition("xgboost", _SUP, "optional_baseline", "flattened_causal_window"),
    _definition("supervised_mlp", _SUP, "implemented_baseline", "flattened_causal_window"),
    _definition("tabular_transformer", _SUP, "implemented_baseline", "causal_sequences"),
    _definition(
        "corridor_log_transformer",
        _SUP,
        "architecture_only",
        "trajectory_sequences_and_controller_text_logs",
        available=False,
    ),
    _definition(
        "static_gcn",
        _SUP,
        "architecture_only",
        "trajectory_features_and_dynamic_v2x_graph",
        scoring="node_localization_head",
    ),
    _definition(
        "async_st_gnn",
        _SUP,
        "architecture_only",
        "causal_trajectory_sequences_and_dynamic_v2x_graph",
        scoring="node_localization_head",
    ),
)


@dataclass(frozen=True)
class V2XGraphSlice:
    """Time-local V2X neighborhood graph built from telemetry only.

    ``member_event_ids`` lists, per node, every event of that sender inside the slice,
    so labels can be attached afterwards with :func:`attach_attacker_labels` without
    the graph construction itself ever reading a label.
    """

    corridor_id: str
    slice_start_ns: int
    source_ids: tuple[str, ...]
    windows: tuple[TrajectoryWindow, ...]
    adjacency: tuple[tuple[bool, ...], ...]
    member_event_ids: tuple[tuple[str, ...], ...]


def build_v2x_graph_slices(
    events: list[CorridorEvent],
    *,
    radius_meters: float = 300.0,
    time_slice_seconds: float = 1.0,
) -> list[V2XGraphSlice]:
    """Create time-local communication-neighborhood graphs from BSM telemetry alone.

    No ground-truth field is read, so the same function serves unlabeled inference
    streams. Each node is represented by the sender's latest event in the slice.
    """
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
    latest: dict[tuple[str, int], dict[str, tuple[CorridorEvent, TrajectoryWindow]]] = defaultdict(
        dict
    )
    members: dict[tuple[str, int], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for event, window in zip(events, windows, strict=True):
        group_key = (event.corridor_id, _event_time_ns(event) // slice_width_ns)
        members[group_key][event.source_id].append(event.event_id)
        current = latest[group_key].get(event.source_id)
        if current is None or (_event_time_ns(event), event.event_id) > (
            _event_time_ns(current[0]),
            current[0].event_id,
        ):
            latest[group_key][event.source_id] = (event, window)

    graph_slices = []
    for (corridor_id, time_bin), senders in sorted(latest.items()):
        source_ids = tuple(sorted(senders))
        nodes = [senders[source_id] for source_id in source_ids]
        adjacency = [[False] * len(nodes) for _ in nodes]
        positions = [_position_xy(event) for event, _ in nodes]
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
        group_members = members[(corridor_id, time_bin)]
        graph_slices.append(
            V2XGraphSlice(
                corridor_id=corridor_id,
                slice_start_ns=time_bin * slice_width_ns,
                source_ids=source_ids,
                windows=tuple(window for _, window in nodes),
                adjacency=tuple(tuple(row) for row in adjacency),
                member_event_ids=tuple(
                    tuple(sorted(group_members[source_id])) for source_id in source_ids
                ),
            )
        )
    return graph_slices


def attach_attacker_labels(
    graph_slices: list[V2XGraphSlice],
    events: list[CorridorEvent],
) -> list[tuple[bool, ...]]:
    """Attach supervised node labels after label-free graph construction.

    A node is attacker-labeled when any of its member events in the slice carries the
    published VeReMi ``cyber`` label. Raises if a member event is unlabeled.
    """
    labels_by_event = {event.event_id: event.label for event in events}
    attached = []
    for graph_slice in graph_slices:
        node_labels = []
        for event_ids in graph_slice.member_event_ids:
            labels = [labels_by_event.get(event_id) for event_id in event_ids]
            if any(label not in {"benign", "cyber"} for label in labels):
                raise ValueError(
                    f"Slice {graph_slice.corridor_id}@{graph_slice.slice_start_ns} has "
                    "a member event without a supported VeReMi label"
                )
            node_labels.append(any(label == "cyber" for label in labels))
        attached.append(tuple(node_labels))
    return attached


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
        not_run = benchmark.get("models_not_run", {})
        matrix["benchmark_summary_sha256"] = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
        for model in matrix["models"]:
            name = model["name"]
            if name in benchmark_models:
                record = benchmark_models[name]
                if not isinstance(record, dict):
                    raise ValueError(f"Invalid benchmark record for {name}")
                per_seed = record.get("per_seed") or [{}]
                model["evaluation_status"] = "evaluated_on_official_veremi_splits"
                model["evaluation_reference"] = "results/benchmark_summary.json"
                model["training_status"] = per_seed[0].get("training_status", "unknown")
                model["training_seeds"] = len(per_seed)
            elif name in not_run:
                model["evaluation_status"] = "not_run"
                model["not_run_reason"] = not_run[name]
            if model["implementation_status"] == "architecture_only" and (
                model["evaluation_status"] != "not_evaluated_under_track_a1_protocol"
            ):
                raise ValueError(f"Architecture-only model {name} cannot be marked evaluated")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(matrix, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return matrix
