from __future__ import annotations

import json
from datetime import datetime, timezone

from corridor_aiops.data.audit import _capabilities
from corridor_aiops.data.tracks import (
    build_model_eligibility_matrix,
    build_v2x_graph_slices,
    write_model_eligibility_matrix,
)
from corridor_aiops.schema import CorridorEvent, EventKind


def _event(
    event_id: str,
    source_id: str,
    timestamp_ns: int,
    x: float,
    label: str,
    corridor_id: str = "scenario-train",
) -> CorridorEvent:
    return CorridorEvent(
        event_id=event_id,
        timestamp=datetime.fromtimestamp(timestamp_ns / 1e9, tz=timezone.utc),
        receive_time_ns=timestamp_ns,
        corridor_id=corridor_id,
        source_id=source_id,
        kind=EventKind.BSM,
        label=label,
        metrics={
            "speed": 10.0,
            "speed_valid": 1.0,
            "acceleration": 0.0,
            "acceleration_valid": 1.0,
            "heading": 0.0,
            "heading_valid": 1.0,
            "position_x": x,
            "position_y": 0.0,
            "position_z": 0.0,
            "position_xy_valid": 1.0,
            "position_z_valid": 1.0,
        },
    )


def _capability_matrix() -> dict:
    datasets = ("veremi", "pems_bay", "tampa_cv", "caltrans_lcs")
    return {
        "datasets": {
            dataset: {"metrics": _capabilities(dataset)}
            for dataset in datasets
        }
    }


def test_v2x_graph_slices_keep_splits_and_build_local_symmetric_edges() -> None:
    events = [
        _event("a-1", "a", 1_100_000_000, 0.0, "cyber"),
        _event("a-2", "a", 1_800_000_000, 0.0, "benign"),
        _event("b-1", "b", 1_200_000_000, 100.0, "benign"),
        _event("c-1", "c", 1_300_000_000, 500.0, "benign"),
        _event("a-test", "a", 1_100_000_000, 0.0, "benign", "scenario-test"),
    ]

    slices = build_v2x_graph_slices(events, radius_meters=150.0)
    by_corridor = {graph_slice.corridor_id: graph_slice for graph_slice in slices}
    train_slice = by_corridor["scenario-train"]

    assert set(by_corridor) == {"scenario-test", "scenario-train"}
    assert train_slice.source_ids == ("a", "b", "c")
    assert train_slice.attacker_labels == (True, False, False)
    assert train_slice.windows[0].event.event_id == "a-2"
    assert train_slice.adjacency[0][1] and train_slice.adjacency[1][0]
    assert not train_slice.adjacency[0][2]
    assert not train_slice.adjacency[1][2]


def test_model_eligibility_distinguishes_data_support_from_causal_rca() -> None:
    matrix = build_model_eligibility_matrix(_capability_matrix())
    models = {model["name"]: model for model in matrix["models"]}

    assert len(models) == 8
    for model in models.values():
        tasks = model["data_task_eligibility"]
        expected_a1 = (
            "NOT_EVALUABLE"
            if model["name"] == "corridor_log_transformer"
            else "DATA_ELIGIBLE"
        )
        assert tasks["track_a1_detection"] == expected_a1
        assert tasks["track_a1_attacker_emitter_localization"] == expected_a1
        assert tasks["causal_root_cause_localization"] == "NOT_EVALUABLE"
        assert tasks["work_zone_false_alarm_rate"] == "NOT_EVALUABLE"
        assert model["evaluation_status"] == "not_evaluated_under_track_a1_protocol"
    assert models["async_st_gnn"]["implementation_status"] == "architecture_only"
    assert models["tabular_transformer"]["implementation_status"] == "architecture_only"
    assert (
        models["corridor_log_transformer"]["data_task_eligibility"]["track_a1_detection"]
        == "NOT_EVALUABLE"
    )


def test_eligibility_writer_records_the_audit_digest(tmp_path) -> None:
    capability_path = tmp_path / "capability.json"
    benchmark_path = tmp_path / "benchmark_summary.json"
    output_path = tmp_path / "model_eligibility.json"
    capability_path.write_text(json.dumps(_capability_matrix()), encoding="utf-8")
    benchmark_path.write_text(
        json.dumps(
            {
                "models": {
                    "lstm_autoencoder": {},
                    "robust_threshold_rules": {},
                },
                "training": {
                    "sequence_models": {
                        "lstm_autoencoder": {
                            "training_status": (
                                "trained_on_non_malicious_official_training_sequences"
                            )
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = write_model_eligibility_matrix(capability_path, output_path)
    models = {model["name"]: model for model in result["models"]}

    assert result["dataset_capability_matrix_sha256"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == result
    assert models["lstm_autoencoder"]["implementation_status"] == "trained_sequence_baseline"
    assert models["lstm_autoencoder"]["evaluation_status"] == "evaluated_on_official_veremi_splits"
    assert (
        models["lstm_autoencoder"]["training_status"]
        == "trained_on_non_malicious_official_training_sequences"
    )
