from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from corridor_aiops.data.audit import _capabilities
from corridor_aiops.data.tracks import (
    attach_attacker_labels,
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
    return {"datasets": {dataset: {"metrics": _capabilities(dataset)} for dataset in datasets}}


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
    train_index = [graph_slice.corridor_id for graph_slice in slices].index("scenario-train")
    assert attach_attacker_labels(slices, events)[train_index] == (True, False, False)
    assert train_slice.windows[0].event.event_id == "a-2"
    assert train_slice.adjacency[0][1] and train_slice.adjacency[1][0]
    assert not train_slice.adjacency[0][2]
    assert not train_slice.adjacency[1][2]


def test_model_eligibility_distinguishes_data_support_from_causal_rca() -> None:
    matrix = build_model_eligibility_matrix(_capability_matrix())
    models = {model["name"]: model for model in matrix["models"]}

    assert len(models) == 13
    for model in models.values():
        tasks = model["data_task_eligibility"]
        expected_a1 = (
            "NOT_EVALUABLE" if model["name"] == "corridor_log_transformer" else "DATA_ELIGIBLE"
        )
        assert tasks["track_a1_detection"] == expected_a1
        assert tasks["track_a1_attacker_emitter_localization"] == expected_a1
        assert tasks["causal_root_cause_localization"] == "NOT_EVALUABLE"
        assert tasks["work_zone_false_alarm_rate"] == "NOT_EVALUABLE"
        assert model["evaluation_status"] == "not_evaluated_under_track_a1_protocol"
    for name in ("async_st_gnn", "static_gcn", "corridor_log_transformer"):
        assert models[name]["implementation_status"] == "architecture_only"
    assert models["lstm_autoencoder"]["supervision_regime"] == "unsupervised_or_one_class"
    assert models["random_forest"]["supervision_regime"] == "supervised"


def _write_inputs(tmp_path, benchmark: dict):
    capability_path = tmp_path / "capability.json"
    capability_path.write_text(json.dumps(_capability_matrix()), encoding="utf-8")
    (tmp_path / "benchmark_summary.json").write_text(json.dumps(benchmark), encoding="utf-8")
    return capability_path, tmp_path / "model_eligibility.json"


def test_eligibility_writer_records_the_audit_digest(tmp_path) -> None:
    capability_path, output_path = _write_inputs(
        tmp_path,
        {
            "models": {
                "lstm_autoencoder": {
                    "per_seed": [
                        {"training_status": "trained_on_non_malicious_training_sequences"},
                        {"training_status": "trained_on_non_malicious_training_sequences"},
                    ]
                },
                "robust_threshold_rules": {"per_seed": [{"training_status": "fixed_rule"}]},
            },
            "models_not_run": {"xgboost": "NOT_RUN: the xgboost package is not installed"},
        },
    )

    result = write_model_eligibility_matrix(capability_path, output_path)
    models = {model["name"]: model for model in result["models"]}

    assert result["dataset_capability_matrix_sha256"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == result
    assert models["lstm_autoencoder"]["evaluation_status"] == "evaluated_on_official_veremi_splits"
    assert models["lstm_autoencoder"]["training_seeds"] == 2
    assert models["xgboost"]["evaluation_status"] == "not_run"
    assert models["async_st_gnn"]["evaluation_status"] == "not_evaluated_under_track_a1_protocol"


def test_eligibility_writer_refuses_to_mark_architecture_only_models_evaluated(tmp_path) -> None:
    capability_path, output_path = _write_inputs(
        tmp_path, {"models": {"async_st_gnn": {"per_seed": [{}]}}}
    )

    with pytest.raises(ValueError, match="Architecture-only"):
        write_model_eligibility_matrix(capability_path, output_path)
