"""P0 regression tests required before any full experiment is run."""

from __future__ import annotations

import asyncio
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from corridor_aiops.data.audit import composite_sender_cluster_audit
from corridor_aiops.data.parse import (
    cross_check_veremi_ground_truth,
    veremi_events,
    veremi_ground_truth_paths,
)
from corridor_aiops.data.tracks import attach_attacker_labels, build_v2x_graph_slices
from corridor_aiops.evaluation.benchmark import (
    _WeightedScorer,
    composite_cluster_bootstrap,
    composite_cluster_codes,
)
from corridor_aiops.schema import CorridorEvent, EventKind
from corridor_aiops.streaming.pipeline import run_pipeline

# ---------------------------------------------------------------------------
# 1. Composite (corridor_id, source_id) bootstrap with colliding source_id
# ---------------------------------------------------------------------------


def test_composite_cluster_codes_separate_colliding_source_ids_across_archives() -> None:
    corridors = ["replay-test", "replay-test", "dos-test", "dos-test", "replay-test"]
    sources = ["veh_1", "veh_2", "veh_1", "veh_1", "veh_1"]

    codes, count = composite_cluster_codes(corridors, sources)

    assert count == 3  # bare source_id would give only 2 clusters
    assert codes[0] == codes[4]
    assert codes[0] != codes[2]
    assert codes[2] == codes[3]


def test_composite_bootstrap_resamples_composite_clusters_and_reports_replicates() -> None:
    labels = np.asarray([0, 1, 0, 1, 0, 1])
    corridors = np.asarray(["a-test", "a-test", "b-test", "b-test", "a-test", "a-test"])
    sources = np.asarray(["veh_1", "veh_1", "veh_1", "veh_1", "veh_2", "veh_2"])
    scores = {
        "tabular_transformer": np.asarray([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]),
        "baseline": np.asarray([0.4, 0.6, 0.5, 0.5, 0.6, 0.4]),
    }

    result = composite_cluster_bootstrap(
        labels,
        corridors,
        sources,
        scores,
        {"tabular_transformer": 0.5, "baseline": 0.5},
        reference="tabular_transformer",
        seed=3,
        samples=200,
        preview_rows=2,
    )

    assert result["composite_clusters"] == 3
    assert result["valid_replicates"] + result["skipped_replicates"] == 200
    assert result["valid_replicates"] > 0
    delta = result["paired_differences_reference_minus_model"]["baseline"]
    assert set(delta) == {"delta_f1_cyber_vs_benign", "delta_auroc"}
    assert delta["delta_auroc"]["ci_95pct"][0] <= delta["delta_auroc"]["ci_95pct"][1]
    assert "tabular_transformer" not in result["paired_differences_reference_minus_model"]
    units = {row["resampling_unit"] for row in result["_preview_rows"]}
    assert units == {"composite_corridor_id_source_id_cluster"}


def test_weighted_scorer_matches_sklearn_with_ties_and_weights() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=300)
    scores = np.round(rng.random(300), 1)  # heavy ties
    weights = rng.integers(0, 4, size=300).astype(float)

    result = _WeightedScorer(labels, scores, 0.5)(weights)

    assert result is not None
    assert result["auroc"] == pytest.approx(
        roc_auc_score(labels, scores, sample_weight=weights), abs=1e-12
    )
    assert result["f1_cyber_vs_benign"] == pytest.approx(
        f1_score(labels, scores >= 0.5, sample_weight=weights), abs=1e-12
    )


def test_audit_counts_cross_archive_source_id_collisions() -> None:
    audit = composite_sender_cluster_audit(
        {
            "replay.zip": {"train": {"veh_1", "veh_2"}, "test": {"veh_1", "veh_3"}},
            "dos.zip": {"train": {"veh_1"}, "test": {"veh_3", "veh_4"}},
        }
    )

    assert audit["unique_composite_clusters_by_split"] == {"test": 4, "train": 3}
    assert audit["cross_archive_source_id_collisions_by_split"] == {"test": 1, "train": 1}
    overlap = audit["train_test_sender_overlap_by_archive"]
    assert overlap["replay.zip"]["train_test_same_archive_source_id_overlap"] == 1
    assert overlap["dos.zip"]["train_test_same_archive_source_id_overlap"] == 0


# ---------------------------------------------------------------------------
# 2. Streaming pipeline: consumer failure with maxsize=1 must not deadlock
# ---------------------------------------------------------------------------


def _stream_event(index: int) -> CorridorEvent:
    return CorridorEvent(
        event_id=f"e{index}",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        corridor_id="c",
        source_id="s",
        kind=EventKind.BSM,
    )


def test_consumer_exception_propagates_immediately_with_full_bounded_queue() -> None:
    produced: list[int] = []

    async def endless_source():
        index = 0
        while True:
            produced.append(index)
            yield _stream_event(index)
            index += 1

    async def failing_handler(event: CorridorEvent) -> None:
        raise RuntimeError(f"handler failed on {event.event_id}")

    async def scenario() -> None:
        await asyncio.wait_for(
            run_pipeline(endless_source(), failing_handler, max_queue_size=1), timeout=5.0
        )

    with pytest.raises(RuntimeError, match="handler failed on e0"):
        asyncio.run(scenario())
    assert len(produced) <= 4  # producer was cancelled, not left blocked or running


def test_producer_exception_propagates_and_cancels_consumer() -> None:
    async def broken_source():
        yield _stream_event(0)
        raise ValueError("source broke")

    async def slow_handler(event: CorridorEvent) -> None:
        await asyncio.sleep(10)

    async def scenario() -> None:
        await asyncio.wait_for(
            run_pipeline(broken_source(), slow_handler, max_queue_size=1), timeout=5.0
        )

    with pytest.raises(ValueError, match="source broke"):
        asyncio.run(scenario())


# ---------------------------------------------------------------------------
# 3. Official ground truth versus embedded attacker / attack-type labels
# ---------------------------------------------------------------------------


def _record(sender: str, message: int, attacker: int, x: float) -> dict:
    return {
        "rcvTime": 1_000_000_000 + message,
        "sendTime": 900_000_000 + message,
        "sender_id": sender,
        "messageID": message,
        "attacker": attacker,
        "sender": {"pos": f"{x},0.0,0.0", "spd": 10.0, "acl": 0.0, "hed": 0.0},
    }


def _ground_truth(sender: str, message: int, x: float, **extra) -> dict:
    return {
        "rcvTime": str(1_000_000_000 + message),
        "sendTime": str(900_000_000 + message),
        "sender_id": sender,
        "messageID": str(message),
        "pos": f"{x},0.0,0.0",
        "spd": "10.0",
        **extra,
    }


def _write_archive(
    directory: Path,
    archive_category: str,
    member_category: str,
    records: dict[str, list[dict]],
) -> Path:
    outer_buffer = io.BytesIO()
    with zipfile.ZipFile(outer_buffer, "w") as outer:
        for split_dir, split_records in records.items():
            inner_buffer = io.BytesIO()
            with zipfile.ZipFile(inner_buffer, "w") as inner:
                inner.writestr("veh_receiver.json", json.dumps(split_records))
            outer.writestr(
                f"S_{member_category}/{split_dir}/S_{member_category}.zip",
                inner_buffer.getvalue(),
            )
    path = directory / f"S_{archive_category}.zip"
    path.write_bytes(outer_buffer.getvalue())
    return path


def _write_ground_truth(directory: Path, test_rows: list[dict]) -> None:
    (directory / "S_Train_groundTruth.json").write_text(
        json.dumps([_ground_truth("veh_1", 1, 5.0)]), encoding="utf-8"
    )
    (directory / "S_Validation_groundTruth.json").write_text(
        json.dumps([_ground_truth("veh_1", 2, 6.0)]), encoding="utf-8"
    )
    (directory / "S_Test_ground_truth.json").write_text(json.dumps(test_rows), encoding="utf-8")


def _split_records(test_benign_x: float) -> dict[str, list[dict]]:
    return {
        "Train": [_record("veh_1", 1, 0, 5.0)],
        "Validation": [_record("veh_1", 2, 0, 6.0)],
        "Test": [_record("veh_1", 3, 0, test_benign_x), _record("veh_9", 4, 1, 99.0)],
    }


def test_ground_truth_cross_check_passes_and_reports_attacker_outcomes(tmp_path) -> None:
    archive = _write_archive(tmp_path, "dataReplay", "dataReplay", _split_records(7.0))
    _write_ground_truth(tmp_path, [_ground_truth("veh_1", 3, 7.0), _ground_truth("veh_9", 4, 1.0)])

    result = cross_check_veremi_ground_truth(archive, veremi_ground_truth_paths(tmp_path, "S"))

    assert result["status"] == "PASSED"
    assert result["attack_category"] == "dataReplay"
    assert result["record_outcomes_by_split"]["test"] == {
        "attacker_0_payload_matches": 1,
        "attacker_1_payload_differs": 1,
    }
    assert len(result["cross_check_sha256"]) == 64


def test_ground_truth_cross_check_rejects_benign_payload_mismatch(tmp_path) -> None:
    archive = _write_archive(tmp_path, "dataReplay", "dataReplay", _split_records(8.5))
    _write_ground_truth(tmp_path, [_ground_truth("veh_1", 3, 7.0)])

    with pytest.raises(ValueError, match="attacker=0 record payload_differs"):
        cross_check_veremi_ground_truth(archive, veremi_ground_truth_paths(tmp_path, "S"))


def test_ground_truth_cross_check_rejects_attack_type_mismatch(tmp_path) -> None:
    archive = _write_archive(tmp_path, "dosAttack", "dataReplay", _split_records(7.0))
    _write_ground_truth(tmp_path, [_ground_truth("veh_1", 3, 7.0)])

    with pytest.raises(ValueError, match="Attack-type mismatch"):
        cross_check_veremi_ground_truth(archive, veremi_ground_truth_paths(tmp_path, "S"))


def test_veremi_events_split_filter_reads_only_selected_split(tmp_path) -> None:
    archive = _write_archive(tmp_path, "dataReplay", "dataReplay", _split_records(7.0))

    everything = list(veremi_events(archive))
    test_only = list(veremi_events(archive, splits=("test",)))

    assert {event.corridor_id.rsplit("-", 1)[-1] for event in everything} == {
        "train",
        "validation",
        "test",
    }
    assert {event.corridor_id.rsplit("-", 1)[-1] for event in test_only} == {"test"}
    assert [e.event_id for e in test_only] == [
        e.event_id for e in everything if e.corridor_id.endswith("-test")
    ]


def test_ground_truth_cross_check_can_audit_the_test_split_alone(tmp_path) -> None:
    archive = _write_archive(tmp_path, "dataReplay", "dataReplay", _split_records(7.0))
    (tmp_path / "S_Test_ground_truth.json").write_text(
        json.dumps([_ground_truth("veh_1", 3, 7.0), _ground_truth("veh_9", 4, 1.0)]),
        encoding="utf-8",
    )

    paths = veremi_ground_truth_paths(tmp_path, "S")
    assert set(paths) == {"test"}
    with pytest.raises(ValueError, match="Missing official ground-truth files"):
        cross_check_veremi_ground_truth(archive, paths)
    result = cross_check_veremi_ground_truth(archive, paths, splits=("test",))

    assert result["status"] == "PASSED"
    assert result["splits_checked"] == ["test"]
    assert set(result["record_outcomes_by_split"]) == {"test"}


def test_full_ground_truth_audit_digest_has_no_split_restriction_field(tmp_path) -> None:
    archive = _write_archive(tmp_path, "dataReplay", "dataReplay", _split_records(7.0))
    _write_ground_truth(tmp_path, [_ground_truth("veh_1", 3, 7.0), _ground_truth("veh_9", 4, 1.0)])

    result = cross_check_veremi_ground_truth(archive, veremi_ground_truth_paths(tmp_path, "S"))

    assert "splits_checked" not in result


def test_ground_truth_cross_check_rejects_explicit_attacker_label_mismatch(tmp_path) -> None:
    archive = _write_archive(tmp_path, "dataReplay", "dataReplay", _split_records(7.0))
    _write_ground_truth(
        tmp_path,
        [_ground_truth("veh_1", 3, 7.0, attacker="1"), _ground_truth("veh_9", 4, 1.0)],
    )

    with pytest.raises(ValueError, match="ground-truth attacker label differs"):
        cross_check_veremi_ground_truth(archive, veremi_ground_truth_paths(tmp_path, "S"))


# ---------------------------------------------------------------------------
# 4. Transformer padding/causal masks and masked LSTM reconstruction
# ---------------------------------------------------------------------------


def test_corridor_log_transformer_ignores_padding_and_is_causal() -> None:
    torch = pytest.importorskip("torch")
    from corridor_aiops.models.transformer import CorridorLogTransformer

    torch.manual_seed(0)
    model = CorridorLogTransformer(3, 20, hidden_size=16, heads=2, layers=2).eval()
    telemetry = torch.randn(2, 5, 3)
    tokens = torch.tensor([[4, 5, 6, 0, 0], [7, 8, 9, 10, 11]])

    with torch.no_grad():
        base = model(telemetry, tokens)
        padded = telemetry.clone()
        padded[0, 3:] = 1000.0
        changed_padding = model(padded, tokens)
        future = telemetry.clone()
        future[1, 4] = -50.0
        changed_future = model(future, tokens)

    assert torch.allclose(base["logits"][0], changed_padding["logits"][0], atol=1e-5)
    assert torch.allclose(
        base["hidden_states"][1, :4], changed_future["hidden_states"][1, :4], atol=1e-5
    )
    assert not torch.allclose(base["hidden_states"][1, 4], changed_future["hidden_states"][1, 4])
    assert torch.count_nonzero(base["hidden_states"][0, 3:]) == 0


def test_lstm_autoencoder_masked_reconstruction_ignores_padded_garbage() -> None:
    torch = pytest.importorskip("torch")
    from corridor_aiops.models.lstm_autoencoder import (
        LSTMAutoencoder,
        masked_reconstruction_error,
    )

    torch.manual_seed(0)
    model = LSTMAutoencoder(3, hidden_size=8).eval()
    sequence = torch.randn(1, 4, 3)
    valid = torch.ones(1, 4, 3, dtype=torch.bool)
    valid[0, :2] = False  # left padding
    valid[0, 3, 1] = False  # one unobserved feature
    garbage = sequence.clone()
    garbage[0, :2] = 500.0
    garbage[0, 3, 1] = -500.0

    with torch.no_grad():
        clean_score = masked_reconstruction_error(sequence, model(sequence, valid), valid)
        garbage_score = masked_reconstruction_error(garbage, model(garbage, valid), valid)

    assert torch.allclose(clean_score, garbage_score, atol=1e-6)
    reconstruction = torch.zeros(1, 4, 3)
    target = torch.zeros(1, 4, 3)
    target[0, 2, 0] = 2.0
    error = masked_reconstruction_error(target, reconstruction, valid)
    assert error.item() == pytest.approx(4.0 / 5.0)


# ---------------------------------------------------------------------------
# 5. Label-free V2X graph construction for inference
# ---------------------------------------------------------------------------


def _bsm(event_id: str, source: str, time_ns: int, x: float, label: str | None) -> CorridorEvent:
    return CorridorEvent(
        event_id=event_id,
        timestamp=datetime.fromtimestamp(time_ns / 1e9, tz=timezone.utc),
        receive_time_ns=time_ns,
        corridor_id="live-stream",
        source_id=source,
        kind=EventKind.BSM,
        label=label,
        metrics={
            "speed": 10.0,
            "speed_valid": 1.0,
            "position_x": x,
            "position_y": 0.0,
            "position_xy_valid": 1.0,
        },
    )


def test_graph_slices_build_from_unlabeled_telemetry() -> None:
    events = [
        _bsm("a", "veh_a", 1_100_000_000, 0.0, None),
        _bsm("b", "veh_b", 1_200_000_000, 50.0, None),
        _bsm("c", "veh_c", 1_300_000_000, 900.0, None),
    ]

    slices = build_v2x_graph_slices(events, radius_meters=100.0)

    assert len(slices) == 1
    assert slices[0].source_ids == ("veh_a", "veh_b", "veh_c")
    assert slices[0].adjacency[0][1] and not slices[0].adjacency[0][2]
    assert slices[0].member_event_ids == (("a",), ("b",), ("c",))
    assert not hasattr(slices[0], "attacker_labels")
    with pytest.raises(ValueError, match="without a supported VeReMi label"):
        attach_attacker_labels(slices, events)


def test_graph_labels_attach_separately_for_supervised_training() -> None:
    events = [
        _bsm("a1", "veh_a", 1_100_000_000, 0.0, "cyber"),
        _bsm("a2", "veh_a", 1_500_000_000, 0.0, "benign"),
        _bsm("b", "veh_b", 1_200_000_000, 50.0, "benign"),
    ]
    unlabeled = [event.model_copy(update={"label": None}) for event in events]

    labeled_graph = build_v2x_graph_slices(events, radius_meters=100.0)
    unlabeled_graph = build_v2x_graph_slices(unlabeled, radius_meters=100.0)

    assert labeled_graph[0].adjacency == unlabeled_graph[0].adjacency
    assert labeled_graph[0].member_event_ids == unlabeled_graph[0].member_event_ids
    assert attach_attacker_labels(unlabeled_graph, events) == [(True, False)]


# ---------------------------------------------------------------------------
# Leakage and determinism audits (second-round review, P0.3)
# ---------------------------------------------------------------------------


def _tie_bsm(
    event_id: str, source_id: str, nanos: int, x: float, *, broadcast: str | None = None
) -> CorridorEvent:
    return CorridorEvent(
        event_id=event_id,
        timestamp=datetime(2022, 1, 1, tzinfo=timezone.utc),
        corridor_id="trace",
        source_id=source_id,
        kind=EventKind.BSM,
        label="benign",
        metrics={
            "position_x": x,
            "position_y": 0.0,
            "position_z": 0.0,
            "position_xy_valid": 1.0,
            "position_z_valid": 1.0,
            "speed": 10.0 + x / 100.0,
            "speed_valid": 1.0,
            "acceleration": 0.0,
            "acceleration_valid": 1.0,
            "heading": 0.0,
            "heading_valid": 1.0,
        },
        broadcast_id=broadcast,
        receive_time_ns=nanos,
    )


def _features_by_event(events: list[CorridorEvent]) -> dict[str, tuple]:
    from corridor_aiops.data.trajectory import extract_trajectory_windows

    return {
        window.event.event_id: (window.values, window.feature_mask)
        for window in extract_trajectory_windows(events)
    }


def test_local_density_is_invariant_to_input_permutation_under_timestamp_ties() -> None:
    # four senders share one nanosecond timestamp; two of them also share a later one
    events = [
        _tie_bsm("e1", "veh_a", 1_000, 0.0, broadcast="veh_a:1:1"),
        _tie_bsm("e2", "veh_b", 1_000, 10.0, broadcast="veh_b:1:1"),
        _tie_bsm("e3", "veh_c", 1_000, 20.0, broadcast="veh_c:1:1"),
        _tie_bsm("e4", "veh_d", 1_000, 400.0, broadcast="veh_d:1:1"),
        _tie_bsm("e5", "veh_a", 2_000, 1.0, broadcast="veh_a:2:2"),
        _tie_bsm("e6", "veh_b", 2_000, 11.0, broadcast="veh_b:2:2"),
    ]
    reference = _features_by_event(events)
    rng = np.random.default_rng(7)
    for _ in range(10):
        permuted = [events[i] for i in rng.permutation(len(events))]
        assert _features_by_event(permuted) == reference


def test_future_events_do_not_change_past_windows() -> None:
    past = [
        _tie_bsm(f"p{i}", sender, 1_000_000 * i, 5.0 * i + offset)
        for i in range(1, 6)
        for sender, offset in (("veh_a", 0.0), ("veh_b", 30.0))
    ]
    future = [
        _tie_bsm(f"f{i}", sender, 1_000_000 * (10 + i), 5.0 * i + offset)
        for i in range(1, 4)
        for sender, offset in (("veh_a", 0.0), ("veh_b", 30.0))
    ]
    mutated = [
        _tie_bsm(event.event_id, event.source_id, event.receive_time_ns, 9_999.0)
        for event in future
    ]

    original = _features_by_event(past + future)
    changed = _features_by_event(past + mutated)

    for event in past:
        assert original[event.event_id] == changed[event.event_id]
    assert any(original[e.event_id] != changed[e.event_id] for e in future)


def test_parser_keeps_earliest_receive_time_across_receiver_copies(tmp_path) -> None:
    broadcast = {
        "sendTime": 900_000_000,
        "sender_id": "veh_1",
        "messageID": 7,
        "attacker": 0,
        "sender": {"pos": "5.0,0.0,0.0", "spd": 10.0, "acl": 0.0, "hed": 0.0},
    }
    outer_buffer = io.BytesIO()
    with zipfile.ZipFile(outer_buffer, "w") as outer:
        inner_buffer = io.BytesIO()
        with zipfile.ZipFile(inner_buffer, "w") as inner:
            # three receivers heard the same broadcast; the earliest copy is in the middle
            inner.writestr("a_receiver.json", json.dumps([{**broadcast, "rcvTime": 1_000_500}]))
            inner.writestr("b_receiver.json", json.dumps([{**broadcast, "rcvTime": 1_000_100}]))
            inner.writestr("c_receiver.json", json.dumps([{**broadcast, "rcvTime": 1_000_900}]))
        outer.writestr("S_dataReplay/Train/S_dataReplay.zip", inner_buffer.getvalue())
    path = tmp_path / "S_dataReplay.zip"
    path.write_bytes(outer_buffer.getvalue())

    events = list(veremi_events(path))

    assert len(events) == 1
    assert events[0].receive_time_ns == 1_000_100
    assert events[0].broadcast_id == "veh_1:7:900000000"


@pytest.mark.parametrize("protocol", ["loao", "heldout"])
def test_normalization_and_thresholds_ignore_target_records(protocol: str) -> None:
    from types import SimpleNamespace

    from corridor_aiops.data.trajectory import FEATURE_NAMES
    from corridor_aiops.evaluation.benchmark import FULL_CONFIG, EventTable, run_models

    rng = np.random.default_rng(3)
    per_group = 60
    groups = (
        [("dataReplay", "train"), ("dataReplay", "validation"), ("dosAttack", "test")]
        if protocol == "loao"
        else [("highway", "train"), ("highway", "validation"), ("urban", "test")]
    )
    archive, split, labels = [], [], []
    for name, part in groups:
        archive += [name] * per_group
        split += [part] * per_group
        labels += [int(i % 4 == 0) for i in range(per_group)]
    count = len(labels)
    values = rng.normal(size=(count, 10, len(FEATURE_NAMES))).astype(np.float32)
    values[np.asarray(labels) == 1] += 1.5

    def table(array: np.ndarray, test_labels: np.ndarray) -> EventTable:
        all_labels = np.asarray(labels, dtype=np.int64).copy()
        all_labels[-per_group:] = test_labels
        return EventTable(
            events=[SimpleNamespace() for _ in range(count)],
            values=array,
            feature_mask=np.ones((count, 10, len(FEATURE_NAMES)), dtype=bool),
            step_mask=np.ones((count, 10), dtype=bool),
            labels=all_labels,
            corridor=np.asarray([f"{a}-{s}" for a, s in zip(archive, split, strict=True)]),
            source=np.asarray([f"veh_{i % 15}" for i in range(count)]),
            archive=np.asarray(archive),
            split=np.asarray(split),
            feature_names=FEATURE_NAMES,
        )

    split_rows = {
        "train": np.arange(0, per_group),
        "validation": np.arange(per_group, 2 * per_group),
        "test": np.arange(2 * per_group, 3 * per_group),
    }
    original_labels = np.asarray(labels[-per_group:])
    mutated = values.copy()
    mutated[split_rows["test"]] = rng.normal(50.0, 20.0, size=mutated[split_rows["test"]].shape)

    def run(array: np.ndarray, test_labels: np.ndarray):
        return run_models(
            table(array, test_labels),
            split_rows,
            ["logistic_regression", "isolation_forest"],
            config=FULL_CONFIG,
            seeds=[42],
            epochs=1,
            label="isolation test",
        )

    base_summary, _, base_rep = run(values, original_labels)
    new_summary, _, new_rep = run(mutated, 1 - original_labels)

    np.testing.assert_array_equal(base_rep.medians, new_rep.medians)
    np.testing.assert_array_equal(base_rep.scales, new_rep.scales)
    for name in base_summary:
        old = base_summary[name]["per_seed"][0]
        new = new_summary[name]["per_seed"][0]
        for key in ("validation_f1_threshold", "validation_auroc", "checkpoint_sha256"):
            assert old[key] == new[key], (name, key)
