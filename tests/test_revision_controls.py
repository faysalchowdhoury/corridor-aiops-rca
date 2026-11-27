"""Tests for the peer-review controls: benign overlap audit, sender-disjoint LOAO,
chronological holdout, matched L = 10 one-class controls, and threshold procedures."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from corridor_aiops.data.trajectory import FEATURE_NAMES, RAW_COORDINATE_FEATURE_NAMES
from corridor_aiops.evaluation.benchmark import (
    FULL_CONFIG,
    ONE_CLASS_L10_MODELS,
    EventTable,
    build_representation,
    chronological_partition,
    fit_model,
    regime,
)
from corridor_aiops.evaluation.sensitivity import (
    benign_overlap_audit,
    overlap_counts,
    procedure_threshold,
    sender_disjoint_loao_partitions,
    youden_threshold,
)

NAMES = FEATURE_NAMES + RAW_COORDINATE_FEATURE_NAMES


def _table(
    archives: list[str],
    splits: list[str],
    sources: list[str],
    labels: list[int],
    times: list[int],
    broadcast_ids: list[str] | None = None,
    values: np.ndarray | None = None,
) -> EventTable:
    count = len(archives)
    ids = broadcast_ids or [
        f"{s}:{i}:{t}" for i, (s, t) in enumerate(zip(sources, times, strict=True))
    ]
    events = [
        SimpleNamespace(broadcast_id=b, receive_time_ns=t, corridor_id=f"S_{a}-{s}")
        for a, s, b, t in zip(archives, splits, ids, times, strict=True)
    ]
    if values is None:
        values = np.random.default_rng(0).normal(size=(count, 10, len(NAMES))).astype(np.float32)
    return EventTable(
        events=events,
        values=values,
        feature_mask=np.ones((count, 10, len(NAMES)), dtype=bool),
        step_mask=np.ones((count, 10), dtype=bool),
        labels=np.asarray(labels, dtype=np.int64),
        corridor=np.asarray([f"S_{a}-{s}" for a, s in zip(archives, splits, strict=True)]),
        source=np.asarray(sources),
        archive=np.asarray(archives),
        split=np.asarray(splits),
        feature_names=NAMES,
    )


def _two_archive_table(per_split: int = 24) -> EventTable:
    archives, splits, sources, labels, times = [], [], [], [], []
    for archive in ("dataReplay", "dosAttack"):
        for split_index, split in enumerate(("train", "validation", "test")):
            for index in range(per_split):
                archives.append(archive)
                splits.append(split)
                sources.append(f"veh_{index % 12}")
                labels.append(int(index % 12 < 4 and index % 2 == 0))
                times.append(split_index * 10_000 + index * 10)
    return _table(archives, splits, sources, labels, times)


# (a) cross-archive benign overlap audit --------------------------------------


def test_overlap_audit_counts_identical_benign_messages_across_archives() -> None:
    table = _table(
        ["dataReplay", "dataReplay", "dosAttack", "dosAttack"],
        ["train"] * 4,
        ["veh_1", "veh_2", "veh_1", "veh_3"],
        [0, 0, 0, 0],
        [10, 20, 10, 30],
        broadcast_ids=["veh_1:1:5", "veh_2:2:6", "veh_1:1:5", "veh_3:3:7"],
        values=np.zeros((4, 10, len(NAMES)), dtype=np.float32),
    )

    audit = benign_overlap_audit(table, {})
    whole = audit["whole_archive_non_malicious"]

    assert whole["message_sender_messageID_sendTime"]["shared"] == 1
    assert whole["receive_time_sender_rcvTime"]["shared"] == 1
    assert whole["bare_source_id"]["shared"] == 1
    assert whole["trajectory_window_L10"]["shared"] == 1  # all-zero windows coincide


def test_overlap_audit_ignores_attacker_records() -> None:
    table = _table(
        ["dataReplay", "dosAttack"],
        ["train", "train"],
        ["veh_1", "veh_1"],
        [1, 1],
        [10, 10],
        broadcast_ids=["veh_1:1:5", "veh_1:1:5"],
    )

    whole = benign_overlap_audit(table, {})["whole_archive_non_malicious"]

    assert whole["message_sender_messageID_sendTime"]["left_unique"] == 0
    assert whole["message_sender_messageID_sendTime"]["shared"] == 0


# (b) sender-disjoint deduplicated LOAO ----------------------------------------


def test_sender_disjoint_loao_has_no_shared_senders_or_messages() -> None:
    table = _two_archive_table()

    partitions, meta = sender_disjoint_loao_partitions(table)

    assert set(partitions) == {"dataReplay->dosAttack", "dosAttack->dataReplay"}
    assert meta["source_side_senders"] + meta["target_side_senders"] == 12
    for direction, partition in partitions.items():
        source, target = direction.split("->")
        fit = np.concatenate((partition["train"], partition["validation"]))
        assert set(table.archive[fit]) == {source}
        assert set(table.archive[partition["test"]]) == {target}
        assert set(table.split[partition["test"]]) == {"test"}
        counts = overlap_counts(table, fit, partition["test"])
        assert counts["bare_source_id"]["shared"] == 0
        assert counts["message_sender_messageID_sendTime"]["shared"] == 0


def test_sender_disjoint_loao_is_deterministic_and_keeps_attackers_on_both_sides() -> None:
    table = _two_archive_table()

    first, _ = sender_disjoint_loao_partitions(table)
    second, _ = sender_disjoint_loao_partitions(table)

    for direction in first:
        for split in ("train", "validation", "test"):
            assert np.array_equal(first[direction][split], second[direction][split])
        fit = np.concatenate((first[direction]["train"], first[direction]["validation"]))
        assert table.labels[fit].sum() > 0
        assert table.labels[first[direction]["test"]].sum() > 0


# (c) chronological holdout -----------------------------------------------------


def test_chronological_partition_is_time_ordered_within_each_archive() -> None:
    count = 100
    rng = np.random.default_rng(1)
    times = rng.permutation(np.arange(count) * 1_000).tolist() * 2
    table = _table(
        ["dataReplay"] * count + ["dosAttack"] * count,
        ["train"] * (2 * count),
        [f"veh_{i % 7}" for i in range(2 * count)],
        [i % 3 == 0 for i in range(2 * count)],
        times,
    )

    rows, meta = chronological_partition(table)
    t = np.asarray(times)

    for archive in ("dataReplay", "dosAttack"):
        local = {split: t[r][table.archive[r] == archive] for split, r in rows.items()}
        assert local["train"].max() < local["validation"].min()
        assert local["validation"].max() < local["test"].min()
        assert len(local["train"]) == 60 and len(local["test"]) == 20
    assert meta["timestamp_overlap_between_partitions"] == 0


def test_chronological_partition_never_splits_a_shared_timestamp() -> None:
    times = [0] * 50 + [1] * 50
    table = _table(["dataReplay"] * 100, ["train"] * 100, ["v"] * 100, [0, 1] * 50, times)

    with pytest.raises(RuntimeError, match="without sharing a timestamp"):
        chronological_partition(table)


# (d) matched L = 10 one-class controls --------------------------------------------


@pytest.mark.parametrize("name", ONE_CLASS_L10_MODELS)
def test_one_class_l10_controls_use_the_flattened_window(name: str) -> None:
    table = _two_archive_table(per_split=30)
    rows = {
        split: table.rows(archive="dataReplay", split=split)
        for split in ("train", "validation", "test")
    }
    train = rows["train"]
    representation = build_representation(table, FULL_CONFIG, train[table.labels[train] == 0])

    result = fit_model(name, representation, table.labels, rows, seed=7, epochs=1)

    assert regime(name) == "unsupervised_or_one_class"
    assert result["training_status"].endswith("flattened_L10_window_and_masks")
    assert result["test_scores"].shape == (len(rows["test"]),)
    assert np.isfinite(result["test_scores"]).all()


def test_one_class_l10_differs_from_current_step_input() -> None:
    table = _two_archive_table(per_split=30)
    rows = {
        split: table.rows(archive="dataReplay", split=split)
        for split in ("train", "validation", "test")
    }
    train = rows["train"]
    representation = build_representation(table, FULL_CONFIG, train[table.labels[train] == 0])

    windowed = fit_model(
        "isolation_forest_L10", representation, table.labels, rows, seed=7, epochs=1
    )
    current = fit_model("isolation_forest", representation, table.labels, rows, seed=7, epochs=1)

    assert not np.allclose(windowed["test_scores"], current["test_scores"])


# (e) threshold-procedure comparison -------------------------------------------------


def test_youden_threshold_maximizes_tpr_minus_fpr() -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9])

    threshold = youden_threshold(labels, scores)
    alerts = scores >= threshold
    j = alerts[labels == 1].mean() - alerts[labels == 0].mean()

    assert j == pytest.approx(0.75)


def test_procedure_thresholds_respect_their_definitions() -> None:
    rng = np.random.default_rng(3)
    labels = np.asarray([0] * 400 + [1] * 100)
    scores = np.concatenate((rng.normal(0, 1, 400), rng.normal(2, 1, 100)))

    for budget in (0.01, 0.05):
        threshold = procedure_threshold(f"far_{budget * 100:.1f}pct", labels, scores)
        assert np.mean(scores[labels == 0] >= threshold) <= budget + 1e-12
    max_f1 = procedure_threshold("max_f1", labels, scores)
    youden = procedure_threshold("youden_j", labels, scores)
    assert np.isfinite(max_f1) and np.isfinite(youden)
    with pytest.raises(ValueError, match="Unknown threshold procedure"):
        procedure_threshold("median", labels, scores)
