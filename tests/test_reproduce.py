from __future__ import annotations

import json
from datetime import datetime, timezone

from corridor_aiops.data.trajectory import (
    FEATURE_NAMES,
    RAW_COORDINATE_FEATURE_NAMES,
    extract_trajectory_windows,
)
from corridor_aiops.evaluation.reproduce import compare_outputs
from corridor_aiops.schema import CorridorEvent, EventKind


def _write(directory, summary: dict, table: str) -> None:
    directory.mkdir()
    (directory / "benchmark_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (directory / "ablation_summary.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
    (directory / "results_table.tex").write_text(table, encoding="utf-8")


def _summary(auroc: float, checkpoint: str, seconds: float) -> dict:
    return {
        "models": {
            "m": {
                "per_seed": [
                    {"auroc": auroc, "checkpoint_sha256": checkpoint, "training_seconds": seconds}
                ]
            }
        },
        "runtime_seconds": seconds,
    }


def test_compare_outputs_ignores_timing_and_reports_identical_runs(tmp_path) -> None:
    _write(tmp_path / "first", _summary(0.9, "abc", 1.0), "x")
    _write(tmp_path / "second", _summary(0.9, "abc", 7.0), "x")

    result = compare_outputs(tmp_path / "first", tmp_path / "second")

    assert result["determinism_status"] == "BIT_IDENTICAL_EXCLUDING_WALL_CLOCK_TIMING"
    assert result["checkpoint_sha256"] == {"compared": 1, "identical": 1, "different": []}
    assert result["timing_fields_excluded"] == 2


def test_compare_outputs_reports_metric_and_checkpoint_differences(tmp_path) -> None:
    _write(tmp_path / "first", _summary(0.9, "abc", 1.0), "x")
    _write(tmp_path / "second", _summary(0.8, "def", 1.0), "y")

    result = compare_outputs(tmp_path / "first", tmp_path / "second")

    assert result["determinism_status"] == "NOT_BIT_IDENTICAL"
    assert result["max_abs_numeric_delta"] == 0.09999999999999998
    assert len(result["checkpoint_sha256"]["different"]) == 1
    assert result["derived_file_sha256"]["results_table.tex"]["identical"] is False


def test_raw_coordinate_features_are_opt_in_and_appended() -> None:
    event = CorridorEvent(
        event_id="e",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        receive_time_ns=1,
        corridor_id="c",
        source_id="s",
        kind=EventKind.BSM,
        metrics={
            "position_x": 12.0,
            "position_y": 34.0,
            "position_xy_valid": 1.0,
            "position_z": 0.5,
            "position_z_valid": 1.0,
            "heading": 90.0,
            "heading_valid": 1.0,
        },
    )

    default = extract_trajectory_windows([event])[0]
    extended = extract_trajectory_windows([event], include_raw_coordinates=True)[0]

    assert len(default.values[-1]) == len(FEATURE_NAMES)
    assert len(extended.values[-1]) == len(FEATURE_NAMES) + len(RAW_COORDINATE_FEATURE_NAMES)
    assert extended.values[-1][: len(FEATURE_NAMES)] == default.values[-1]
    assert extended.values[-1][len(FEATURE_NAMES) :] == (12.0, 34.0, 0.5, 90.0)


def test_incomplete_audit_records_attempts_and_makes_no_claim(tmp_path) -> None:
    from corridor_aiops.evaluation.reproduce import record_incomplete_audit

    record_incomplete_audit(tmp_path, "the first reason", detail="a")
    audit = record_incomplete_audit(tmp_path, "the second reason", detail="b")
    sentence = (tmp_path / "reproduction_macros.tex").read_text(encoding="utf-8")

    assert audit["determinism_status"] == "NOT_COMPLETED"
    assert [item["outcome"] for item in audit["attempts"]] == [
        "the first reason",
        "the second reason",
    ]
    assert "no claim of bit-identical determinism" in sentence
    assert r"\newcommand{\ResReproSentence}" in sentence


def test_completed_audit_sentence_reports_measured_counts(tmp_path) -> None:
    from corridor_aiops.evaluation.reproduce import write_reproduction_macros

    write_reproduction_macros(
        {
            "determinism_status": "BIT_IDENTICAL_EXCLUDING_WALL_CLOCK_TIMING",
            "numeric_values_compared": 12345,
            "numeric_values_with_nonzero_delta": 0,
            "max_abs_numeric_delta": 0.0,
            "checkpoint_sha256": {"compared": 7, "identical": 7, "different": []},
            "derived_file_sha256": {"a.csv": {"identical": True}},
        },
        tmp_path / "macros.tex",
    )
    sentence = (tmp_path / "macros.tex").read_text(encoding="utf-8")

    assert "0 of 12{,}345 compared numeric values differed" in sentence
    assert "7 of 7 checkpoint SHA-256 hashes matched" in sentence
