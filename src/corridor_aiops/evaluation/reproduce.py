"""Clean second-run reproduction audit.

Runs the benchmark again in a fresh Python process into ``results_repro_check/``,
compares every numeric value and checkpoint hash with ``results/``, writes
``results/reproduction_audit.json``, and deletes the temporary directory.
Determinism is reported only from this measured comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TIMING_KEYS = frozenset({"inference_ms_per_event", "training_seconds", "runtime_seconds"})
NUMERICAL_TOLERANCE = 1e-6
MODEL_NAMES = (
    "robust_threshold_rules",
    "isolation_forest",
    "one_class_svm",
    "lstm_autoencoder",
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
    "xgboost",
    "supervised_mlp",
    "tabular_transformer",
)
COMPARED_JSON = ("benchmark_summary.json", "ablation_summary.json")
COMPARED_FILES = (
    "seed_metrics.csv",
    "bootstrap_preview.csv",
    "results_table.tex",
    "results_operating_points.tex",
    "results_bootstrap.tex",
    "results_generalization.tex",
    "results_ablation.tex",
    "results_macros.tex",
)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        items: dict[str, Any] = {}
        for key, child in value.items():
            items.update(_flatten(child, f"{prefix}.{key}" if prefix else str(key)))
        return items
    if isinstance(value, list):
        items = {}
        for index, child in enumerate(value):
            items.update(_flatten(child, f"{prefix}[{index}]"))
        return items
    return {prefix: value}


def _is_timing(path: str) -> bool:
    leaf = path.rsplit(".", 1)[-1].split("[", 1)[0]
    return leaf in TIMING_KEYS or any(f".{key}." in f".{path}." for key in TIMING_KEYS)


def compare_outputs(reference_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    numeric_deltas: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    checkpoints = {"compared": 0, "identical": 0, "different": []}
    compared_numbers = 0
    excluded_timing = 0
    max_abs_delta = 0.0
    for name in COMPARED_JSON:
        reference_path = Path(reference_dir) / name
        candidate_path = Path(candidate_dir) / name
        if not reference_path.exists() or not candidate_path.exists():
            mismatches.append({"file": name, "issue": "missing in one run"})
            continue
        reference = _flatten(json.loads(reference_path.read_text(encoding="utf-8")))
        candidate = _flatten(json.loads(candidate_path.read_text(encoding="utf-8")))
        for path in sorted(set(reference) | set(candidate)):
            if path not in reference or path not in candidate:
                mismatches.append({"file": name, "path": path, "issue": "key missing"})
                continue
            left, right = reference[path], candidate[path]
            if _is_timing(path):
                excluded_timing += 1
                continue
            if path.endswith("checkpoint_sha256"):
                checkpoints["compared"] += 1
                if left == right:
                    checkpoints["identical"] += 1
                else:
                    checkpoints["different"].append({"file": name, "path": path})
                continue
            if (
                isinstance(left, (int, float))
                and isinstance(right, (int, float))
                and not (isinstance(left, bool) or isinstance(right, bool))
            ):
                compared_numbers += 1
                delta = abs(float(left) - float(right))
                if not math.isfinite(delta) or delta > 0.0:
                    max_abs_delta = max(max_abs_delta, delta)
                    numeric_deltas.append(
                        {
                            "file": name,
                            "path": path,
                            "first": left,
                            "second": right,
                            "abs_delta": delta,
                        }
                    )
            elif left != right:
                mismatches.append({"file": name, "path": path, "first": left, "second": right})
    file_hashes = {}
    for name in COMPARED_FILES:
        paths = [Path(reference_dir) / name, Path(candidate_dir) / name]
        if not all(path.exists() for path in paths):
            continue
        digests = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
        file_hashes[name] = {
            "first_sha256": digests[0],
            "second_sha256": digests[1],
            "identical": digests[0] == digests[1],
        }
    identical = (
        not numeric_deltas
        and not mismatches
        and not checkpoints["different"]
        and checkpoints["compared"] > 0
        and all(item["identical"] for item in file_hashes.values())
    )
    return {
        "determinism_status": (
            "BIT_IDENTICAL_EXCLUDING_WALL_CLOCK_TIMING" if identical else "NOT_BIT_IDENTICAL"
        ),
        "numeric_values_compared": compared_numbers,
        "numeric_values_with_nonzero_delta": len(numeric_deltas),
        "max_abs_numeric_delta": max_abs_delta,
        "nonzero_deltas": numeric_deltas[:500],
        "nonzero_deltas_truncated": len(numeric_deltas) > 500,
        "non_numeric_mismatches": mismatches[:200],
        "checkpoint_sha256": {
            "compared": checkpoints["compared"],
            "identical": checkpoints["identical"],
            "different": checkpoints["different"][:200],
        },
        "derived_file_sha256": file_hashes,
        "timing_fields_excluded": excluded_timing,
        "timing_exclusion_rule": sorted(TIMING_KEYS),
    }


def run_reproduction_audit(
    *,
    results_dir: Path = Path("results"),
    check_dir: Path = Path("results_repro_check"),
    data_dir: Path = Path("data/raw/veremi"),
    keep: bool = False,
) -> dict[str, Any]:
    results_dir = Path(results_dir)
    check_dir = Path(check_dir)
    summary = json.loads((results_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    protocol = summary["protocol"]
    bootstrap = summary["uncertainty"]["sender_cluster_bootstrap"]
    command = [
        sys.executable,
        "-m",
        "corridor_aiops.evaluation.benchmark",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(check_dir),
        "--train-seeds",
        str(len(protocol["training_seeds"])),
        "--bootstrap-samples",
        str(bootstrap["requested_replicates"]),
        "--bootstrap-preview-rows",
        str(protocol["bootstrap_preview_rows"]),
        "--epochs",
        str(protocol["epochs"]),
    ]
    if not summary.get("generalization"):
        command.append("--skip-generalization")
    if not (results_dir / "ablation_summary.json").exists():
        command.append("--skip-ablations")
    if check_dir.exists():
        shutil.rmtree(check_dir)
    started = time.perf_counter()
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        tail = completed.stderr[-4000:]
        reason = (
            "the second run ran out of system memory"
            if "MemoryError" in tail
            else f"the second run exited with code {completed.returncode}"
        )
        record_incomplete_audit(results_dir, reason, detail=tail[-1500:])
        if check_dir.exists() and not keep:
            shutil.rmtree(check_dir)
        raise RuntimeError("Second benchmark run failed:\n" + tail)
    comparison = compare_outputs(results_dir, check_dir)
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "procedure": (
            "independent second run of the full benchmark in a fresh Python process "
            "from the same raw data into results_repro_check/, then value-by-value "
            "comparison against results/"
        ),
        "command": [Path(command[0]).name, *command[1:]],
        "second_run_wall_clock_seconds": elapsed,
        **comparison,
        "numerical_tolerance": NUMERICAL_TOLERANCE,
        "within_numerical_tolerance": comparison["max_abs_numeric_delta"] <= NUMERICAL_TOLERANCE,
        "temporary_directory_deleted": not keep,
        "scope": (
            "Same machine, same interpreter, same library versions, CPU only. The status "
            "does not establish determinism across hardware or library versions."
        ),
    }
    try:
        import numpy
        import sklearn
        import torch

        audit["environment"] = {
            "python": sys.version.split()[0],
            "numpy": numpy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "platform": sys.platform,
        }
    except ImportError:  # pragma: no cover
        pass
    (results_dir / "reproduction_audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_reproduction_macros(audit, results_dir / "reproduction_macros.tex")
    if not keep:
        shutil.rmtree(check_dir)
    return audit


def _write_sentence(path: Path, sentence: str) -> None:
    lines = [
        "% Generated by corridor_aiops.evaluation.reproduce; do not edit by hand.",
        f"\\newcommand{{\\ResReproSentence}}{{{sentence}}}",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reproduction_macros(audit: dict[str, Any], path: Path) -> None:
    """Emit the manuscript sentence that states only the measured determinism result."""
    checkpoints = audit["checkpoint_sha256"]

    def count(value: int) -> str:
        return f"{value:,}".replace(",", "{,}")

    files_identical = sum(item["identical"] for item in audit["derived_file_sha256"].values())
    tolerance = audit.get("numerical_tolerance", NUMERICAL_TOLERANCE)
    within = audit["max_abs_numeric_delta"] <= tolerance
    affected = sorted(
        {
            name
            for name in MODEL_NAMES
            for item in audit.get("nonzero_deltas", [])
            if name in item["path"]
        }
    )
    status = audit["determinism_status"].replace("_", " ").lower()
    sentence = (
        "The complete benchmark was run a second time in a fresh process from the same raw "
        "files into a temporary directory and compared value by value with the reported "
        "outputs, excluding wall-clock timing fields. The measured status was "
        f"\\emph{{{status}}}: {count(audit['numeric_values_with_nonzero_delta'])} of "
        f"{count(audit['numeric_values_compared'])} compared numeric values differed (maximum "
        f"absolute difference {audit['max_abs_numeric_delta']:.3g}), "
        f"{checkpoints['identical']} of {checkpoints['compared']} checkpoint SHA-256 hashes "
        f"matched, and {files_identical} of {len(audit['derived_file_sha256'])} derived tables "
        "and CSV files were byte-identical. "
        + (
            f"Every difference is below the predefined numerical tolerance of {tolerance:g}, so "
            "the outputs are numerically reproducible within that tolerance but not "
            "bit-identical. "
            if within
            else f"Some differences exceed the predefined tolerance of {tolerance:g}. "
        )
        + (
            f"All differing values belong to {', '.join(a.replace('_', ' ') for a in affected)}, "
            "whose checkpoints were identical, which is consistent with floating-point "
            "summation order across parallel threads at prediction time. "
            if affected and checkpoints["identical"] == checkpoints["compared"]
            else ""
        )
        + "The comparison covers one machine and one library "
        "environment, recorded in \\texttt{results/reproduction\\_audit.json}."
    )
    _write_sentence(path, sentence)


def record_incomplete_audit(
    results_dir: Path, reason: str, *, detail: str = "", attempts: list[dict] | None = None
) -> dict[str, Any]:
    """Record an audit that did not complete; no determinism claim is made."""
    results_dir = Path(results_dir)
    audit_path = results_dir / "reproduction_audit.json"
    history: list[dict] = []
    if audit_path.exists():
        previous = json.loads(audit_path.read_text(encoding="utf-8"))
        if previous.get("determinism_status") == "NOT_COMPLETED":
            history = previous.get("attempts", [])
    history.extend(attempts or [])
    history.append(
        {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "outcome": reason,
            "detail": detail,
        }
    )
    audit = {
        "determinism_status": "NOT_COMPLETED",
        "determinism_claim": "none; no completed second-run comparison exists",
        "attempts": history,
        "how_to_complete": "run `corridor-reproduce` with sufficient free memory",
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    sentence = (
        "A complete second run of the benchmark in a fresh process was attempted to audit "
        f"reproducibility, but it did not finish because {reason}. No second-run comparison "
        "is therefore available, and this paper makes no claim of bit-identical determinism. "
        "The attempts are recorded in \\texttt{results/reproduction\\_audit.json}, and "
        "\\texttt{corridor-reproduce} performs the audit when sufficient memory is available."
    )
    _write_sentence(results_dir / "reproduction_macros.tex", sentence)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean second-run reproduction audit.")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--check-dir", type=Path, default=Path("results_repro_check"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/veremi"))
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    audit = run_reproduction_audit(
        results_dir=args.results, check_dir=args.check_dir, data_dir=args.data_dir, keep=args.keep
    )
    print(
        json.dumps(
            {
                key: audit[key]
                for key in (
                    "determinism_status",
                    "numeric_values_compared",
                    "numeric_values_with_nonzero_delta",
                    "max_abs_numeric_delta",
                    "checkpoint_sha256",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
