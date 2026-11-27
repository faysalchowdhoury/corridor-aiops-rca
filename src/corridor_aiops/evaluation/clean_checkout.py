"""Record a clean-checkout reproduction of the headline metrics and Linux CI checks.

The repository is cloned into a temporary directory at a given commit, the
official-split benchmark (all models and training seeds, no generalization or
ablation contexts) is rerun from that clone, and every per-seed headline metric is
compared with the committed ``results/benchmark_summary.json``. ``ruff check .`` and
``pytest`` outcomes from a Linux (WSL) run of the same commit are recorded alongside.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from corridor_aiops.evaluation.tables import scientific

HEADLINE_METRICS = (
    "f1_cyber_vs_benign",
    "auroc",
    "average_precision",
    "false_alarm_rate_non_malicious",
    "test_roc_tpr_at_far_1.0pct",
    "test_roc_tpr_at_far_5.0pct",
    "validation_calibrated_recall_at_far_1.0pct",
    "validation_calibrated_recall_at_far_5.0pct",
)
TOLERANCE = 1e-6


def compare_headline(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    compared = 0
    over = 0
    max_delta = 0.0
    hashes_matched = 0
    hashes_compared = 0
    missing = []
    for name, block in reference["models"].items():
        if name not in candidate["models"]:
            missing.append(name)
            continue
        new_rows = candidate["models"][name]["per_seed"]
        for old, new in zip(block["per_seed"], new_rows, strict=True):
            hashes_compared += 1
            hashes_matched += int(old["checkpoint_sha256"] == new["checkpoint_sha256"])
            for key in HEADLINE_METRICS:
                delta = abs(float(old[key]) - float(new[key]))
                compared += 1
                over += int(delta > TOLERANCE)
                max_delta = max(max_delta, delta)
    reference_boot = reference["uncertainty"]["sender_cluster_bootstrap"]
    candidate_boot = candidate["uncertainty"]["sender_cluster_bootstrap"]
    boot_delta = max(
        abs(a - b)
        for name, metrics in reference_boot["intervals_95pct"].items()
        for metric, interval in metrics.items()
        for a, b in zip(interval, candidate_boot["intervals_95pct"][name][metric], strict=True)
    )
    return {
        "metrics": list(HEADLINE_METRICS),
        "tolerance": TOLERANCE,
        "per_seed_values_compared": compared,
        "values_over_tolerance": over,
        "max_abs_delta": max_delta,
        "checkpoint_hashes_matched": hashes_matched,
        "checkpoint_hashes_compared": hashes_compared,
        "bootstrap_interval_max_abs_delta": boot_delta,
        "models_missing": missing,
        "status": (
            "REPRODUCED_WITHIN_TOLERANCE"
            if over == 0 and not missing and boot_delta <= TOLERANCE
            else "DIFFERS"
        ),
    }


def _check(name: str, exit_code: int, log: Path) -> dict[str, Any]:
    lines = [line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        "command": name,
        "exit_code": exit_code,
        "status": "PASSED" if exit_code == 0 else "FAILED",
        "summary_line": lines[-1] if lines else "",
    }


def _count(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def write_sentence(audit: dict[str, Any], path: Path) -> None:
    head = audit["headline_reproduction"]
    checks = audit["linux_checks"]
    reproduced = head["status"] == "REPRODUCED_WITHIN_TOLERANCE"
    sentence = (
        "A clean clone of commit \\texttt{"
        + audit["commit"][:7]
        + "} in a temporary directory reran the official-split comparison for all "
        + f"{audit['models']} detectors and {audit['training_seeds']} training seeds from "
        + "checksum-verified raw files. "
        + (
            f"All {_count(head['per_seed_values_compared'])} per-seed headline values "
            f"matched the reported results within {head['tolerance']:.0e} "
            f"(maximum absolute difference {scientific(head['max_abs_delta'])}), "
            f"{head['checkpoint_hashes_matched']} of {head['checkpoint_hashes_compared']} "
            "checkpoint hashes matched, and the largest bootstrap-interval difference was "
            f"{scientific(head['bootstrap_interval_max_abs_delta'])}. "
            if reproduced
            else f"{head['values_over_tolerance']} of {head['per_seed_values_compared']} "
            "per-seed headline values differed by more than the tolerance; see "
            "\\texttt{results/clean\\_checkout\\_audit.json}. "
        )
        + "For commit \\texttt{"
        + audit["linux_commit"][:7]
        + "}, on Linux ("
        + f"{audit['linux_environment']['platform']}, Python "
        + f"{audit['linux_environment']['python']}), \\texttt{{ruff check .}} "
        + ("passed" if checks["ruff"]["status"] == "PASSED" else "failed")
        + " and \\texttt{pytest} "
        + (
            "passed (" + checks["pytest"]["summary_line"].strip("= ").replace("_", "\\_") + ")."
            if checks["pytest"]["status"] == "PASSED"
            else "failed; see \\texttt{results/clean\\_checkout\\_audit.json}."
        )
    )
    path.write_text(
        "% Generated by corridor_aiops.evaluation.clean_checkout; do not edit by hand.\n"
        f"\\newcommand{{\\ResCleanCheckoutSentence}}{{{sentence}}}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, default=Path("results/benchmark_summary.json"))
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--commit", required=True, help="commit used for the headline rerun")
    parser.add_argument("--linux-commit", required=True, help="commit used for ruff and pytest")
    parser.add_argument("--clone-dir", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--ruff-exit", type=int, required=True)
    parser.add_argument("--ruff-log", type=Path, required=True)
    parser.add_argument("--pytest-exit", type=int, required=True)
    parser.add_argument("--pytest-log", type=Path, required=True)
    parser.add_argument("--linux-platform", required=True)
    parser.add_argument("--linux-python", required=True)
    parser.add_argument("--output", type=Path, default=Path("results/clean_checkout_audit.json"))
    parser.add_argument("--macros", type=Path, default=Path("results/clean_checkout_macros.tex"))
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "procedure": (
            "git clone of the committed repository into a temporary directory; official-split "
            "benchmark rerun from the clone (PYTHONPATH set to the clone's src); per-seed "
            "headline metrics compared with the committed results; ruff and pytest run on "
            "Linux (WSL) from a separate clone of the same commit"
        ),
        "commit": args.commit,
        "linux_commit": args.linux_commit,
        "clone_dir": args.clone_dir,
        "command": args.command,
        "models": len(reference["models"]),
        "training_seeds": len(reference["protocol"]["training_seeds"]),
        "headline_reproduction": compare_headline(reference, candidate),
        "linux_environment": {"platform": args.linux_platform, "python": args.linux_python},
        "linux_checks": {
            "ruff": _check("ruff check .", args.ruff_exit, args.ruff_log),
            "pytest": _check("pytest", args.pytest_exit, args.pytest_log),
        },
        "scope_note": (
            "Covers the official-split headline comparison only; generalization and ablation "
            "contexts are covered by results/reproduction_audit.json. Not a bit-identical "
            "determinism claim."
        ),
    }
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    write_sentence(audit, args.macros)
    print(json.dumps(audit["headline_reproduction"], indent=2))


if __name__ == "__main__":
    main()
