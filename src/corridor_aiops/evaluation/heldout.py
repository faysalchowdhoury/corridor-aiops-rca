"""Held-out scenario evaluation on an independently simulated VeReMi NextGen scenario.

Every model is fit, normalized, checkpoint-selected, and thresholded on the InTAS
highway 2 official train and validation splits exactly as in the main benchmark, then
applied unchanged to the official *test* split of a different InTAS scenario
(default ``InTAS_urban_2``). Nothing is fit, tuned, or selected on the held-out
scenario. Its archives are verified against the Zenodo MD5 checksums by
``corridor-ingest --heldout-only`` and its test-split labels pass the same official
ground-truth consistency audit as the main data.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from corridor_aiops.data.parse import (
    cross_check_veremi_ground_truth,
    veremi_archive_identity,
    veremi_events,
    veremi_ground_truth_paths,
)
from corridor_aiops.evaluation.benchmark import (
    FULL_CONFIG,
    _context_bootstrap,
    _log,
    _split_counts,
    build_event_table,
    run_models,
    split_eligibility,
    summarize_seeds,
)
from corridor_aiops.evaluation.sensitivity import confusion
from corridor_aiops.evaluation.tables import (
    GROUPS,
    MACRO_MODEL_NAMES,
    _ci,
    _escape,
    _ms,
    _write,
    scientific,
)

THRESHOLD_CHECK_TOLERANCE = 1e-6


def scenario_of(corridor_ids: np.ndarray) -> np.ndarray:
    """``InTAS_urban_2_dosAttack-test`` -> ``InTAS_urban_2``."""
    return np.asarray(
        [corridor.rsplit("-", 1)[0].rpartition("_")[0] for corridor in corridor_ids.tolist()]
    )


def load_combined(data_dir: Path, heldout_dir: Path):
    events: list = []
    archive_of_event: list[str] = []
    record_counts: dict[str, int] = {}
    for directory, splits in ((data_dir, None), (heldout_dir, ("test",))):
        archives = sorted(Path(directory).glob("*.zip"))
        if not archives:
            raise FileNotFoundError(f"No VeReMi archives in {directory}")
        for archive in archives:
            _, category = veremi_archive_identity(archive)
            parsed = list(veremi_events(archive, splits=splits))
            record_counts[archive.name] = len(parsed)
            events.extend(parsed)
            archive_of_event.extend([category] * len(parsed))
            _log(f"parsed {archive.name}: {len(parsed)} de-duplicated messages")
    table = build_event_table(events, archive_of_event)
    return table, scenario_of(table.corridor), record_counts


def _verify_against_main(summary: dict[str, Any], heldout_models: dict[str, Any]) -> dict[str, Any]:
    """Thresholds and validation AUROC depend only on highway 2 data and must match."""
    compared = 0
    max_delta = 0.0
    for name, block in heldout_models.items():
        for new, old in zip(block["per_seed"], summary["models"][name]["per_seed"], strict=True):
            for key in ("validation_f1_threshold", "validation_auroc"):
                compared += 1
                max_delta = max(max_delta, abs(float(new[key]) - float(old[key])))
    return {
        "values_compared": compared,
        "max_abs_delta": max_delta,
        "tolerance": THRESHOLD_CHECK_TOLERANCE,
        "status": "MATCH" if max_delta <= THRESHOLD_CHECK_TOLERANCE else "MISMATCH",
        "meaning": (
            "validation-F1 thresholds and validation AUROC of every model and seed equal "
            "the main benchmark values, so the held-out scenario did not influence fitting, "
            "checkpoint selection, or thresholds"
        ),
    }


def run_heldout(
    *,
    data_dir: Path,
    heldout_dir: Path,
    results_dir: Path,
    bootstrap_samples: int | None = None,
) -> dict[str, Any]:
    import torch

    results_dir = Path(results_dir)
    summary = json.loads((results_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    protocol = summary["protocol"]
    torch.set_num_threads(int(protocol.get("torch_threads") or os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    seeds = list(protocol["training_seeds"])
    models = list(summary["models"])
    samples = (
        bootstrap_samples
        or summary["uncertainty"]["sender_cluster_bootstrap"]["requested_replicates"]
    )

    manifest_path = Path(heldout_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audits = []
    for archive in sorted(Path(heldout_dir).glob("*.zip")):
        scenario, _ = veremi_archive_identity(archive)
        _log(f"ground-truth audit (test split): {archive.name}")
        audits.append(
            cross_check_veremi_ground_truth(
                archive, veremi_ground_truth_paths(Path(heldout_dir), scenario), splits=("test",)
            )
        )

    _log("loading main and held-out scenarios")
    table, scenario, record_counts = load_combined(data_dir, heldout_dir)
    main_scenario = summary["dataset"].replace("VeReMi NextGen ", "").replace(" ", "_")
    heldout_scenarios = sorted(set(scenario.tolist()) - {main_scenario})
    if len(heldout_scenarios) != 1:
        raise ValueError(f"Expected one held-out scenario, found {heldout_scenarios}")
    heldout = heldout_scenarios[0]
    in_main = scenario == main_scenario
    split_rows = {
        "train": np.flatnonzero(in_main & (table.split == "train")),
        "validation": np.flatnonzero(in_main & (table.split == "validation")),
        "test": np.flatnonzero((scenario == heldout) & (table.split == "test")),
    }
    reason = split_eligibility(table.labels, split_rows)
    if reason is not None:
        raise ValueError(f"Held-out evaluation is not evaluable: {reason}")
    if set(table.split[split_rows["test"]].tolist()) != {"test"}:
        raise ValueError("Held-out rows must come from the official test split only")

    summaries, raw, _ = run_models(
        table,
        split_rows,
        models,
        config=FULL_CONFIG,
        seeds=seeds,
        epochs=int(protocol["epochs"]),
        label=f"heldout {heldout}",
    )
    verification = _verify_against_main(summary, summaries)
    _log(f"heldout: threshold verification {verification['status']}")

    labels = table.labels
    test = split_rows["test"]
    categories = sorted(set(table.archive[test].tolist()))
    per_category: dict[str, Any] = {}
    for category in categories:
        mask = table.archive[test] == category
        block: dict[str, Any] = {
            "test_records": int(mask.sum()),
            "test_attackers": int(labels[test][mask].sum()),
            "protocol": (
                "pooled highway 2 models and their highway 2 validation-F1 thresholds; "
                "held-out test restricted to this attack category"
            ),
            "models": {},
        }
        for name in models:
            rows = []
            for index, run in enumerate(raw[name]):
                threshold = summaries[name]["per_seed"][index]["validation_f1_threshold"]
                scores = run["test_scores"][mask]
                rows.append(
                    {
                        "auroc": float(roc_auc_score(labels[test][mask], scores)),
                        **confusion(labels[test][mask], scores, threshold),
                    }
                )
            block["models"][name] = {
                "per_seed": rows,
                "mean_std_across_training_seeds": summarize_seeds(rows),
            }
        per_category[category] = block

    _log("heldout: composite sender-cluster bootstrap")
    bootstrap = _context_bootstrap(table, split_rows, raw, summaries, samples=samples)
    bootstrap.pop("_preview_rows")
    pooled_confusion = {}
    for name in models:
        rows = [
            confusion(
                labels[test],
                run["test_scores"],
                summaries[name]["per_seed"][index]["validation_f1_threshold"],
            )
            for index, run in enumerate(raw[name])
        ]
        pooled_confusion[name] = summarize_seeds(rows)
    for name in models:
        summaries[name]["confusion_mean_std_across_training_seeds"] = pooled_confusion[name]
        summaries[name]["in_scenario_reference"] = {
            key: summary["models"][name]["mean_std_across_training_seeds"][key]
            for key in ("f1_cyber_vs_benign", "auroc", "average_precision")
        }

    result = {
        "question": (
            "do detectors fit and thresholded on InTAS highway 2 keep their performance on "
            "an independently simulated scenario of the same release?"
        ),
        "status": "EVALUABLE",
        "training_scenario": main_scenario,
        "heldout_scenario": heldout,
        "protocol": (
            "fit, robust normalization, neural checkpoint selection, and validation-F1 "
            f"thresholds use only {main_scenario} official train and validation splits; "
            f"models are applied unchanged to the {heldout} official test split; no "
            "training, tuning, or model selection uses the held-out scenario"
        ),
        "heldout_files": manifest,
        "heldout_ground_truth_audit": audits,
        "heldout_record_counts_test_split": {
            name: count for name, count in record_counts.items() if heldout in name
        },
        "class_counts": _split_counts(labels, split_rows),
        "heldout_test_composite_clusters": int(
            len(set(zip(table.corridor[test].tolist(), table.source[test].tolist(), strict=True)))
        ),
        "threshold_verification": verification,
        "models": summaries,
        "per_attack_category": per_category,
        "composite_cluster_bootstrap": bootstrap,
        "not_evaluated": {
            scenario: (
                "NOT_EVALUATED: published in the same Zenodo record with dataReplay and "
                f"dosAttack archives totalling about {size}, roughly {ratio} times the "
                "evaluated highway 2 archives; not acquired in this revision because the "
                "evaluation machine had 7.7 GB of RAM and about 3 GB of free disk at "
                "acquisition time. No result is claimed for it."
            )
            for scenario, size, ratio in (
                ("InTAS_highway_7", "1.9 GB", 46),
                ("InTAS_urban_7", "2.3 GB", 55),
            )
        },
    }
    (results_dir / "heldout_scenario.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_heldout_tables(summary, result, results_dir)
    return result


def write_heldout_tables(
    summary: dict[str, Any], result: dict[str, Any], results_dir: Path
) -> None:
    categories = list(result["per_attack_category"])
    body = [
        f"& \\multicolumn{{4}}{{c}}{{Held-out \\texttt{{{_escape(result['heldout_scenario'])}}} "
        "test (pooled)} & "
        + " & ".join(f"\\texttt{{{c}}}" for c in categories)
        + " & In-scenario \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-" + str(5 + len(categories)) + "}",
        "Model & F1 & AUROC & Recall & FAR (\\%) & "
        + " & ".join("AUROC" for _ in categories)
        + " & AUROC \\\\",
    ]
    columns = 6 + len(categories)
    for group, title in GROUPS:
        body += ["\\midrule", f"\\multicolumn{{{columns}}}{{l}}{{\\textit{{{title}}}}} \\\\"]
        for name, block in result["models"].items():
            if block["supervision_regime"] != group:
                continue
            stats = block["mean_std_across_training_seeds"]
            conf = block["confusion_mean_std_across_training_seeds"]
            cells = [
                _ms(stats, "f1_cyber_vs_benign"),
                _ms(stats, "auroc"),
                f"{conf['recall']['mean']:.3f}",
                f"{100 * conf['false_alarm_rate']['mean']:.2f}",
            ]
            for category in categories:
                cat = result["per_attack_category"][category]["models"][name]
                cells.append(_ms(cat["mean_std_across_training_seeds"], "auroc"))
            cells.append(f"{block['in_scenario_reference']['auroc']['mean']:.3f}")
            body.append(f"{_escape(block['display_name'])} & " + " & ".join(cells) + " \\\\")
    _write(results_dir / "results_heldout.tex", "l" + "c" * (columns - 1), columns, body)

    lines = ["% Generated by corridor_aiops.evaluation.heldout; do not edit by hand."]

    def define(name: str, value: Any) -> None:
        if not name.isalpha():
            raise ValueError(f"Invalid LaTeX macro name: {name}")
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    def count(value: int) -> str:
        return f"{value:,}".replace(",", "{,}")

    define("ResHeldoutScenario", _escape(result["heldout_scenario"]))
    counts = result["class_counts"]["test"]
    define("ResHeldoutRecords", count(counts["records"]))
    define("ResHeldoutAttacker", count(counts["attacker_1_misbehavior"]))
    define("ResHeldoutBenign", count(counts["attacker_0_non_malicious"]))
    define("ResHeldoutClusters", count(result["heldout_test_composite_clusters"]))
    define("ResHeldoutThresholdStatus", result["threshold_verification"]["status"])
    define(
        "ResHeldoutThresholdMaxDelta",
        scientific(result["threshold_verification"]["max_abs_delta"]),
    )
    bootstrap = result["composite_cluster_bootstrap"]
    define("ResHeldoutBootValid", count(bootstrap["valid_replicates"]))
    define("ResHeldoutBootRequested", count(bootstrap["requested_replicates"]))
    for category, block in result["per_attack_category"].items():
        tag = {"dataReplay": "Replay", "dosAttack": "DoS"}.get(category, category)
        define(f"ResHeldout{tag}Records", count(block["test_records"]))
        define(f"ResHeldout{tag}Attackers", count(block["test_attackers"]))
        for name, model in block["models"].items():
            stats = model["mean_std_across_training_seeds"]
            define(f"ResHeldout{tag}{MACRO_MODEL_NAMES[name]}AUROC", _ms(stats, "auroc"))
    for name, block in result["models"].items():
        tag = MACRO_MODEL_NAMES[name]
        stats = block["mean_std_across_training_seeds"]
        conf = block["confusion_mean_std_across_training_seeds"]
        define(f"ResHeldout{tag}AUROC", _ms(stats, "auroc"))
        define(f"ResHeldout{tag}FOne", _ms(stats, "f1_cyber_vs_benign"))
        define(f"ResHeldout{tag}Recall", f"{conf['recall']['mean']:.3f}")
        define(f"ResHeldout{tag}FAR", f"{100 * conf['false_alarm_rate']['mean']:.2f}")
        define(f"ResHeldout{tag}AUROCCI", _ci(bootstrap["intervals_95pct"][name]["auroc"]))
        delta = bootstrap["paired_differences_reference_minus_model"].get(name)
        if delta:
            define(f"ResHeldout{tag}DeltaAUROCCI", _ci(delta["delta_auroc"]["ci_95pct"]))
            define(
                f"ResHeldout{tag}DeltaFOneCI", _ci(delta["delta_f1_cyber_vs_benign"]["ci_95pct"])
            )
    (results_dir / "heldout_macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/veremi"))
    parser.add_argument("--heldout-dir", type=Path, default=Path("data/raw/veremi_heldout"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument(
        "--tables-only", action="store_true", help="rewrite tables from heldout_scenario.json"
    )
    args = parser.parse_args()
    if args.tables_only:
        summary = json.loads(
            (args.results_dir / "benchmark_summary.json").read_text(encoding="utf-8")
        )
        result = json.loads(
            (args.results_dir / "heldout_scenario.json").read_text(encoding="utf-8")
        )
        write_heldout_tables(summary, result, args.results_dir)
        return
    run_heldout(
        data_dir=args.data_dir,
        heldout_dir=args.heldout_dir,
        results_dir=args.results_dir,
        bootstrap_samples=args.bootstrap_samples,
    )
    _log("held-out scenario evaluation finished")


if __name__ == "__main__":
    main()
