"""Per-seed bootstrap sensitivity (R4) and attack-transfer diagnostics (R2).

Phase A (``--rerun``) refits the official-split and leave-one-attack-type-out (LOAO)
models for every training seed with the benchmark code, saves every validation and
test score to ``results/scores/per_seed_scores.npz``, and verifies each per-seed
metric against ``results/benchmark_summary.json``.

Phase B reads the saved scores and writes:

* ``results/seed_bootstrap_sensitivity.json``: the composite-cluster paired bootstrap
  repeated with each training seed's scores and thresholds, and whether each paired
  delta (Tabular Transformer minus model) keeps its sign and CI across seeds;
* ``results/loao_diagnostics.json``: confusion matrices at validation-selected
  thresholds, per-feature attacker-vs-benign separability by attack category, benign
  cross-archive shift, and recall by official ground-truth outcome subgroup;
* generated LaTeX tables and ``results/sensitivity_macros.tex``.

Bootstrap replicates are resamples of the same test senders, not independent
experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from corridor_aiops.data.parse import (
    _broadcast_key,
    _ground_truth_state,
    _records_from_json,
    veremi_archive_identity,
    veremi_ground_truth_paths,
)
from corridor_aiops.data.trajectory import FEATURE_NAMES
from corridor_aiops.evaluation.benchmark import (
    BOOTSTRAP_SEED,
    FULL_CONFIG,
    REFERENCE_MODEL,
    SPLITS,
    TIMING_KEYS,
    EventTable,
    _log,
    composite_cluster_bootstrap,
    load_event_table,
    run_models,
)
from corridor_aiops.evaluation.tables import (
    MACRO_MODEL_NAMES,
    _ci,
    _escape,
    _write,
    generalization_focus_models,
)

SCORES_FILE = Path("scores") / "per_seed_scores.npz"
VERIFY_TOLERANCE = 1e-6
DIRECTION_MARGIN = 0.05
CATEGORY_TAGS = {"dataReplay": "Replay", "dosAttack": "DoS"}
FEATURE_LABELS = {
    "speed_mps": "Speed",
    "acceleration_mps2": "Acceleration",
    "delta_t_s": "Inter-arrival time",
    "causal_jitter_s": "Causal jitter",
    "displacement_xy_m": "Horizontal displacement",
    "vertical_step_m": "Vertical step",
    "speed_change_mps": "Speed change",
    "heading_change_rad": "Heading change",
    "local_density_150m": "Local density",
    "duplicate_timestamp": "Duplicate timestamp",
    "implausible_gap": "Long gap",
}
FEATURE_TAGS = {
    "speed_mps": "Speed",
    "acceleration_mps2": "Accel",
    "delta_t_s": "DeltaT",
    "causal_jitter_s": "Jitter",
    "displacement_xy_m": "Disp",
    "vertical_step_m": "Vert",
    "speed_change_mps": "SpeedChange",
    "heading_change_rad": "Heading",
    "local_density_150m": "Density",
    "duplicate_timestamp": "Dup",
    "implausible_gap": "Gap",
}
SUBGROUPS = {
    "attacker_payload_differs": "content differs from ground truth",
    "attacker_no_ground_truth": "no ground-truth broadcast (injected)",
    "attacker_payload_matches": "content matches ground truth",
}
SUBGROUP_TAGS = {
    "attacker_payload_differs": "Differs",
    "attacker_no_ground_truth": "Injected",
    "attacker_payload_matches": "Genuine",
}


# ---------------------------------------------------------------------------
# Small, testable helpers
# ---------------------------------------------------------------------------


def ci_verdict(interval: list[float]) -> str:
    """Classify a paired-difference CI as reference better, worse, or inconclusive."""
    low, high = interval
    if low > 0:
        return "reference_better"
    if high < 0:
        return "reference_worse"
    return "inconclusive"


def confusion(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Confusion counts, precision, recall, F1, and FAR for alerts ``score >= threshold``."""
    alerts = np.asarray(scores) >= threshold
    positive = np.asarray(labels) == 1
    tp = int(np.sum(alerts & positive))
    fp = int(np.sum(alerts & ~positive))
    fn = int(np.sum(~alerts & positive))
    tn = int(np.sum(~alerts & ~positive))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def _mean_std(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    out = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows if row[key] is not None], dtype=float)
        if values.size:
            out[key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            }
    return out


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    return float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else None


def ground_truth_outcomes(table: EventTable, data_dir: Path) -> np.ndarray:
    """Per-event outcome of the official ground-truth comparison.

    ``benign`` for attacker = 0 (the consistency audit guarantees these match), and for
    attacker = 1 one of ``attacker_payload_differs``, ``attacker_no_ground_truth``, or
    ``attacker_payload_matches``. Uses the same key and tolerance as the audit.
    """
    scenarios = {veremi_archive_identity(path)[0] for path in Path(data_dir).glob("*.zip")}
    reference: dict[tuple[str, str, str, str], tuple[float, float, float] | None] = {}
    for scenario in scenarios:
        for split, path in veremi_ground_truth_paths(Path(data_dir), scenario).items():
            for record in _records_from_json(Path(path).read_bytes()):
                reference[_broadcast_key(split, record)] = _ground_truth_state(record)
    outcomes = np.empty(len(table.events), dtype=object)
    for index, event in enumerate(table.events):
        if table.labels[index] == 0:
            outcomes[index] = "benign"
            continue
        sender, message, send_time = (event.broadcast_id or "::").split(":", 2)
        key = (
            str(table.split[index]),
            sender,
            message,
            send_time,
        )
        if key not in reference:
            outcomes[index] = "attacker_no_ground_truth"
            continue
        official = reference[key]
        metrics = event.metrics
        embedded = (metrics["position_x"], metrics["position_y"], metrics["speed"])
        same = official is not None and all(
            abs(a - b) <= 1e-6 for a, b in zip(embedded, official, strict=True)
        )
        outcomes[index] = "attacker_payload_matches" if same else "attacker_payload_differs"
    return outcomes.astype(str)


# ---------------------------------------------------------------------------
# Phase A: refit and save scores
# ---------------------------------------------------------------------------


def _contexts(table: EventTable) -> dict[str, dict[str, np.ndarray]]:
    contexts = {"official": {split: table.rows(split=split) for split in SPLITS}}
    archives = sorted(set(table.archive.tolist()))
    for source in archives:
        for target in archives:
            if source != target:
                contexts[f"loao:{source}->{target}"] = {
                    "train": table.rows(split="train", archive=source),
                    "validation": table.rows(split="validation", archive=source),
                    "test": table.rows(split="test", archive=target),
                }
    return contexts


def _reference_models(summary: dict[str, Any], context: str) -> dict[str, Any]:
    if context == "official":
        return summary["models"]
    direction = context.split(":", 1)[1]
    return summary["generalization"]["leave_one_attack_type_out"][direction]["models"]


def compute_scores(table: EventTable, summary: dict[str, Any], results_dir: Path) -> dict[str, Any]:
    """Refit official and LOAO models for every seed, save scores, verify metrics."""
    import torch

    protocol = summary["protocol"]
    torch.set_num_threads(int(protocol.get("torch_threads") or os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    seeds = list(protocol["training_seeds"])
    models = list(summary["models"])
    arrays: dict[str, np.ndarray] = {}
    verification: dict[str, Any] = {}
    for context, split_rows in _contexts(table).items():
        summaries, raw, _ = run_models(
            table,
            split_rows,
            models,
            config=FULL_CONFIG,
            seeds=seeds,
            epochs=int(protocol["epochs"]),
            label=f"sensitivity {context}",
        )
        for split in ("validation", "test"):
            arrays[f"{context}|rows|{split}"] = split_rows[split]
        reference = _reference_models(summary, context)
        compared = 0
        over_tolerance = 0
        max_delta = 0.0
        hashes_matched = 0
        for name in models:
            for index, seed in enumerate(seeds):
                run = raw[name][index]
                arrays[f"{context}|{name}|{seed}|validation"] = np.asarray(
                    run["validation_scores"], dtype=np.float64
                )
                arrays[f"{context}|{name}|{seed}|test"] = np.asarray(
                    run["test_scores"], dtype=np.float64
                )
                new = summaries[name]["per_seed"][index]
                old = reference[name]["per_seed"][index]
                hashes_matched += int(new["checkpoint_sha256"] == old["checkpoint_sha256"])
                for key, value in old.items():
                    if key in TIMING_KEYS or isinstance(value, (bool, str)) or value is None:
                        continue
                    delta = abs(float(new[key]) - float(value))
                    compared += 1
                    over_tolerance += int(delta > VERIFY_TOLERANCE)
                    max_delta = max(max_delta, delta)
        verification[context] = {
            "per_seed_metric_values_compared": compared,
            "values_over_tolerance": over_tolerance,
            "max_abs_delta": max_delta,
            "checkpoint_hashes_matched": hashes_matched,
            "checkpoint_hashes_compared": len(models) * len(seeds),
        }
        _log(f"sensitivity {context}: verification {verification[context]}")
    path = Path(results_dir) / SCORES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    meta = {
        "scores_file": str(SCORES_FILE).replace("\\", "/"),
        "scores_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "tolerance": VERIFY_TOLERANCE,
        "verification_against_benchmark_summary": verification,
        "status": (
            "MATCHES_WITHIN_TOLERANCE"
            if all(block["values_over_tolerance"] == 0 for block in verification.values())
            else "DIFFERS_FROM_BENCHMARK_SUMMARY"
        ),
    }
    (path.parent / "per_seed_scores_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return meta


# ---------------------------------------------------------------------------
# Phase B: analyses from saved scores
# ---------------------------------------------------------------------------


class _Scores:
    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            self.arrays = {key: data[key] for key in data.files}

    def rows(self, context: str, split: str) -> np.ndarray:
        return self.arrays[f"{context}|rows|{split}"]

    def get(self, context: str, model: str, seed: int, split: str) -> np.ndarray:
        return self.arrays[f"{context}|{model}|{seed}|{split}"]


def _threshold(summary_models: dict[str, Any], name: str, index: int) -> float:
    return float(summary_models[name]["per_seed"][index]["validation_f1_threshold"])


def seed_bootstrap_sensitivity(
    table: EventTable, summary: dict[str, Any], scores: _Scores, samples: int
) -> dict[str, Any]:
    seeds = list(summary["protocol"]["training_seeds"])
    models = list(summary["models"])
    test = scores.rows("official", "test")
    per_seed = {}
    for index, seed in enumerate(seeds):
        _log(f"per-seed bootstrap: training seed {seed}")
        result = composite_cluster_bootstrap(
            table.labels[test],
            table.corridor[test],
            table.source[test],
            {name: scores.get("official", name, seed, "test") for name in models},
            {name: _threshold(summary["models"], name, index) for name in models},
            reference=REFERENCE_MODEL,
            seed=BOOTSTRAP_SEED,
            samples=samples,
        )
        result.pop("_preview_rows")
        per_seed[str(seed)] = result
    first = per_seed[str(seeds[0])]["paired_differences_reference_minus_model"]
    stored = summary["uncertainty"]["sender_cluster_bootstrap"][
        "paired_differences_reference_minus_model"
    ]
    first_seed_max_delta = max(
        abs(a - b)
        for name in stored
        for metric in stored[name]
        for a, b in zip(
            first[name][metric]["ci_95pct"], stored[name][metric]["ci_95pct"], strict=True
        )
    )
    comparisons = {}
    for name in models:
        if name == REFERENCE_MODEL:
            continue
        comparisons[name] = {}
        for metric in ("delta_f1_cyber_vs_benign", "delta_auroc"):
            rows = [
                per_seed[str(seed)]["paired_differences_reference_minus_model"][name][metric]
                for seed in seeds
            ]
            verdicts = [ci_verdict(row["ci_95pct"]) for row in rows]
            means = [row["mean"] for row in rows]
            comparisons[name][metric] = {
                "per_seed": {
                    str(seed): {**row, "verdict": verdict}
                    for seed, row, verdict in zip(seeds, rows, verdicts, strict=True)
                },
                "mean_range": [float(min(means)), float(max(means))],
                "ci_envelope": [
                    float(min(row["ci_95pct"][0] for row in rows)),
                    float(max(row["ci_95pct"][1] for row in rows)),
                ],
                "sign_of_mean_stable": bool(
                    all(value > 0 for value in means) or all(value < 0 for value in means)
                ),
                "verdict_counts": {
                    verdict: verdicts.count(verdict)
                    for verdict in ("reference_better", "inconclusive", "reference_worse")
                },
                "ci_verdict_stable": len(set(verdicts)) == 1,
            }
    stable = sum(
        block["ci_verdict_stable"] for metrics in comparisons.values() for block in metrics.values()
    )
    total = sum(len(metrics) for metrics in comparisons.values())
    return {
        "method": (
            "composite (corridor_id, source_id) paired cluster bootstrap repeated once per "
            "training seed, each with that seed's test scores and validation-F1 thresholds; "
            "the same bootstrap random seed is used for every training seed, so all seeds "
            "are scored on identical cluster resamples and differences between seeds "
            "reflect model fitting only"
        ),
        "interpretation_note": (
            "Bootstrap replicates are resamples of one fixed set of test senders, not "
            "independent experiments; they quantify sender-sampling uncertainty conditional "
            "on this scenario and do not replace training-seed variability."
        ),
        "reference_model": REFERENCE_MODEL,
        "training_seeds": seeds,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "requested_replicates": samples,
        "valid_replicates_by_seed": {
            seed: block["valid_replicates"] for seed, block in per_seed.items()
        },
        "composite_clusters": per_seed[str(seeds[0])]["composite_clusters"],
        "first_seed_matches_benchmark_summary_max_abs_ci_delta": first_seed_max_delta,
        "verdict_definition": (
            "reference_better: 95% CI of (Transformer - model) above 0; reference_worse: "
            "CI below 0; inconclusive: CI contains 0"
        ),
        "comparisons": comparisons,
        "ci_verdict_stable_count": stable,
        "comparison_count": total,
        "per_seed_bootstrap": {
            seed: {
                key: block[key]
                for key in ("intervals_95pct", "paired_differences_reference_minus_model")
            }
            for seed, block in per_seed.items()
        },
    }


def feature_diagnostics(table: EventTable) -> dict[str, Any]:
    archives = sorted(set(table.archive.tolist()))
    current = table.values[:, -1, :]
    valid = table.feature_mask[:, -1, :]
    out: dict[str, Any] = {
        "unit": "current-step (last window position) raw feature values, official splits",
        "auroc_definition": (
            "single-feature AUROC of attacker vs non-malicious messages within one attack "
            "category, over messages where the feature is valid; > 0.5 means attackers have "
            "larger values, < 0.5 smaller values"
        ),
        "benign_shift_definition": (
            "single-feature AUROC separating non-malicious dosAttack from non-malicious "
            "dataReplay training messages; 0.5 means no benign distribution shift"
        ),
        "direction_margin": DIRECTION_MARGIN,
        "features": {},
    }
    for column, name in enumerate(FEATURE_NAMES):
        block: dict[str, Any] = {"label": FEATURE_LABELS[name]}
        for archive in archives:
            for split in ("train", "test"):
                rows = table.rows(archive=archive, split=split)
                rows = rows[valid[rows, column]]
                labels = table.labels[rows]
                values = current[rows, column].astype(float)
                block[f"{archive}|{split}"] = {
                    "attacker_vs_benign_auroc": _auroc(labels, values),
                    "benign_median": float(np.median(values[labels == 0])),
                    "attacker_median": float(np.median(values[labels == 1])),
                    "benign_iqr": [
                        float(np.quantile(values[labels == 0], q)) for q in (0.25, 0.75)
                    ],
                    "attacker_iqr": [
                        float(np.quantile(values[labels == 1], q)) for q in (0.25, 0.75)
                    ],
                    "valid_messages": int(len(rows)),
                }
        a = table.rows(archive=archives[0], split="train")
        b = table.rows(archive=archives[1], split="train")
        a = a[(table.labels[a] == 0) & valid[a, column]]
        b = b[(table.labels[b] == 0) & valid[b, column]]
        block["benign_cross_archive_auroc_train"] = _auroc(
            np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[current[a, column], current[b, column]]
        )
        directions = []
        for archive in archives:
            auroc = block[f"{archive}|train"]["attacker_vs_benign_auroc"]
            directions.append(
                0
                if auroc is None or abs(auroc - 0.5) < DIRECTION_MARGIN
                else int(np.sign(auroc - 0.5))
            )
        block["direction_by_archive_train"] = dict(zip(archives, directions, strict=True))
        block["direction_reverses_between_categories"] = directions[0] * directions[1] < 0
        block["informative_in_one_category_only"] = (directions[0] == 0) != (directions[1] == 0)
        out["features"][name] = block
    out["archives"] = archives
    out["reversed_features"] = [
        name
        for name, block in out["features"].items()
        if block["direction_reverses_between_categories"]
    ]
    return out


def confusion_diagnostics(
    table: EventTable, summary: dict[str, Any], scores: _Scores, outcomes: np.ndarray
) -> dict[str, Any]:
    seeds = list(summary["protocol"]["training_seeds"])
    models = list(summary["models"])
    labels = table.labels
    archives = sorted(set(table.archive.tolist()))
    contexts: dict[str, Any] = {}

    # pooled official models, sliced per attack category; threshold re-selected on the
    # same category's validation records (the per_attack_category protocol)
    from corridor_aiops.evaluation.benchmark import _best_f1_threshold

    validation = scores.rows("official", "validation")
    test = scores.rows("official", "test")
    for archive in archives:
        v_mask = table.archive[validation] == archive
        t_mask = table.archive[test] == archive
        context = {
            "protocol": (
                "pooled official-split models; threshold maximizes F1 on this category's "
                "validation records; test restricted to this category"
            ),
            "test_class_counts": {
                "attacker": int(labels[test][t_mask].sum()),
                "non_malicious": int((labels[test][t_mask] == 0).sum()),
            },
            "models": {},
        }
        for name in models:
            per_seed = []
            subgroup_rows = []
            for seed in seeds:
                v_scores = scores.get("official", name, seed, "validation")[v_mask]
                t_scores = scores.get("official", name, seed, "test")[t_mask]
                threshold = _best_f1_threshold(labels[validation][v_mask], v_scores)
                per_seed.append(confusion(labels[test][t_mask], t_scores, threshold))
                subgroup_rows.append(
                    _subgroups(labels[test][t_mask], outcomes[test][t_mask], t_scores, threshold)
                )
            context["models"][name] = _summarize_context(per_seed, subgroup_rows)
        contexts[f"pooled:{archive}"] = context

    for direction in summary["generalization"]["leave_one_attack_type_out"]:
        key = f"loao:{direction}"
        validation = scores.rows(key, "validation")
        test = scores.rows(key, "test")
        reference = summary["generalization"]["leave_one_attack_type_out"][direction]["models"]
        context = {
            "protocol": (
                "fit and threshold on the source category's official train and validation "
                "splits; test on the target category's official test split"
            ),
            "test_class_counts": {
                "attacker": int(labels[test].sum()),
                "non_malicious": int((labels[test] == 0).sum()),
            },
            "models": {},
        }
        for name in models:
            per_seed = []
            source_validation = []
            subgroup_rows = []
            for index, seed in enumerate(seeds):
                threshold = _threshold(reference, name, index)
                t_scores = scores.get(key, name, seed, "test")
                per_seed.append(confusion(labels[test], t_scores, threshold))
                source_validation.append(
                    confusion(
                        labels[validation], scores.get(key, name, seed, "validation"), threshold
                    )
                )
                subgroup_rows.append(_subgroups(labels[test], outcomes[test], t_scores, threshold))
            block = _summarize_context(per_seed, subgroup_rows)
            block["source_validation_mean_std"] = _mean_std(source_validation)
            context["models"][name] = block
        contexts[key] = context
    return contexts


def _subgroups(
    labels: np.ndarray, outcomes: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    benign = labels == 0
    for group in SUBGROUPS:
        members = outcomes == group
        if not members.any():
            continue
        selected = benign | members
        out[f"{group}|count"] = float(members.sum())
        out[f"{group}|recall"] = float(np.mean(scores[members] >= threshold))
        out[f"{group}|auroc_vs_all_non_malicious"] = _auroc(
            members[selected].astype(int), scores[selected]
        )
    return out


def _summarize_context(
    per_seed: list[dict[str, float]], subgroup_rows: list[dict[str, float | None]]
) -> dict[str, Any]:
    return {
        "per_seed": per_seed,
        "mean_std": _mean_std(per_seed),
        "subgroups_mean_std": _mean_std(subgroup_rows) if subgroup_rows[0] else {},
    }


# ---------------------------------------------------------------------------
# Tables and macros
# ---------------------------------------------------------------------------

VERDICT_TEXT = {
    "reference_better": "TT better",
    "reference_worse": "TT worse",
    "inconclusive": "n.s.",
}


def _verdict_cell(block: dict[str, Any]) -> str:
    counts = {key: value for key, value in block["verdict_counts"].items() if value}
    if len(counts) == 1:
        verdict, count = next(iter(counts.items()))
        return f"{VERDICT_TEXT[verdict]} ({count}/{count})"
    return "mixed: " + ", ".join(f"{count} {VERDICT_TEXT[v]}" for v, count in counts.items())


def _seed_bootstrap_table(summary: dict[str, Any], sensitivity: dict[str, Any]) -> list[str]:
    body = [
        "& \\multicolumn{3}{c}{$\\Delta$F1 (TT $-$ model)} & "
        "\\multicolumn{3}{c}{$\\Delta$AUROC (TT $-$ model)} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        "Model & Seed means & CI envelope & CI verdict & "
        "Seed means & CI envelope & CI verdict \\\\",
        "\\midrule",
    ]
    for name, metrics in sensitivity["comparisons"].items():
        cells = []
        for metric in ("delta_f1_cyber_vs_benign", "delta_auroc"):
            block = metrics[metric]
            cells += [_ci(block["mean_range"]), _ci(block["ci_envelope"]), _verdict_cell(block)]
        body.append(
            f"{_escape(summary['models'][name]['display_name'])} & " + " & ".join(cells) + " \\\\"
        )
    return body


def _loao_confusion_table(summary: dict[str, Any], diagnostics: dict[str, Any]) -> list[str]:
    focus = generalization_focus_models(summary)
    body = [
        "Context & Model & Precision & Recall & FAR (\\%) & TP & FN & FP & TN \\\\",
    ]
    for context, block in diagnostics["contexts"].items():
        kind, name = context.split(":", 1)
        label = (
            f"Pooled, {name} test"
            if kind == "pooled"
            else "LOAO " + name.replace("->", " $\\rightarrow$ ")
        )
        body.append("\\midrule")
        for index, model in enumerate(m for m in block["models"] if m in focus):
            stats = block["models"][model]["mean_std"]
            body.append(
                f"{label if index == 0 else ''} & "
                f"{_escape(summary['models'][model]['display_name'])} & "
                f"{stats['precision']['mean']:.3f} & {stats['recall']['mean']:.3f} & "
                f"{100 * stats['false_alarm_rate']['mean']:.2f} & "
                + " & ".join(
                    f"{stats[key]['mean']:,.0f}".replace(",", "{,}")
                    for key in ("tp", "fn", "fp", "tn")
                )
                + " \\\\"
            )
    return body


def _fmt(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.3g}" if abs(value) < 100 else f"{value:.0f}"


def _feature_table(diagnostics: dict[str, Any]) -> list[str]:
    features = diagnostics["features"]
    a, b = features["archives"]
    body = [
        "& \\multicolumn{2}{c}{Attacker vs.\\ non-malicious AUROC} & Benign shift & "
        "\\multicolumn{3}{c}{Median (train)} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){5-7}",
        f"Feature & \\texttt{{{a}}} & \\texttt{{{b}}} & AUROC & Non-mal. & "
        f"Att.\\ \\texttt{{{a}}} & Att.\\ \\texttt{{{b}}} \\\\",
        "\\midrule",
    ]
    for block in features["features"].values():
        left = block[f"{a}|train"]
        right = block[f"{b}|train"]
        benign = np.median([left["benign_median"], right["benign_median"]])
        marker = "$^\\dagger$" if block["direction_reverses_between_categories"] else ""
        body.append(
            f"{block['label']}{marker} & "
            f"{left['attacker_vs_benign_auroc']:.3f} & {right['attacker_vs_benign_auroc']:.3f} & "
            f"{block['benign_cross_archive_auroc_train']:.3f} & {_fmt(float(benign))} & "
            f"{_fmt(left['attacker_median'])} & {_fmt(right['attacker_median'])} \\\\"
        )
    return body


def _subgroup_table(summary: dict[str, Any], diagnostics: dict[str, Any]) -> list[str]:
    focus = generalization_focus_models(summary)
    body = [
        "& & \\multicolumn{2}{c}{Injected (no ground-truth broadcast)} & "
        "\\multicolumn{2}{c}{Genuine content (matches ground truth)} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
        "Context & Model & AUROC & Recall & AUROC & Recall \\\\",
    ]
    for context, block in diagnostics["contexts"].items():
        if not context.endswith("dosAttack"):
            continue
        kind, name = context.split(":", 1)
        label = (
            f"Pooled, {name} test"
            if kind == "pooled"
            else "LOAO " + name.replace("->", " $\\rightarrow$ ")
        )
        body.append("\\midrule")
        for index, model in enumerate(m for m in block["models"] if m in focus):
            stats = block["models"][model]["subgroups_mean_std"]
            cells = []
            for group in ("attacker_no_ground_truth", "attacker_payload_matches"):
                for metric in ("auroc_vs_all_non_malicious", "recall"):
                    entry = stats[f"{group}|{metric}"]
                    cells.append(f"{entry['mean']:.3f} $\\pm$ {entry['std']:.3f}")
            body.append(
                f"{label if index == 0 else ''} & "
                f"{_escape(summary['models'][model]['display_name'])} & "
                + " & ".join(cells)
                + " \\\\"
            )
    return body


def _write_macros(
    summary: dict[str, Any],
    sensitivity: dict[str, Any],
    diagnostics: dict[str, Any],
    meta: dict[str, Any],
    path: Path,
) -> None:
    lines = ["% Generated by corridor_aiops.evaluation.sensitivity; do not edit by hand."]
    defined: set[str] = set()

    def define(name: str, value: Any) -> None:
        if not name.isalpha():
            raise ValueError(f"Invalid LaTeX macro name: {name}")
        if name in defined:
            return
        defined.add(name)
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    def count(value: float) -> str:
        return f"{value:,.0f}".replace(",", "{,}")

    blocks = meta["verification_against_benchmark_summary"].values()

    def total(key: str) -> int:
        return sum(block[key] for block in blocks)

    define("ResSensVerifyValues", count(total("per_seed_metric_values_compared")))
    define("ResSensVerifyOver", count(total("values_over_tolerance")))
    define("ResSensVerifyMaxDelta", f"{max(block['max_abs_delta'] for block in blocks):.1e}")
    define("ResSensHashesMatched", count(total("checkpoint_hashes_matched")))
    define("ResSensHashesCompared", count(total("checkpoint_hashes_compared")))
    define("ResSeedBootStable", sensitivity["ci_verdict_stable_count"])
    define("ResSeedBootComparisons", sensitivity["comparison_count"])
    define(
        "ResSeedBootFirstSeedDelta",
        f"{sensitivity['first_seed_matches_benchmark_summary_max_abs_ci_delta']:.1e}",
    )
    for name, metrics in sensitivity["comparisons"].items():
        tag = MACRO_MODEL_NAMES[name]
        for metric, suffix in (("delta_f1_cyber_vs_benign", "FOne"), ("delta_auroc", "AUROC")):
            block = metrics[metric]
            define(f"ResSeedBoot{tag}{suffix}Verdict", _verdict_cell(block))
            define(f"ResSeedBoot{tag}{suffix}Envelope", _ci(block["ci_envelope"]))
            define(f"ResSeedBoot{tag}{suffix}Means", _ci(block["mean_range"]))
    features = diagnostics["features"]
    for name, block in features["features"].items():
        for archive in features["archives"]:
            define(
                f"ResFeat{FEATURE_TAGS[name]}{CATEGORY_TAGS[archive]}AUROC",
                f"{block[f'{archive}|train']['attacker_vs_benign_auroc']:.3f}",
            )
            define(
                f"ResFeat{FEATURE_TAGS[name]}{CATEGORY_TAGS[archive]}AttMedian",
                _fmt(block[f"{archive}|train"]["attacker_median"]),
            )
        define(
            f"ResFeat{FEATURE_TAGS[name]}BenignShift",
            f"{block['benign_cross_archive_auroc_train']:.3f}",
        )
    shifts = [b["benign_cross_archive_auroc_train"] for b in features["features"].values()]
    define("ResFeatBenignShiftMin", f"{min(shifts):.3f}")
    define("ResFeatBenignShiftMax", f"{max(shifts):.3f}")
    define("ResFeatReversedCount", len(features["reversed_features"]))
    define("ResFeatMargin", f"{features['direction_margin']:.2f}")
    for context, block in diagnostics["contexts"].items():
        kind, name = context.split(":", 1)
        if kind == "pooled":
            ctag = f"Pooled{CATEGORY_TAGS[name]}"
        else:
            source, target = name.split("->")
            ctag = f"LOAO{CATEGORY_TAGS[source]}To{CATEGORY_TAGS[target]}"
        define(f"ResConf{ctag}Attackers", count(block["test_class_counts"]["attacker"]))
        for model, result in block["models"].items():
            tag = MACRO_MODEL_NAMES[model]
            stats = result["mean_std"]
            define(f"ResConf{ctag}{tag}Precision", f"{stats['precision']['mean']:.3f}")
            define(f"ResConf{ctag}{tag}Recall", f"{stats['recall']['mean']:.3f}")
            define(f"ResConf{ctag}{tag}FAR", f"{100 * stats['false_alarm_rate']['mean']:.2f}")
            if "source_validation_mean_std" in result:
                src = result["source_validation_mean_std"]
                define(f"ResConf{ctag}{tag}SourceRecall", f"{src['recall']['mean']:.3f}")
                define(
                    f"ResConf{ctag}{tag}SourceFAR", f"{100 * src['false_alarm_rate']['mean']:.2f}"
                )
            for key, entry in result["subgroups_mean_std"].items():
                group, metric = key.split("|")
                if metric == "count":
                    define(f"ResSub{ctag}{SUBGROUP_TAGS[group]}Count", count(entry["mean"]))
                    continue
                mtag = "Recall" if metric == "recall" else "AUROC"
                define(f"ResSub{ctag}{tag}{SUBGROUP_TAGS[group]}{mtag}", f"{entry['mean']:.3f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(
    table: EventTable,
    summary: dict[str, Any],
    results_dir: Path,
    data_dir: Path,
    samples: int,
) -> dict[str, Any]:
    results_dir = Path(results_dir)
    meta = json.loads(
        (results_dir / SCORES_FILE.parent / "per_seed_scores_meta.json").read_text(encoding="utf-8")
    )
    score_path = results_dir / SCORES_FILE
    if hashlib.sha256(score_path.read_bytes()).hexdigest() != meta["scores_sha256"]:
        raise ValueError("Saved score file does not match its recorded SHA-256")
    scores = _Scores(score_path)
    sensitivity = seed_bootstrap_sensitivity(table, summary, scores, samples)
    sensitivity["scores"] = meta
    (results_dir / "seed_bootstrap_sensitivity.json").write_text(
        json.dumps(sensitivity, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _log("LOAO diagnostics: ground-truth outcomes, features, confusion matrices")
    outcomes = ground_truth_outcomes(table, data_dir)
    test_rows = scores.rows("official", "test")
    outcome_counts = {
        archive: {
            group: int(
                np.sum((table.archive[test_rows] == archive) & (outcomes[test_rows] == group))
            )
            for group in SUBGROUPS
        }
        for archive in sorted(set(table.archive.tolist()))
    }
    diagnostics = {
        "question": (
            "why do supervised detectors trained on one attack category rank the other "
            "category's attacker messages below non-malicious messages (AUROC < 0.5)?"
        ),
        "threshold_policy": "validation-selected maximum-F1 threshold, per model and seed",
        "subgroup_definitions": SUBGROUPS,
        "official_test_attacker_subgroup_counts": outcome_counts,
        "features": feature_diagnostics(table),
        "contexts": confusion_diagnostics(table, summary, scores, outcomes),
        "scores": meta,
    }
    (results_dir / "loao_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write(
        results_dir / "results_seed_bootstrap.tex",
        "lcccccc",
        7,
        _seed_bootstrap_table(summary, sensitivity),
    )
    _write(
        results_dir / "results_loao_confusion.tex",
        "llccccccc",
        9,
        _loao_confusion_table(summary, diagnostics),
    )
    _write(results_dir / "results_feature_shift.tex", "lcccccc", 7, _feature_table(diagnostics))
    _write(
        results_dir / "results_loao_subgroups.tex",
        "llcccc",
        6,
        _subgroup_table(summary, diagnostics),
    )
    _write_macros(summary, sensitivity, diagnostics, meta, results_dir / "sensitivity_macros.tex")
    return {"sensitivity": sensitivity, "diagnostics": diagnostics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/veremi"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument(
        "--rerun", action="store_true", help="refit models and overwrite the saved scores"
    )
    args = parser.parse_args()
    summary = json.loads((args.results_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    samples = (
        args.bootstrap_samples
        or summary["uncertainty"]["sender_cluster_bootstrap"]["requested_replicates"]
    )
    _log("Loading official VeReMi archives and extracting causal windows")
    table, _ = load_event_table(args.data_dir)
    if args.rerun or not (args.results_dir / SCORES_FILE).exists():
        compute_scores(table, summary, args.results_dir)
    analyze(table, summary, args.results_dir, args.data_dir, samples)
    _log("sensitivity analysis finished")


if __name__ == "__main__":
    main()
