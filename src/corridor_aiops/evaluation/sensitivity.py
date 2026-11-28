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
    scientific,
    write_text_macros,
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
# The VeReMi NextGen paper (Hermann et al., VNC 2026) defines the attacker flag per message as
# a significant deviation of at least one attribute; the ground-truth audit compares position
# and speed only. Records whose position or speed the official ground truth contradicts
# (replay divergence) or that it lacks (injected flood messages) are message-corroborated;
# dosAttack records whose position and speed match the ground truth are state-matching.
# The key "sender_level_genuine_state" is a legacy name kept for compatibility with existing
# result files and consumers; it does not mean that the dataset labels whole senders.
LABEL_GRANULARITY_GROUPS = {
    "message_corroborated": ("attacker_payload_differs", "attacker_no_ground_truth"),
    "sender_level_genuine_state": ("attacker_payload_matches",),
}
LABEL_GRANULARITY_TAGS = {
    "message_corroborated": "Msg",
    "sender_level_genuine_state": "SenderLevel",
}
EFFECT_SIZE_NOTE = (
    "Percentile intervals from a cluster bootstrap whose minority class is concentrated "
    "in relatively few sender clusters can be too narrow or biased when a small number of "
    "attacker clusters dominates a metric, and they condition on one fixed set of test "
    "senders. Seed-level effect sizes rest on few training seeds, so the standardized "
    "effect d_z (mean paired difference divided by its seed standard deviation) is "
    "imprecise and undefined when the paired difference has zero variance."
)


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
                "effect_size": seed_effect_size(summary, name, metric, rows),
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
        "effect_size_note": EFFECT_SIZE_NOTE,
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


def seed_effect_size(
    summary: dict[str, Any], name: str, metric: str, bootstrap_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Seed-level paired effect of the reference model over ``name``.

    The paired difference per seed is the reference metric minus the model metric on the
    full official test split (no resampling). d_z is its mean over its sample standard
    deviation across seeds; it is ``None`` when that deviation is zero.
    """
    key = metric.removeprefix("delta_")
    reference = summary["models"][REFERENCE_MODEL]["per_seed"]
    other = summary["models"][name]["per_seed"]
    deltas = np.asarray(
        [float(r[key]) - float(o[key]) for r, o in zip(reference, other, strict=True)]
    )
    sd = float(deltas.std(ddof=1)) if deltas.size > 1 else 0.0
    fractions = [float(row["fraction_of_replicates_reference_better"]) for row in bootstrap_rows]
    return {
        "paired_difference_per_seed": deltas.tolist(),
        "mean_paired_difference": float(deltas.mean()),
        "sd_paired_difference": sd,
        "standardized_effect_dz": float(deltas.mean() / sd) if sd > 1e-12 else None,
        "fraction_of_replicates_reference_better_per_seed": fractions,
        "fraction_of_replicates_reference_better_mean": float(np.mean(fractions)),
    }


def sender_overlap_breakdown(
    table: EventTable, summary: dict[str, Any], scores: _Scores
) -> dict[str, Any]:
    """Official test split divided into senders seen and unseen in the official train split.

    A test message is seen-sender when its bare source_id occurs in the official training
    split of the same archive. Every model uses its own pooled validation-F1 threshold.
    """
    seeds = list(summary["protocol"]["training_seeds"])
    labels = table.labels
    test = scores.rows("official", "test")
    train = table.rows(split="train")
    seen = np.zeros(len(test), dtype=bool)
    counts: dict[str, Any] = {}
    for archive in sorted(set(table.archive.tolist())):
        train_sources = set(table.source[train][table.archive[train] == archive].tolist())
        in_archive = table.archive[test] == archive
        archive_seen = in_archive & np.isin(table.source[test], sorted(train_sources))
        seen |= archive_seen
        block = {}
        for group, mask in (("seen", archive_seen), ("unseen", in_archive & ~archive_seen)):
            block[group] = {
                "senders": int(len(set(table.source[test][mask].tolist()))),
                "attacker_senders": int(
                    len(set(table.source[test][mask & (labels[test] == 1)].tolist()))
                ),
                "messages": int(mask.sum()),
                "attacker_messages": int(labels[test][mask].sum()),
            }
        counts[archive] = block
    groups = {"seen": seen, "unseen": ~seen}
    models: dict[str, Any] = {}
    for name in summary["models"]:
        threshold_seed = [_threshold(summary["models"], name, i) for i in range(len(seeds))]
        block: dict[str, Any] = {}
        for group, mask in groups.items():
            rows = []
            for i, seed in enumerate(seeds):
                t_scores = scores.get("official", name, seed, "test")[mask]
                row = confusion(labels[test][mask], t_scores, threshold_seed[i])
                row["auroc"] = _auroc(labels[test][mask], t_scores)
                rows.append(row)
            block[group] = {"per_seed": rows, "mean_std": _mean_std(rows)}
        models[name] = block
    sender_disjoint = summary["generalization"]["sender_disjoint_holdout"]
    return {
        "definition": (
            "seen-sender test messages: official test messages whose bare source_id occurs "
            "in the official training split of the same archive; unseen-sender: all other "
            "official test messages. Thresholds are the pooled validation-F1 thresholds."
        ),
        "counts_by_archive": counts,
        "models": models,
        "sender_disjoint_holdout_all_models": {
            name: {
                metric: block["mean_std_across_training_seeds"][metric]
                for metric in ("auroc", "f1_cyber_vs_benign", "false_alarm_rate_non_malicious")
            }
            for name, block in sender_disjoint["models"].items()
        },
    }


def _group_metrics(
    labels: np.ndarray, outcomes: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    benign = labels == 0
    for group, members_of in LABEL_GRANULARITY_GROUPS.items():
        members = np.isin(outcomes, members_of)
        out[f"{group}|count"] = float(members.sum())
        if not members.any():
            out[f"{group}|recall"] = None
            out[f"{group}|auroc_vs_all_non_malicious"] = None
            continue
        selected = benign | members
        out[f"{group}|recall"] = float(np.mean(scores[members] >= threshold))
        out[f"{group}|auroc_vs_all_non_malicious"] = _auroc(
            members[selected].astype(int), scores[selected]
        )
    return out


def label_granularity_breakdown(
    table: EventTable, summary: dict[str, Any], scores: _Scores, outcomes: np.ndarray
) -> dict[str, Any]:
    """Message-corroborated versus state-matching attacker records, all models."""
    seeds = list(summary["protocol"]["training_seeds"])
    labels = table.labels
    contexts: dict[str, Any] = {}
    test = scores.rows("official", "test")
    loao = summary["generalization"]["leave_one_attack_type_out"]
    plans = [("official", "official", None, summary["models"])]
    plans += [
        (f"loao:{direction}", f"loao:{direction}", None, block["models"])
        for direction, block in loao.items()
        if block.get("status") == "EVALUABLE"
    ]
    for name_key, score_key, _, reference in plans:
        rows_test = test if score_key == "official" else scores.rows(score_key, "test")
        block: dict[str, Any] = {
            "protocol": (
                "pooled official models on the full official test split with their pooled "
                "validation-F1 thresholds"
                if score_key == "official"
                else "LOAO models with source-category validation-F1 thresholds"
            ),
            "group_counts": {
                group: int(np.isin(outcomes[rows_test], members).sum())
                for group, members in LABEL_GRANULARITY_GROUPS.items()
            },
            "models": {},
        }
        for name in summary["models"]:
            per_seed = []
            for index, seed in enumerate(seeds):
                t_scores = scores.get(score_key, name, seed, "test")
                per_seed.append(
                    _group_metrics(
                        labels[rows_test],
                        outcomes[rows_test],
                        t_scores,
                        _threshold(reference, name, index),
                    )
                )
            block["models"][name] = {"per_seed": per_seed, "mean_std": _mean_std(per_seed)}
        contexts[name_key] = block
    return {
        "definition": (
            "The VeReMi NextGen paper defines the attacker flag per message as a significant "
            "deviation of at least one attribute; the ground-truth audit compares position "
            "and speed only. message_corroborated: attacker records whose position or speed "
            "diverges from the official ground truth (dataReplay) or that have no "
            "ground-truth broadcast (dosAttack injected); sender_level_genuine_state (legacy "
            "key name, kept for compatibility): dosAttack attacker records whose position and "
            "speed match their ground-truth broadcast, so a detector cannot flag them through "
            "falsified position or speed. AUROC compares each group with all non-malicious "
            "test records of the same context."
        ),
        "contexts": contexts,
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
        block["benign_median_pooled_train"] = float(
            np.median(np.r_[current[a, column], current[b, column]])
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

    # the same pooled models under the single pooled validation-F1 threshold of each
    # seed (no per-category re-selection), sliced per attack category
    for archive in archives:
        t_mask = table.archive[test] == archive
        context = {
            "protocol": (
                "pooled official-split models and their single pooled validation-F1 "
                "threshold; test restricted to this category"
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
            for index, seed in enumerate(seeds):
                threshold = _threshold(summary["models"], name, index)
                t_scores = scores.get("official", name, seed, "test")[t_mask]
                per_seed.append(confusion(labels[test][t_mask], t_scores, threshold))
                subgroup_rows.append(
                    _subgroups(labels[test][t_mask], outcomes[test][t_mask], t_scores, threshold)
                )
            context["models"][name] = _summarize_context(per_seed, subgroup_rows)
        contexts[f"pooledthreshold:{archive}"] = context

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

# ---------------------------------------------------------------------------
# Cross-archive benign overlap and sender-disjoint deduplicated LOAO
# ---------------------------------------------------------------------------

DEDUP_SALT = "corridor-aiops-loao-sender-disjoint-v1"
DEDUP_SOURCE_FRACTION = 0.5
OVERLAP_LEVELS = (
    "message_sender_messageID_sendTime",
    "receive_time_sender_rcvTime",
    "bare_source_id",
    "trajectory_window_L10",
)


def _window_keys(table: EventTable, rows: np.ndarray) -> np.ndarray:
    """Hash of the model-input L = 10 window (feature values and validity masks)."""
    columns = [table.feature_names.index(name) for name in FULL_CONFIG.feature_names]
    values = np.round(table.values[rows][:, :, columns].astype(np.float64), 9)
    masks = table.feature_mask[rows][:, :, columns]
    return np.asarray(
        [
            hashlib.sha256(value.tobytes() + mask.tobytes()).hexdigest()
            for value, mask in zip(values, masks, strict=True)
        ]
    )


def _level_keys(table: EventTable, rows: np.ndarray, level: str) -> set:
    if level == "message_sender_messageID_sendTime":
        return {table.events[i].broadcast_id for i in rows}
    if level == "receive_time_sender_rcvTime":
        return {(table.source[i], table.events[i].receive_time_ns) for i in rows}
    if level == "bare_source_id":
        return set(table.source[rows].tolist())
    if level == "trajectory_window_L10":
        return set(_window_keys(table, rows).tolist())
    raise ValueError(f"Unknown overlap level: {level}")


def overlap_counts(table: EventTable, left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Shared keys between two row sets at every overlap level."""
    out = {}
    for level in OVERLAP_LEVELS:
        a = _level_keys(table, left, level)
        b = _level_keys(table, right, level)
        out[level] = {"left_unique": len(a), "right_unique": len(b), "shared": len(a & b)}
    return out


def benign_overlap_audit(
    table: EventTable, loao_partitions: dict[str, dict[str, np.ndarray]]
) -> dict[str, Any]:
    """Exact overlap of non-malicious records between the two archives.

    Reported for the whole archives, within each official split, and between the
    training/validation side and the test side of every LOAO partition supplied.
    """
    archives = sorted(set(table.archive.tolist()))
    benign = table.labels == 0

    def rows(**conditions: str) -> np.ndarray:
        selected = table.rows(**conditions)
        return selected[benign[selected]]

    audit: dict[str, Any] = {
        "levels": {
            "message_sender_messageID_sendTime": "(sender_id, messageID, sendTime) broadcast key",
            "receive_time_sender_rcvTime": "(sender_id, rcvTime) of the retained receive copy",
            "bare_source_id": "sender identifier without the archive prefix",
            "trajectory_window_L10": (
                "SHA-256 of the L = 10 model-input window (11 features and validity masks)"
            ),
        },
        "archives": archives,
        "whole_archive_non_malicious": overlap_counts(
            table, rows(archive=archives[0]), rows(archive=archives[1])
        ),
        "same_official_split_non_malicious": {
            split: overlap_counts(
                table,
                rows(archive=archives[0], split=split),
                rows(archive=archives[1], split=split),
            )
            for split in SPLITS
        },
        "loao_train_validation_vs_test_non_malicious": {},
    }
    for name, partition in loao_partitions.items():
        fit = np.concatenate((partition["train"], partition["validation"]))
        audit["loao_train_validation_vs_test_non_malicious"][name] = overlap_counts(
            table, fit[benign[fit]], partition["test"][benign[partition["test"]]]
        )
    return audit


def sender_disjoint_loao_partitions(
    table: EventTable,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """LOAO partitions with disjoint bare source_id sets on the two sides.

    Bare source_ids are ordered by a salted SHA-256 within attacker-ever and
    never-attacker strata; the first half may only appear in source-archive train and
    validation rows, the second half only in target-archive test rows.
    """
    attacker_ids = set(table.source[table.labels == 1].tolist())
    all_ids = sorted(set(table.source.tolist()))
    source_side: set[str] = set()
    for stratum in (
        [sid for sid in all_ids if sid in attacker_ids],
        [sid for sid in all_ids if sid not in attacker_ids],
    ):
        ordered = sorted(
            stratum, key=lambda sid: hashlib.sha256(f"{DEDUP_SALT}:{sid}".encode()).hexdigest()
        )
        source_side.update(ordered[: int(round(DEDUP_SOURCE_FRACTION * len(ordered)))])
    on_source = np.isin(table.source, sorted(source_side))
    archives = sorted(set(table.archive.tolist()))
    partitions: dict[str, dict[str, np.ndarray]] = {}
    for source in archives:
        for target in archives:
            if source == target:
                continue
            train = table.rows(split="train", archive=source)
            validation = table.rows(split="validation", archive=source)
            test = table.rows(split="test", archive=target)
            partitions[f"{source}->{target}"] = {
                "train": train[on_source[train]],
                "validation": validation[on_source[validation]],
                "test": test[~on_source[test]],
            }
    for name, partition in partitions.items():
        fit = np.concatenate((partition["train"], partition["validation"]))
        shared = set(table.source[fit].tolist()) & set(table.source[partition["test"]].tolist())
        if shared:
            raise RuntimeError(f"Sender-disjoint LOAO {name} shares {len(shared)} source_ids")
    return partitions, {
        "partition_rule": (
            "bare source_ids ordered by sha256(salt:source_id) within attacker-ever and "
            "never-attacker strata; the first half is the source side (source-archive "
            "train and validation), the second half the target side (target-archive test)"
        ),
        "salt": DEDUP_SALT,
        "source_side_fraction": DEDUP_SOURCE_FRACTION,
        "source_side_senders": len(source_side),
        "target_side_senders": len(set(all_ids) - source_side),
    }


def run_sender_disjoint_loao(
    table: EventTable, summary: dict[str, Any], results_dir: Path
) -> dict[str, Any]:
    """Fit every detector and seed on the source side; evaluate on the disjoint target."""
    import torch

    from corridor_aiops.evaluation.benchmark import _split_counts, split_eligibility

    protocol = summary["protocol"]
    torch.set_num_threads(int(protocol.get("torch_threads") or os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    seeds = list(protocol["training_seeds"])
    models = list(summary["models"])
    partitions, meta = sender_disjoint_loao_partitions(table)
    directions: dict[str, Any] = {}
    for direction, split_rows in partitions.items():
        reason = split_eligibility(table.labels, split_rows)
        if reason is not None:
            directions[direction] = {"status": "NOT_EVALUABLE", "reason": reason}
            continue
        summaries, raw, _ = run_models(
            table,
            split_rows,
            models,
            config=FULL_CONFIG,
            seeds=seeds,
            epochs=int(protocol["epochs"]),
            label=f"sender-disjoint LOAO {direction}",
        )
        test = split_rows["test"]
        for name in models:
            rows = [
                confusion(
                    table.labels[test],
                    run["test_scores"],
                    summaries[name]["per_seed"][index]["validation_f1_threshold"],
                )
                for index, run in enumerate(raw[name])
            ]
            summaries[name]["confusion_per_seed"] = rows
            summaries[name]["confusion_mean_std"] = _mean_std(rows)
        directions[direction] = {
            "status": "EVALUABLE",
            "class_counts": _split_counts(table.labels, split_rows),
            "overlap_train_validation_vs_test_non_malicious": overlap_counts(
                table,
                np.concatenate((split_rows["train"], split_rows["validation"]))[
                    table.labels[np.concatenate((split_rows["train"], split_rows["validation"]))]
                    == 0
                ],
                test[table.labels[test] == 0],
            ),
            "models": summaries,
        }
    result = {
        "protocol": (
            "normalization, all detectors, checkpoints, and validation-F1 thresholds fit on "
            "the source side only; evaluated on the disjoint target side"
        ),
        **meta,
        "training_seeds": seeds,
        "directions": directions,
    }
    (Path(results_dir) / "loao_sender_disjoint.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


# ---------------------------------------------------------------------------
# Threshold-selection procedures
# ---------------------------------------------------------------------------

THRESHOLD_PROCEDURES = ("max_f1", "youden_j", "far_1.0pct", "far_5.0pct")
PROCEDURE_LABELS = {
    "max_f1": "Max-F1",
    "youden_j": "Youden $J$",
    "far_1.0pct": "FAR budget 1\\%",
    "far_5.0pct": "FAR budget 5\\%",
}
PROCEDURE_TAGS = {
    "max_f1": "MaxF",
    "youden_j": "Youden",
    "far_1.0pct": "BudgetOne",
    "far_5.0pct": "BudgetFive",
}


def youden_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Validation threshold maximizing TPR - FPR (ties resolved to the higher threshold)."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores)
    finite = np.isfinite(thresholds)
    index = int(np.argmax(np.where(finite, tpr - fpr, -np.inf)))
    return float(thresholds[index])


def procedure_threshold(procedure: str, labels: np.ndarray, scores: np.ndarray) -> float:
    from corridor_aiops.evaluation.benchmark import _best_f1_threshold, far_budget_threshold

    if procedure == "max_f1":
        return _best_f1_threshold(labels, scores)
    if procedure == "youden_j":
        return youden_threshold(labels, scores)
    if procedure.startswith("far_"):
        budget = float(procedure.removeprefix("far_").removesuffix("pct")) / 100.0
        return far_budget_threshold(scores[labels == 0], budget)
    raise ValueError(f"Unknown threshold procedure: {procedure}")


def threshold_procedures(
    table: EventTable, summary: dict[str, Any], scores: _Scores
) -> dict[str, Any]:
    """Test metrics under each validation threshold procedure and seed-wise differences."""
    seeds = list(summary["protocol"]["training_seeds"])
    validation = scores.rows("official", "validation")
    test = scores.rows("official", "test")
    out: dict[str, Any] = {
        "procedures": {
            "max_f1": "threshold maximizing validation F1",
            "youden_j": "threshold maximizing validation TPR - FPR (Youden's J)",
            "far_1.0pct": "conservative validation quantile: non-malicious alert rate <= 1%",
            "far_5.0pct": "conservative validation quantile: non-malicious alert rate <= 5%",
        },
        "reference_procedure": "max_f1",
        "models": {},
    }
    for name in summary["models"]:
        per_procedure: dict[str, list[dict[str, float]]] = {p: [] for p in THRESHOLD_PROCEDURES}
        for seed in seeds:
            v_scores = scores.get("official", name, seed, "validation")
            t_scores = scores.get("official", name, seed, "test")
            for procedure in THRESHOLD_PROCEDURES:
                threshold = procedure_threshold(procedure, table.labels[validation], v_scores)
                per_procedure[procedure].append(
                    {"threshold": threshold, **confusion(table.labels[test], t_scores, threshold)}
                )
        differences = {}
        for procedure in THRESHOLD_PROCEDURES[1:]:
            rows = [
                {
                    metric: new[metric] - old[metric]
                    for metric in ("f1", "recall", "precision", "false_alarm_rate")
                }
                for new, old in zip(per_procedure[procedure], per_procedure["max_f1"], strict=True)
            ]
            differences[procedure] = {
                "seedwise_mean_std": _mean_std(rows),
                "seeds_with_higher_recall": int(sum(row["recall"] > 0 for row in rows)),
                "seeds_with_higher_far": int(sum(row["false_alarm_rate"] > 0 for row in rows)),
            }
        out["models"][name] = {
            "per_procedure_mean_std": {p: _mean_std(rows) for p, rows in per_procedure.items()},
            "per_procedure_per_seed": per_procedure,
            "paired_difference_vs_max_f1": differences,
        }
    return out


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


def _context_label(context: str) -> str:
    kind, name = context.split(":", 1)
    if kind == "pooled":
        return f"Pooled, {name} test"
    if kind == "pooledthreshold":
        return f"Pooled threshold, {name} test"
    return "LOAO " + name.replace("->", " $\\rightarrow$ ")


def _context_tag(context: str) -> str:
    kind, name = context.split(":", 1)
    if kind == "pooled":
        return f"Pooled{CATEGORY_TAGS[name]}"
    if kind == "pooledthreshold":
        return f"PooledThr{CATEGORY_TAGS[name]}"
    source, target = name.split("->")
    return f"LOAO{CATEGORY_TAGS[source]}To{CATEGORY_TAGS[target]}"


def _loao_confusion_table(summary: dict[str, Any], diagnostics: dict[str, Any]) -> list[str]:
    focus = generalization_focus_models(summary)
    body = [
        "Context & Model & Precision & Recall & FAR (\\%) & TP & FN & FP & TN \\\\",
    ]
    for context, block in diagnostics["contexts"].items():
        label = _context_label(context)
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
        benign = block["benign_median_pooled_train"]
        marker = "$^\\dagger$" if block["direction_reverses_between_categories"] else ""
        body.append(
            f"{block['label']}{marker} & "
            f"{left['attacker_vs_benign_auroc']:.3f} & {right['attacker_vs_benign_auroc']:.3f} & "
            f"{block['benign_cross_archive_auroc_train']:.3f} & {_fmt(float(benign))} & "
            f"{_fmt(left['attacker_median'])} & {_fmt(right['attacker_median'])} \\\\"
        )
    return body


def _regime_groups(summary: dict[str, Any]) -> list[tuple[str, list[str]]]:
    return [
        (
            "Unsupervised / one-class (fit on non-malicious records)",
            [n for n, b in summary["models"].items() if b["supervision_regime"] != "supervised"],
        ),
        (
            "Supervised (fit on training labels)",
            [n for n, b in summary["models"].items() if b["supervision_regime"] == "supervised"],
        ),
    ]


def _cell(stats: dict[str, Any], key: str, scale: float = 1.0, digits: int = 3) -> str:
    entry = stats.get(key)
    return "n/a" if entry is None else f"{scale * entry['mean']:.{digits}f}"


def _label_granularity_table(summary: dict[str, Any], diagnostics: dict[str, Any]) -> list[str]:
    block = diagnostics["label_granularity"]["contexts"]
    contexts = [key for key in block if key == "official"] + [
        key for key in block if key.startswith("loao:dataReplay")
    ]
    titles = {"official": "Pooled official test"}
    body = [
        "& "
        + " & ".join(
            f"\\multicolumn{{4}}{{c}}{{{titles[c] if c in titles else _context_label(c)}}}"
            for c in contexts
        )
        + " \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
        "& "
        + " & ".join(
            "\\multicolumn{2}{c}{Message-corroborated} & \\multicolumn{2}{c}{State-matching}"
            for _ in contexts
        )
        + " \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
        "Model" + " & AUROC & Recall & AUROC & Recall" * len(contexts) + " \\\\",
    ]
    width = 1 + 4 * len(contexts)
    for title, names in _regime_groups(summary):
        body += ["\\midrule", f"\\multicolumn{{{width}}}{{l}}{{\\textit{{{title}}}}} \\\\"]
        for name in names:
            cells = []
            for context in contexts:
                stats = block[context]["models"][name]["mean_std"]
                for group in LABEL_GRANULARITY_GROUPS:
                    cells += [
                        _cell(stats, f"{group}|auroc_vs_all_non_malicious"),
                        _cell(stats, f"{group}|recall"),
                    ]
            body.append(
                f"{_escape(summary['models'][name]['display_name'])} & "
                + " & ".join(cells)
                + " \\\\"
            )
    return body


def _sender_overlap_table(summary: dict[str, Any], diagnostics: dict[str, Any]) -> list[str]:
    overlap = diagnostics["sender_overlap"]
    body = [
        "& \\multicolumn{3}{c}{Official test, seen senders} & "
        "\\multicolumn{3}{c}{Official test, unseen senders} & "
        "\\multicolumn{3}{c}{Sender-disjoint holdout} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
        "Model & AUROC & Recall & FAR (\\%) & AUROC & Recall & FAR (\\%) "
        "& AUROC & F1 & FAR (\\%) \\\\",
    ]
    for title, names in _regime_groups(summary):
        body += ["\\midrule", f"\\multicolumn{{10}}{{l}}{{\\textit{{{title}}}}} \\\\"]
        for name in names:
            cells = []
            for group in ("seen", "unseen"):
                stats = overlap["models"][name][group]["mean_std"]
                cells += [
                    _cell(stats, "auroc"),
                    _cell(stats, "recall"),
                    _cell(stats, "false_alarm_rate", 100.0, 2),
                ]
            disjoint = overlap["sender_disjoint_holdout_all_models"][name]
            cells += [
                _cell(disjoint, "auroc"),
                _cell(disjoint, "f1_cyber_vs_benign"),
                _cell(disjoint, "false_alarm_rate_non_malicious", 100.0, 2),
            ]
            body.append(
                f"{_escape(summary['models'][name]['display_name'])} & "
                + " & ".join(cells)
                + " \\\\"
            )
    return body


def _dedup_table(summary: dict[str, Any], diagnostics: dict[str, Any]) -> list[str]:
    focus = generalization_focus_models(summary)
    dedup = diagnostics["loao_sender_disjoint"]
    standard = summary["generalization"]["leave_one_attack_type_out"]
    body = [
        "& & \\multicolumn{4}{c}{Standard LOAO} & "
        "\\multicolumn{4}{c}{Sender-disjoint deduplicated LOAO} \\\\",
        "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}",
        "Direction & Model & AUROC & F1 & Recall & FAR (\\%) & AUROC & F1 & Recall & "
        "FAR (\\%) \\\\",
    ]
    for direction, block in dedup["directions"].items():
        body.append("\\midrule")
        label = direction.replace("->", " $\\rightarrow$ ")
        if block["status"] != "EVALUABLE":
            body.append(
                f"{label} & \\multicolumn{{9}}{{l}}{{NOT\\_EVALUABLE: {_escape(block['reason'])}}}"
                " \\\\"
            )
            continue
        for index, name in enumerate(m for m in block["models"] if m in focus):
            std_stats = standard[direction]["models"][name]["mean_std_across_training_seeds"]
            std_conf = diagnostics["contexts"][f"loao:{direction}"]["models"][name]["mean_std"]
            new_stats = block["models"][name]["mean_std_across_training_seeds"]
            new_conf = block["models"][name]["confusion_mean_std"]
            cells = []
            for stats, conf in ((std_stats, std_conf), (new_stats, new_conf)):
                cells += [
                    f"{stats['auroc']['mean']:.3f} $\\pm$ {stats['auroc']['std']:.3f}",
                    f"{stats['f1_cyber_vs_benign']['mean']:.3f}",
                    f"{conf['recall']['mean']:.3f}",
                    f"{100 * conf['false_alarm_rate']['mean']:.2f}",
                ]
            body.append(
                f"{label if index == 0 else ''} & "
                f"{_escape(summary['models'][name]['display_name'])} & "
                + " & ".join(cells)
                + " \\\\"
            )
    return body


def _threshold_table(summary: dict[str, Any], diagnostics: dict[str, Any]) -> list[str]:
    procedures = diagnostics["threshold_procedures"]
    body = [
        "& "
        + " & ".join(
            f"\\multicolumn{{2}}{{c}}{{{PROCEDURE_LABELS[p]}}}" for p in THRESHOLD_PROCEDURES
        )
        + " \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
        "Model" + " & Recall & FAR (\\%)" * len(THRESHOLD_PROCEDURES) + " \\\\",
    ]
    for title, names in (
        (
            "Unsupervised / one-class (fit on non-malicious records)",
            [n for n, b in summary["models"].items() if b["supervision_regime"] != "supervised"],
        ),
        (
            "Supervised (fit on training labels)",
            [n for n, b in summary["models"].items() if b["supervision_regime"] == "supervised"],
        ),
    ):
        body += ["\\midrule", f"\\multicolumn{{9}}{{l}}{{\\textit{{{title}}}}} \\\\"]
        for name in names:
            stats = procedures["models"][name]["per_procedure_mean_std"]
            cells = []
            for procedure in THRESHOLD_PROCEDURES:
                cells += [
                    f"{stats[procedure]['recall']['mean']:.3f}",
                    f"{100 * stats[procedure]['false_alarm_rate']['mean']:.2f}",
                ]
            body.append(
                f"{_escape(summary['models'][name]['display_name'])} & "
                + " & ".join(cells)
                + " \\\\"
            )
    return body


def _extra_macro_lines(summary: dict[str, Any], diagnostics: dict[str, Any]) -> list[str]:
    lines: list[str] = []

    def define(name: str, value: Any) -> None:
        if not name.isalpha():
            raise ValueError(f"Invalid LaTeX macro name: {name}")
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    def count(value: float) -> str:
        return f"{value:,.0f}".replace(",", "{,}")

    audit = diagnostics["benign_overlap_audit"]
    whole = audit["whole_archive_non_malicious"]
    same = audit["same_official_split_non_malicious"]
    level_tags = {
        "message_sender_messageID_sendTime": "Messages",
        "receive_time_sender_rcvTime": "Receives",
        "bare_source_id": "Senders",
        "trajectory_window_L10": "Windows",
    }
    for level, tag in level_tags.items():
        define(f"ResOverlapArchive{tag}", count(whole[level]["shared"]))
        define(
            f"ResOverlapSameSplit{tag}",
            count(sum(block[level]["shared"] for block in same.values())),
        )
        standard = [
            block[level]["shared"]
            for name, block in audit["loao_train_validation_vs_test_non_malicious"].items()
            if name.startswith("standard:")
        ]
        dedup = [
            block[level]["shared"]
            for name, block in audit["loao_train_validation_vs_test_non_malicious"].items()
            if name.startswith("dedup:")
        ]
        define(f"ResOverlapLOAO{tag}", count(max(standard)))
        define(f"ResOverlapDedup{tag}", count(max(dedup)))
    define(
        "ResOverlapArchiveBenignReplay",
        count(whole["message_sender_messageID_sendTime"]["left_unique"]),
    )
    define(
        "ResOverlapArchiveBenignDoS",
        count(whole["message_sender_messageID_sendTime"]["right_unique"]),
    )

    dedup = diagnostics["loao_sender_disjoint"]
    define("ResDedupSourceSenders", count(dedup["source_side_senders"]))
    define("ResDedupTargetSenders", count(dedup["target_side_senders"]))
    supervised = [
        n for n, b in summary["models"].items() if b["supervision_regime"] == "supervised"
    ]
    standard_loao = summary["generalization"]["leave_one_attack_type_out"]
    below_standard = below_dedup = pairs = 0
    for direction, block in dedup["directions"].items():
        if block["status"] != "EVALUABLE":
            continue
        source, target = direction.split("->")
        tag = f"{CATEGORY_TAGS[source]}To{CATEGORY_TAGS[target]}"
        counts_test = block["class_counts"]["test"]
        define(f"ResDedup{tag}TestRecords", count(counts_test["records"]))
        define(f"ResDedup{tag}TestAttackers", count(counts_test["attacker_1_misbehavior"]))
        for name, result in block["models"].items():
            model_tag = MACRO_MODEL_NAMES[name]
            stats = result["mean_std_across_training_seeds"]
            conf = result["confusion_mean_std"]
            define(
                f"ResDedup{tag}{model_tag}AUROC",
                f"{stats['auroc']['mean']:.3f} $\\pm$ {stats['auroc']['std']:.3f}",
            )
            define(f"ResDedup{tag}{model_tag}FOne", f"{stats['f1_cyber_vs_benign']['mean']:.3f}")
            define(f"ResDedup{tag}{model_tag}Recall", f"{conf['recall']['mean']:.3f}")
            define(f"ResDedup{tag}{model_tag}FAR", f"{100 * conf['false_alarm_rate']['mean']:.2f}")
            if name in supervised:
                pairs += 1
                below_dedup += int(stats["auroc"]["mean"] < 0.5)
                below_standard += int(
                    standard_loao[direction]["models"][name]["mean_std_across_training_seeds"][
                        "auroc"
                    ]["mean"]
                    < 0.5
                )
    define("ResDedupSupPairs", pairs)
    define("ResDedupBelowHalf", below_dedup)
    define("ResStdBelowHalf", below_standard)

    procedures = diagnostics["threshold_procedures"]
    for name, block in procedures["models"].items():
        model_tag = MACRO_MODEL_NAMES[name]
        for procedure in THRESHOLD_PROCEDURES:
            stats = block["per_procedure_mean_std"][procedure]
            ptag = PROCEDURE_TAGS[procedure]
            define(f"ResThr{ptag}{model_tag}Recall", f"{stats['recall']['mean']:.3f}")
            define(f"ResThr{ptag}{model_tag}FAR", f"{100 * stats['false_alarm_rate']['mean']:.2f}")
        youden = block["paired_difference_vs_max_f1"]["youden_j"]
        define(
            f"ResThrYoudenDelta{model_tag}FAR",
            f"{100 * youden['seedwise_mean_std']['false_alarm_rate']['mean']:+.2f}",
        )
        define(
            f"ResThrYoudenDelta{model_tag}Recall",
            f"{youden['seedwise_mean_std']['recall']['mean']:+.3f}",
        )
        define(f"ResThrYoudenHigherFARSeeds{model_tag}", youden["seeds_with_higher_far"])
    youden_far_up = sum(
        block["paired_difference_vs_max_f1"]["youden_j"]["seeds_with_higher_far"]
        == len(summary["protocol"]["training_seeds"])
        for name, block in procedures["models"].items()
        if name in supervised
    )
    define("ResThrYoudenFARUpAllSeedsSup", youden_far_up)
    define("ResThrSupCount", len(supervised))

    overlap = diagnostics["sender_overlap"]
    for group, tag in (("seen", "Seen"), ("unseen", "Unseen")):
        for key, suffix in (
            ("senders", "Senders"),
            ("attacker_senders", "AttackerSenders"),
            ("messages", "Messages"),
            ("attacker_messages", "Attackers"),
        ):
            define(
                f"Res{tag}{suffix}",
                count(sum(block[group][key] for block in overlap["counts_by_archive"].values())),
            )
        for name, block in overlap["models"].items():
            stats = block[group]["mean_std"]
            model_tag = MACRO_MODEL_NAMES[name]
            define(f"Res{tag}{model_tag}AUROC", _cell(stats, "auroc"))
            define(f"Res{tag}{model_tag}Recall", _cell(stats, "recall"))
            define(f"Res{tag}{model_tag}FAR", _cell(stats, "false_alarm_rate", 100.0, 2))
    define(
        "ResSeenSendersPerArchive",
        count(max(b["seen"]["senders"] for b in overlap["counts_by_archive"].values())),
    )
    gaps = {
        name: overlap["models"][name]["seen"]["mean_std"]["auroc"]["mean"]
        - overlap["models"][name]["unseen"]["mean_std"]["auroc"]["mean"]
        for name in supervised
    }
    define("ResSeenUnseenSupGapMin", f"{min(gaps.values()):+.3f}")
    define("ResSeenUnseenSupGapMax", f"{max(gaps.values()):+.3f}")
    for name, block in overlap["sender_disjoint_holdout_all_models"].items():
        define(f"ResSD{MACRO_MODEL_NAMES[name]}AUROC", _cell(block, "auroc"))

    granularity = diagnostics["label_granularity"]["contexts"]
    for context, block in granularity.items():
        ctag = "Pooled" if context == "official" else _context_tag(context)
        for group, gtag in LABEL_GRANULARITY_TAGS.items():
            define(f"ResLbl{ctag}{gtag}Count", count(block["group_counts"][group]))
            for name, result in block["models"].items():
                stats = result["mean_std"]
                model_tag = MACRO_MODEL_NAMES[name]
                define(
                    f"ResLbl{ctag}{model_tag}{gtag}AUROC",
                    _cell(stats, f"{group}|auroc_vs_all_non_malicious"),
                )
                define(f"ResLbl{ctag}{model_tag}{gtag}Recall", _cell(stats, f"{group}|recall"))
    return lines


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
        scientific(sensitivity["first_seed_matches_benchmark_summary_max_abs_ci_delta"]),
    )
    for name, metrics in sensitivity["comparisons"].items():
        tag = MACRO_MODEL_NAMES[name]
        for metric, suffix in (("delta_f1_cyber_vs_benign", "FOne"), ("delta_auroc", "AUROC")):
            block = metrics[metric]
            define(f"ResSeedBoot{tag}{suffix}Verdict", _verdict_cell(block))
            define(f"ResSeedBoot{tag}{suffix}Envelope", _ci(block["ci_envelope"]))
            define(f"ResSeedBoot{tag}{suffix}Means", _ci(block["mean_range"]))
            effect = block["effect_size"]
            define(f"ResEff{tag}{suffix}Delta", f"{effect['mean_paired_difference']:+.3f}")
            define(
                f"ResEff{tag}{suffix}DZ",
                "undefined"
                if effect["standardized_effect_dz"] is None
                else f"{effect['standardized_effect_dz']:+.1f}",
            )
            define(
                f"ResEff{tag}{suffix}Frac",
                f"{effect['fraction_of_replicates_reference_better_mean']:.2f}",
            )
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
        ctag = _context_tag(context)
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
    _log("benign overlap audit, sender-disjoint LOAO, threshold procedures")
    dedup_path = results_dir / "loao_sender_disjoint.json"
    if not dedup_path.exists():
        raise FileNotFoundError(
            f"{dedup_path.name} is missing; run with --rerun to fit the sender-disjoint LOAO"
        )
    dedup_partitions, _ = sender_disjoint_loao_partitions(table)
    standard_partitions = {
        key.split(":", 1)[1]: rows for key, rows in _contexts(table).items() if key != "official"
    }
    diagnostics["benign_overlap_audit"] = benign_overlap_audit(
        table,
        {
            **{f"standard:{name}": rows for name, rows in standard_partitions.items()},
            **{f"dedup:{name}": rows for name, rows in dedup_partitions.items()},
        },
    )
    diagnostics["loao_sender_disjoint"] = json.loads(dedup_path.read_text(encoding="utf-8"))
    diagnostics["threshold_procedures"] = threshold_procedures(table, summary, scores)
    diagnostics["sender_overlap"] = sender_overlap_breakdown(table, summary, scores)
    diagnostics["label_granularity"] = label_granularity_breakdown(
        table, summary, scores, outcomes
    )
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
        "lcccccccc",
        9,
        _label_granularity_table(summary, diagnostics),
    )
    _write(
        results_dir / "results_sender_overlap.tex",
        "lccccccccc",
        10,
        _sender_overlap_table(summary, diagnostics),
    )
    _write(
        results_dir / "results_loao_dedup.tex", "llcccccccc", 10, _dedup_table(summary, diagnostics)
    )
    _write(
        results_dir / "results_threshold_procedures.tex",
        "lcccccccc",
        9,
        _threshold_table(summary, diagnostics),
    )
    _write_macros(summary, sensitivity, diagnostics, meta, results_dir / "sensitivity_macros.tex")
    macros_path = results_dir / "sensitivity_macros.tex"
    macros_path.write_text(
        macros_path.read_text(encoding="utf-8")
        + "\n".join(_extra_macro_lines(summary, diagnostics))
        + "\n",
        encoding="utf-8",
    )
    ablation_path = results_dir / "ablation_summary.json"
    write_text_macros(
        summary,
        json.loads(ablation_path.read_text(encoding="utf-8")) if ablation_path.exists() else None,
        json.loads((results_dir / "dataset_capability_matrix.json").read_text(encoding="utf-8")),
        results_dir / "text_macros.tex",
    )
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
    if args.rerun or not (args.results_dir / "loao_sender_disjoint.json").exists():
        run_sender_disjoint_loao(table, summary, args.results_dir)
    analyze(table, summary, args.results_dir, args.data_dir, samples)
    _log("sensitivity analysis finished")


if __name__ == "__main__":
    main()
