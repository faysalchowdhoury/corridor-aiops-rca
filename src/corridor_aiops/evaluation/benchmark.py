from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.svm import OneClassSVM

from corridor_aiops.data.parse import veremi_events
from corridor_aiops.data.trajectory import (
    FEATURE_NAMES,
    TrajectoryWindow,
    extract_trajectory_windows,
)
from corridor_aiops.evaluation.sequence_models import SequenceSplit, fit_sequence_baselines
from corridor_aiops.models.baselines import rule_scores

MODEL_NAMES = (
    "isolation_forest",
    "one_class_svm",
    "robust_threshold_rules",
    "lstm_autoencoder",
    "tabular_transformer",
)


def _official_split(events, split: str):
    return [event for event in events if event.corridor_id.endswith(f"-{split}")]


def _window_matrix(
    windows: list[TrajectoryWindow],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray([window.values[-1] for window in windows], dtype=float)
    observed = np.asarray([window.feature_mask[-1] for window in windows], dtype=bool)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Trajectory feature extraction produced malformed or non-finite data")
    return values, observed


def _sequence_split(
    windows: list[TrajectoryWindow],
    labels: np.ndarray,
    medians: np.ndarray,
    scales: np.ndarray,
) -> SequenceSplit:
    values = np.asarray([window.values for window in windows], dtype=float)
    feature_mask = np.asarray([window.feature_mask for window in windows], dtype=bool)
    normalized = (values - medians) / scales
    normalized[~feature_mask] = 0.0
    step_mask = np.asarray([window.time_mask for window in windows], dtype=bool)
    return SequenceSplit(normalized, feature_mask, step_mask, labels)


def _fit_robust_transform(
    train: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    medians = np.zeros(train.shape[1], dtype=float)
    scales = np.ones(train.shape[1], dtype=float)
    for column in range(train.shape[1]):
        values = train[train_mask[:, column], column]
        if values.size == 0:
            continue
        medians[column] = np.median(values)
        deviation = np.median(np.abs(values - medians[column]))
        if deviation > 1e-8:
            scales[column] = deviation * 1.4826
    return medians, scales


def _transform(
    values: np.ndarray,
    observed: np.ndarray,
    medians: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    normalized = (values - medians) / scales
    normalized[~observed] = 0.0
    return np.concatenate((normalized, observed.astype(float)), axis=1)


def _best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size != 2:
        raise ValueError("Validation split must contain both non-malicious and attacker labels")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        raise ValueError("Validation scores do not provide a classification threshold")
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def _metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    prediction = scores >= threshold
    benign = labels == 0
    if np.unique(labels[sample_weight > 0] if sample_weight is not None else labels).size != 2:
        raise ValueError("Metric sample must contain both attacker classes")
    if sample_weight is None:
        false_alarm_rate = float(np.mean(prediction[benign]))
    else:
        benign_weights = sample_weight[benign]
        false_alarm_rate = float(
            np.sum(prediction[benign] * benign_weights) / np.sum(benign_weights)
        )
    return {
        "f1_cyber_vs_benign": float(
            f1_score(labels, prediction, sample_weight=sample_weight, zero_division=0)
        ),
        "auroc": float(roc_auc_score(labels, scores, sample_weight=sample_weight)),
        "false_alarm_rate_non_malicious": false_alarm_rate,
    }


def _source_cluster_bootstrap(
    labels: np.ndarray,
    source_ids: np.ndarray,
    scores_by_model: dict[str, np.ndarray],
    thresholds: dict[str, float],
    *,
    seed: int,
    samples: int,
    seed_rows: int,
) -> tuple[dict[str, dict[str, list[float]]], list[dict[str, Any]]]:
    unique_sources, row_groups = np.unique(source_ids, return_inverse=True)
    if unique_sources.size < 2:
        raise ValueError("Source-cluster bootstrap requires at least two test senders")
    rng = np.random.default_rng(seed)
    values: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in scores_by_model
    }
    rows = []
    for replicate in range(samples):
        selected = rng.integers(0, unique_sources.size, size=unique_sources.size)
        cluster_weights = np.bincount(selected, minlength=unique_sources.size)
        row_weights = cluster_weights[row_groups].astype(float)
        if row_weights.sum() == 0:
            continue
        for name, scores in scores_by_model.items():
            try:
                metrics = _metrics(labels, scores, thresholds[name], row_weights)
            except ValueError:
                continue
            for metric, value in metrics.items():
                values[name][metric].append(value)
            if replicate < seed_rows:
                rows.append(
                    {
                        "seed": replicate,
                        "resampling_unit": "test_sender_id_cluster",
                        "model": name,
                        **metrics,
                    }
                )
    intervals: dict[str, dict[str, list[float]]] = {}
    for name, metrics in values.items():
        if not metrics["auroc"]:
            raise ValueError("No valid source-cluster bootstrap samples were produced")
        intervals[name] = {
            metric: [
                float(np.quantile(observations, 0.025)),
                float(np.quantile(observations, 0.975)),
            ]
            for metric, observations in metrics.items()
        }
    return intervals, rows


def _emitter_recall_at_k(
    events,
    scores: np.ndarray,
    *,
    k: int,
) -> float | None:
    emitters: dict[tuple[str, str], tuple[float, bool]] = {}
    for event, score in zip(events, scores, strict=True):
        key = (event.corridor_id, event.source_id)
        previous = emitters.get(key)
        emitters[key] = (
            max(float(score), previous[0]) if previous else float(score),
            event.label == "cyber" or (previous is not None and previous[1]),
        )
    positives = sum(is_attacker for _, is_attacker in emitters.values())
    if positives == 0:
        return None
    ranked = sorted(emitters.values(), key=lambda item: item[0], reverse=True)
    found = sum(is_attacker for _, is_attacker in ranked[:k])
    return found / positives


def _simulated_mttd(
    events,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int | None]:
    timelines: dict[tuple[str, str], list[tuple[int, int, bool]]] = defaultdict(list)
    for event, score in zip(events, scores, strict=True):
        timestamp_ns = event.receive_time_ns
        if timestamp_ns is None:
            raise ValueError("VeReMi event is missing its integer simulator receive time")
        timelines[(event.corridor_id, event.source_id)].append(
            (timestamp_ns, int(event.label == "cyber"), bool(score >= threshold))
        )
    delays = []
    onsets = 0
    for timeline in timelines.values():
        timeline.sort(key=lambda item: item[0])
        previous_label: int | None = None
        onset_ns: int | None = None
        detected = False
        for timestamp_ns, label, predicted in timeline:
            if label == 1 and previous_label == 0:
                onset_ns = timestamp_ns
                detected = False
                onsets += 1
            if label == 1 and onset_ns is not None and predicted and not detected:
                delays.append((timestamp_ns - onset_ns) / 1e9)
                detected = True
            if label == 0:
                onset_ns = None
                detected = False
            previous_label = label
    return {
        "attacker_label_onsets": onsets,
        "detected_onsets": len(delays),
        "detection_coverage": len(delays) / onsets if onsets else None,
        "mean_detected_onset_delay_seconds": float(np.mean(delays)) if delays else None,
        "time_basis": (
        "VeReMi simulator receive_time_ns; test-entry positive sequences are "
        "left-censored; not field-incident MTTD"
        ),
    }


def _curve_data(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    max_points: int = 1500,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    roc_auc = float(roc_auc_score(labels, scores))
    roc_indexes = np.linspace(0, len(fpr) - 1, min(max_points, len(fpr)), dtype=int)
    pr_indexes = np.linspace(
        0,
        len(precision) - 1,
        min(max_points, len(precision)),
        dtype=int,
    )
    return (
        {"fpr": fpr[roc_indexes].tolist(), "tpr": tpr[roc_indexes].tolist(), "auroc": roc_auc},
        {
            "precision": precision[pr_indexes].tolist(),
            "recall": recall[pr_indexes].tolist(),
        },
        roc_auc,
    )


def run_benchmark(
    *,
    data_dir: Path = Path("data/raw/veremi"),
    output_dir: Path = Path("results"),
    seeds: int = 20,
    bootstrap_samples: int = 1000,
    epochs: int = 5,
) -> dict[str, Any]:
    if seeds < 1:
        raise ValueError("at least one source-cluster bootstrap row is required")
    if bootstrap_samples < seeds:
        raise ValueError("bootstrap_samples must be at least the requested seed row count")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    archives = sorted(Path(data_dir).glob("*.zip"))
    if not archives:
        raise FileNotFoundError(
            f"No official VeReMi ZIP files found in {data_dir}. Run `corridor-ingest` first."
        )

    events = []
    archive_record_counts = {}
    for archive in archives:
        archive_events = list(veremi_events(archive))
        archive_record_counts[archive.name] = len(archive_events)
        events.extend(archive_events)
    split_names = {event.corridor_id.rsplit("-", 1)[-1] for event in events}
    if not {"train", "validation", "test"}.issubset(split_names):
        raise ValueError(
            "Published VeReMi train, validation, and test splits are required; "
            f"observed splits: {sorted(split_names)}"
        )

    split_events = {
        split: _official_split(events, split) for split in ("train", "validation", "test")
    }
    if any(not rows for rows in split_events.values()):
        raise ValueError("Each official VeReMi split must contain parsed BSM events")
    split_windows = {
        split: extract_trajectory_windows(rows) for split, rows in split_events.items()
    }
    split_values = {}
    split_masks = {}
    for split in split_events:
        split_values[split], split_masks[split] = _window_matrix(split_windows[split])
    train_labels = np.asarray(
        [event.label == "cyber" for event in split_events["train"]], dtype=int
    )
    validation_labels = np.asarray(
        [event.label == "cyber" for event in split_events["validation"]], dtype=int
    )
    test_labels = np.asarray(
        [event.label == "cyber" for event in split_events["test"]], dtype=int
    )
    train_benign = train_labels == 0
    if not train_benign.any():
        raise ValueError("Official training split has no non-malicious observations")
    if np.unique(validation_labels).size != 2 or np.unique(test_labels).size != 2:
        raise ValueError("Official validation and test splits must contain both attacker classes")
    sender_overlap = {}
    for archive in archives:
        train_senders = {
            event.source_id
            for event in split_events["train"]
            if event.corridor_id == f"{archive.stem}-train"
        }
        test_senders = {
            event.source_id
            for event in split_events["test"]
            if event.corridor_id == f"{archive.stem}-test"
        }
        sender_overlap[archive.name] = len(train_senders & test_senders)

    medians, scales = _fit_robust_transform(
        split_values["train"][train_benign],
        split_masks["train"][train_benign],
    )
    matrices = {
        split: _transform(split_values[split], split_masks[split], medians, scales)
        for split in split_events
    }
    train_matrix = matrices["train"]
    validation_matrix = matrices["validation"]
    test_matrix = matrices["test"]

    benign_indices = np.flatnonzero(train_benign)
    rng = np.random.default_rng(42)
    svm_fit_indices = (
        rng.choice(benign_indices, size=10_000, replace=False)
        if len(benign_indices) > 10_000
        else benign_indices
    )
    forest = IsolationForest(
        random_state=42,
        n_estimators=100,
        contamination="auto",
        n_jobs=-1,
    )
    started = time.perf_counter()
    forest.fit(train_matrix[benign_indices])
    forest_fit_seconds = time.perf_counter() - started
    svm = OneClassSVM(gamma="scale", nu=0.05)
    started = time.perf_counter()
    svm.fit(train_matrix[svm_fit_indices])
    svm_fit_seconds = time.perf_counter() - started

    validation_scores = {
        "isolation_forest": -forest.score_samples(validation_matrix),
        "one_class_svm": -svm.decision_function(validation_matrix).reshape(-1),
        "robust_threshold_rules": rule_scores(validation_matrix[:, : len(FEATURE_NAMES)]),
    }
    sequence_splits = {
        split: _sequence_split(
            split_windows[split],
            np.asarray([event.label == "cyber" for event in split_events[split]], dtype=int),
            medians,
            scales,
        )
        for split in split_events
    }
    sequence_results = fit_sequence_baselines(
        sequence_splits["train"],
        sequence_splits["validation"],
        sequence_splits["test"],
        epochs=epochs,
        seed=42,
    )
    for name, result in sequence_results.items():
        validation_scores[name] = result["validation_scores"]
    thresholds = {
        name: _best_f1_threshold(validation_labels, scores)
        for name, scores in validation_scores.items()
    }
    test_scores = {}
    inference_ms_per_event = {
        name: result["inference_ms_per_event"]
        for name, result in sequence_results.items()
    }
    for name, result in sequence_results.items():
        test_scores[name] = result["test_scores"]
    for name, scorer in (
        ("isolation_forest", forest.score_samples),
        ("one_class_svm", svm.decision_function),
    ):
        started = time.perf_counter()
        output = scorer(test_matrix)
        elapsed = time.perf_counter() - started
        test_scores[name] = -np.asarray(output).reshape(-1)
        inference_ms_per_event[name] = elapsed * 1000.0 / len(split_events["test"])
    started = time.perf_counter()
    test_scores["robust_threshold_rules"] = rule_scores(
        test_matrix[:, : len(FEATURE_NAMES)]
    )
    inference_ms_per_event["robust_threshold_rules"] = (
        (time.perf_counter() - started) * 1000.0 / len(split_events["test"])
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_ids = np.asarray([event.source_id for event in split_events["test"]])
    bootstrap_intervals, seed_rows = _source_cluster_bootstrap(
        test_labels,
        source_ids,
        test_scores,
        thresholds,
        seed=8675309,
        samples=bootstrap_samples,
        seed_rows=seeds,
    )
    model_summary = {}
    for name in MODEL_NAMES:
        scores = test_scores[name]
        threshold = thresholds[name]
        roc, precision_recall, curve_auc = _curve_data(test_labels, scores)
        model_summary[name] = {
            **_metrics(test_labels, scores, threshold),
            "average_precision": float(average_precision_score(test_labels, scores)),
            "validation_f1_threshold": threshold,
            "inference_ms_per_event": inference_ms_per_event[name],
            "attacker_emitter_recall_at_1": _emitter_recall_at_k(
                split_events["test"], scores, k=1
            ),
            "attacker_emitter_recall_at_3": _emitter_recall_at_k(
                split_events["test"], scores, k=3
            ),
            "simulated_mttd": _simulated_mttd(
                split_events["test"],
                scores,
                threshold,
            ),
            "source_cluster_bootstrap_95pct_ci": bootstrap_intervals[name],
            "roc": roc,
            "precision_recall": precision_recall,
            "auroc_exact": curve_auc,
        }

    test_benign = test_labels == 0
    test_cyber = test_labels == 1
    test_senders = {
        (event.corridor_id, event.source_id) for event in split_events["test"]
    }
    test_attacker_senders = {
        (event.corridor_id, event.source_id)
        for event in split_events["test"]
        if event.label == "cyber"
    }
    feature_differences = {}
    for index, feature in enumerate(FEATURE_NAMES):
        benign_mask = split_masks["test"][test_benign, index]
        cyber_mask = split_masks["test"][test_cyber, index]
        benign_values = split_values["test"][test_benign, index][benign_mask]
        cyber_values = split_values["test"][test_cyber, index][cyber_mask]
        if benign_values.size and cyber_values.size:
            pooled_values = np.concatenate((benign_values, cyber_values))
            pooled_median = float(np.median(pooled_values))
            median_scale = float(
                np.median(np.abs(pooled_values - pooled_median)) * 1.4826
            )
            interquartile_scale = float(
                (np.quantile(pooled_values, 0.75) - np.quantile(pooled_values, 0.25))
                / 1.349
            )
            pooled_scale = max(median_scale, interquartile_scale, 1e-8)
            feature_differences[feature] = float(
                abs(np.median(cyber_values) - np.median(benign_values)) / pooled_scale
            )
        else:
            feature_differences[feature] = None
    manifest_path = Path(data_dir).parent / "manifest.json"
    source_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    )

    summary = {
        "study_type": "VeReMi NextGen simulated misbehavior detection",
        "dataset": "VeReMi NextGen InTAS highway 2",
        "dataset_doi": "https://doi.org/10.5281/zenodo.19665762",
        "observation_unit": "de-duplicated BSM receive event with a causal trajectory window",
        "feature_names": list(FEATURE_NAMES),
        "official_splits": {
            f"{split}_records": len(split_events[split]) for split in split_events
        },
        "class_counts": {
            split: {
                "attacker_0_non_malicious": int(np.sum(labels == 0)),
                "attacker_1_misbehavior": int(np.sum(labels == 1)),
            }
            for split, labels in (
                ("train", train_labels),
                ("validation", validation_labels),
                ("test", test_labels),
            )
        },
        "archive_unique_record_counts": archive_record_counts,
        "official_train_test_sender_id_overlap_by_archive": sender_overlap,
        "attacker_emitter_localization_protocol": {
            "test_sender_clusters": len(test_senders),
            "test_attacker_labeled_sender_clusters": len(test_attacker_senders),
            "metric": (
                "recall of attacker-labeled sender clusters among the top-k senders "
                "ranked by maximum message anomaly score over the test split"
            ),
            "interpretation": (
                "retrospective attacker-emitter ranking, not event-level component "
                "localization or causal root-cause accuracy"
            ),
        },
        "preprocessing": {
            "normalization_fit": "non-malicious official training records only",
            "missing_values": (
                "feature validity masks retained; missing values set to normalized zero"
            ),
            "split_policy": "official train/validation/test paths; no random test subsampling",
            "receiver_copy_policy": (
                "same sender/messageID/sendTime receiver copies collapsed by parser; "
                "earliest receive_time_ns retained"
            ),
        },
        "models": model_summary,
        "training": {
            "isolation_forest_fit_records": len(benign_indices),
            "one_class_svm_fit_records": len(svm_fit_indices),
            "isolation_forest_fit_seconds": forest_fit_seconds,
            "one_class_svm_fit_seconds": svm_fit_seconds,
            "threshold_selection": "maximum F1 on official validation split",
            "models_evaluated": list(MODEL_NAMES),
            "sequence_model_epochs": epochs,
            "sequence_models": {
                name: {
                    key: value
                    for key, value in result.items()
                    if key
                    not in {
                        "validation_scores",
                        "test_scores",
                        "inference_ms_per_event",
                    }
                }
                for name, result in sequence_results.items()
            },
        },
        "uncertainty": {
            "method": "paired nonparametric bootstrap resampling test sender IDs as clusters",
            "bootstrap_samples": bootstrap_samples,
            "seed_metrics_rows": min(seeds, bootstrap_samples),
            "seed_metrics_note": (
                "Rows are source-cluster bootstrap resamples, not independent corridor-days, "
                "random test partitions, or independent scenarios."
            ),
        },
        "scope_limits": {
            "dataset_attack_labels": "simulated VeReMi attacker/misbehavior labels",
            "attacker_emitter_localization": (
                "sender-level risk ranking only; not causal root-cause attribution"
            ),
            "simulated_mttd": (
                "message-level delay after a labeled simulator attacker transition; "
                "not field-incident MTTD"
            ),
            "work_zone_far": None,
            "causal_root_cause_ac1": None,
            "causal_root_cause_ac3": None,
            "cross_modal_fusion": None,
            "production_latency_sla": None,
        },
        "standardized_feature_median_separation": feature_differences,
        "context_dataset_manifest": source_manifest,
    }
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "seed_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "seed",
            "resampling_unit",
            "model",
            "f1_cyber_vs_benign",
            "auroc",
            "false_alarm_rate_non_malicious",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(seed_rows)
    with (output_dir / "results_table.tex").open("w", encoding="utf-8") as stream:
        stream.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        stream.write("Model & F1 & AUROC & FAR (\\%) & ms/event \\\\\n\\midrule\n")
        for name, metrics in model_summary.items():
            latex_name = name.replace("_", "\\_")
            stream.write(
                f"{latex_name} & {metrics['f1_cyber_vs_benign']:.3f} & "
                f"{metrics['auroc']:.3f} & "
                f"{100 * metrics['false_alarm_rate_non_malicious']:.2f} & "
                f"{metrics['inference_ms_per_event']:.5f} \\\\\n"
            )
        stream.write("\\bottomrule\n\\end{tabular}\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate de-duplicated VeReMi events on official data splits."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/veremi"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    result = run_benchmark(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seeds=args.seeds,
        bootstrap_samples=args.bootstrap_samples,
        epochs=args.epochs,
    )
    print(
        json.dumps(
            {
                "official_splits": result["official_splits"],
                "models": {
                    name: {
                        "auroc": metrics["auroc"],
                        "f1_cyber_vs_benign": metrics["f1_cyber_vs_benign"],
                    }
                    for name, metrics in result["models"].items()
                },
                "uncertainty": result["uncertainty"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
