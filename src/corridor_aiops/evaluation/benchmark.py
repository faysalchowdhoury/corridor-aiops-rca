from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.svm import OneClassSVM

from corridor_aiops.data.parse import veremi_events
from corridor_aiops.models.baselines import rule_scores

FEATURES = (
    "speed",
    "acceleration",
    "heading",
    "position_x",
    "position_y",
    "position_z",
)
MODEL_NAMES = ("isolation_forest", "one_class_svm", "rules")


def _event_matrix(events) -> np.ndarray:
    matrix = np.asarray(
        [[event.metrics.get(name, 0.0) for name in FEATURES] for event in events],
        dtype=float,
    )
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("VeReMi records contain non-finite or malformed numeric features")
    return matrix


def _robust_transform(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(train, axis=0)
    mad = np.median(np.abs(train - median), axis=0)
    mad[mad < 1e-8] = 1.0
    return (values - median) / mad, mad


def _bootstrap_metric_intervals(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    seed: int,
    samples: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    f1_values = []
    auc_values = []
    prediction = scores >= threshold
    for _ in range(samples):
        indexes = rng.integers(0, len(labels), size=len(labels))
        sample_labels = labels[indexes]
        if np.unique(sample_labels).size < 2:
            continue
        sample_scores = scores[indexes]
        sample_prediction = prediction[indexes]
        f1_values.append(float(f1_score(sample_labels, sample_prediction, zero_division=0)))
        auc_values.append(float(roc_auc_score(sample_labels, sample_scores)))
    if not f1_values:
        raise ValueError("Test split needs both attacker classes for bootstrap intervals")
    return {
        "f1_95pct_ci": [float(x) for x in np.quantile(f1_values, [0.025, 0.975])],
        "auroc_95pct_ci": [float(x) for x in np.quantile(auc_values, [0.025, 0.975])],
    }


def _partition_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predicted = scores >= threshold
    benign = labels == 0
    return {
        "f1_cyber_vs_benign": float(f1_score(labels, predicted, zero_division=0)),
        "auroc": float(roc_auc_score(labels, scores)),
        "false_alarm_rate_benign": (
            float(np.mean(predicted[benign])) if benign.any() else float("nan")
        ),
    }


def _seed_partitions(labels: np.ndarray, seed_count: int) -> list[np.ndarray]:
    partitions = []
    for seed in range(seed_count):
        rng = np.random.default_rng(seed)
        selected = []
        for label in np.unique(labels):
            class_rows = np.flatnonzero(labels == label)
            sample_size = max(1, int(0.8 * len(class_rows)))
            selected.extend(rng.choice(class_rows, size=sample_size, replace=False).tolist())
        partitions.append(np.asarray(sorted(selected), dtype=int))
    return partitions


def run_benchmark(
    *,
    data_dir: Path = Path("data/raw/veremi"),
    output_dir: Path = Path("results"),
    seeds: int = 20,
    bootstrap_samples: int = 1000,
) -> dict[str, Any]:
    if seeds < 2:
        raise ValueError("at least two deterministic evaluation partitions are required")
    if bootstrap_samples < 1:
        raise ValueError("at least one record-level bootstrap sample is required")
    archives = sorted(data_dir.glob("*.zip"))
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
    required = {"train", "validation", "test"}
    if not required.issubset(split_names):
        raise ValueError(
            "Published VeReMi train, validation, and test splits are required; "
            f"observed splits: {sorted(split_names)}"
        )

    train_events = [event for event in events if event.corridor_id.endswith("-train")]
    validation_events = [event for event in events if event.corridor_id.endswith("-validation")]
    test_events = [event for event in events if event.corridor_id.endswith("-test")]
    train_mask = np.asarray([event.label == "benign" for event in train_events], dtype=bool)
    validation_labels = np.asarray(
        [event.label == "cyber" for event in validation_events], dtype=int
    )
    test_labels = np.asarray([event.label == "cyber" for event in test_events], dtype=int)
    if not train_mask.any() or not (validation_labels == 0).any():
        raise ValueError(
            "Official train and validation splits must include benign attacker==0 records"
        )
    if np.unique(test_labels).size != 2:
        raise ValueError(
            "Official VeReMi test split must contain attacker==0 and attacker==1 records"
        )

    train_matrix = _event_matrix(train_events)
    validation_matrix = _event_matrix(validation_events)
    test_matrix = _event_matrix(test_events)
    train_benign = train_matrix[train_mask]
    train_scaled, _ = _robust_transform(train_benign, train_benign)
    validation_scaled, _ = _robust_transform(train_benign, validation_matrix)
    test_scaled, _ = _robust_transform(train_benign, test_matrix)

    forest = IsolationForest(random_state=42, n_estimators=100, contamination="auto").fit(
        train_scaled
    )
    svm = OneClassSVM(gamma="scale", nu=0.05).fit(train_scaled)
    scorers = {
        "isolation_forest": (
            lambda values: -forest.score_samples(values),
            validation_scaled,
            test_scaled,
        ),
        "one_class_svm": (
            lambda values: -svm.decision_function(values).reshape(-1),
            validation_scaled,
            test_scaled,
        ),
        "rules": (
            lambda values: rule_scores(values),
            validation_scaled,
            test_scaled,
        ),
    }
    val_benign = validation_labels == 0
    thresholds = {}
    test_scores = {}
    latencies = {}
    for name, (score, val_values, test_values) in scorers.items():
        val_scores = np.asarray(score(val_values), dtype=float)
        thresholds[name] = float(np.quantile(val_scores[val_benign], 0.95))
        started = time.perf_counter()
        test_scores[name] = np.asarray(score(test_values), dtype=float)
        latencies[name] = 1000 * (time.perf_counter() - started) / max(1, len(test_matrix))

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    model_summary = {}
    for name in MODEL_NAMES:
        scores = test_scores[name]
        threshold = thresholds[name]
        metrics = _partition_metrics(test_labels, scores, threshold)
        fpr, tpr, _ = roc_curve(test_labels, scores)
        precision, recall, _ = precision_recall_curve(test_labels, scores)
        summary = {
            **metrics,
            "average_precision": float(average_precision_score(test_labels, scores)),
            "batch_score_ms_per_event": latencies[name],
            "test_record_bootstrap_95pct_ci": _bootstrap_metric_intervals(
                test_labels, scores, threshold, seed=8675309, samples=bootstrap_samples
            ),
            "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))},
            "precision_recall": {"precision": precision.tolist(), "recall": recall.tolist()},
        }
        model_summary[name] = summary
        for seed, indexes in enumerate(_seed_partitions(test_labels, seeds)):
            seed_metrics = _partition_metrics(test_labels[indexes], scores[indexes], threshold)
            seed_rows.append({"seed": seed, "model": name, **seed_metrics})

    benign_mean = np.mean(test_scaled[test_labels == 0], axis=0)
    cyber_mean = np.mean(test_scaled[test_labels == 1], axis=0)
    feature_differences = {
        feature: float(abs(cyber - benign))
        for feature, benign, cyber in zip(FEATURES, benign_mean, cyber_mean, strict=True)
    }
    manifest_path = data_dir.parent / "manifest.json"
    source_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    )

    summary = {
        "study_type": "published VeReMi NextGen labeled replay traces",
        "dataset": "VeReMi NextGen InTAS highway 2",
        "dataset_doi": "https://doi.org/10.5281/zenodo.19665762",
        "official_splits": {
            "train_records": len(train_events),
            "validation_records": len(validation_events),
            "test_records": len(test_events),
        },
        "class_counts": {
            split: {
                "attacker_0_non_malicious": sum(event.label == "benign" for event in split_events),
                "attacker_1_misbehavior": sum(event.label == "cyber" for event in split_events),
            }
            for split, split_events in (
                ("train", train_events),
                ("validation", validation_events),
                ("test", test_events),
            )
        },
        "archive_record_counts": archive_record_counts,
        "context_dataset_manifest": source_manifest,
        "replay_partitions": seeds,
        "labels": {"attacker_0": "benign/non-malicious", "attacker_1": "cyber/misbehavior"},
        "work_zone_false_alarm_rate": None,
        "work_zone_metric_note": (
            "Not evaluated: current Caltrans planned closures are not time-aligned with the "
            "VeReMi traces and do not provide attack labels."
        ),
        "mttd_seconds": None,
        "root_cause_ac1": None,
        "root_cause_ac3": None,
        "unmeasured_metrics_note": (
            "VeReMi labels messages and attackers, not corridor intervention timing or causal "
            "root-cause nodes. No MTTD or root-cause accuracy is inferred from message labels."
        ),
        "not_evaluated_models": {
            "lstm_autoencoder": "Architecture scaffold only; no trained checkpoint.",
            "tabular_transformer": "Architecture scaffold only; no trained checkpoint.",
            "corridorlogtransformer_rca": (
                "No trained model or synchronized controller/incident text corpus is available."
            ),
        },
        "standardized_feature_mean_difference": feature_differences,
        "models": model_summary,
    }
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (output_dir / "seed_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)
    with (output_dir / "results_table.tex").open("w", encoding="utf-8") as stream:
        stream.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        stream.write("Model & F1 & AUROC & FAR (\\%) & ms/record \\\\\n\\midrule\n")
        for name, model in model_summary.items():
            latex_name = name.replace("_", "\\_")
            stream.write(
                f"{latex_name} & {model['f1_cyber_vs_benign']:.3f} & "
                f"{model['auroc']:.3f} & "
                f"{100 * model['false_alarm_rate_benign']:.1f}\\% & "
                f"{model['batch_score_ms_per_event']:.4f} \\\\\n"
            )
        stream.write("\\bottomrule\n\\end{tabular}\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate official VeReMi dataset splits.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/veremi"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    result = run_benchmark(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seeds=args.seeds,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "models"}, indent=2))
    for name, metrics in result["models"].items():
        print(f"{name}: AUROC={metrics['auroc']:.3f}, F1={metrics['f1_cyber_vs_benign']:.3f}")


if __name__ == "__main__":
    main()
