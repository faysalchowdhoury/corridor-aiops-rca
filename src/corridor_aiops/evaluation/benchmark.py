"""Official-split VeReMi misbehavior-detection benchmark.

Every bootstrap and overlap computation in this module identifies a sender cluster by
the composite key ``(corridor_id, source_id)``, where ``corridor_id`` is
``<archive>-<split>``. Training-seed variability and sender-cluster bootstrap
uncertainty are computed and reported separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.svm import OneClassSVM

from corridor_aiops.data.parse import veremi_archive_identity, veremi_events
from corridor_aiops.data.trajectory import (
    CAUSAL_JITTER_DEFINITION,
    FEATURE_NAMES,
    LOCAL_DENSITY_DEFINITION,
    RAW_COORDINATE_FEATURE_NAMES,
    TIMING_FEATURES,
    TRANSLATION_INVARIANT_KINEMATIC_FEATURES,
    extract_trajectory_windows,
)
from corridor_aiops.evaluation.sequence_models import (
    SequenceSplit,
    fit_lstm_autoencoder,
    fit_supervised_mlp,
    fit_tabular_transformer,
)
from corridor_aiops.models.baselines import rule_scores

UNSUPERVISED_MODELS = (
    "robust_threshold_rules",
    "isolation_forest",
    "one_class_svm",
    "lstm_autoencoder",
)
SUPERVISED_MODELS = (
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
    "xgboost",
    "supervised_mlp",
    "tabular_transformer",
)
REFERENCE_MODEL = "tabular_transformer"
DISPLAY_NAMES = {
    "robust_threshold_rules": "Robust rules",
    "isolation_forest": "Isolation forest",
    "one_class_svm": "One-class SVM",
    "lstm_autoencoder": "LSTM autoencoder",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Hist. gradient boosting",
    "xgboost": "XGBoost",
    "supervised_mlp": "Supervised MLP",
    "tabular_transformer": "Tabular Transformer",
}
FAR_BUDGETS = (0.01, 0.05)
SEED_BASE = 42
BOOTSTRAP_SEED = 8675309
SENDER_HOLDOUT_SALT = "corridor-aiops-sender-disjoint-v1"
SENDER_HOLDOUT_FRACTIONS = (0.6, 0.2, 0.2)
TIMING_KEYS = frozenset({"inference_ms_per_event", "training_seconds"})
SPLITS = ("train", "validation", "test")


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
    except ImportError:
        return False
    return True


def regime(model: str) -> str:
    if model in UNSUPERVISED_MODELS:
        return "unsupervised_or_one_class"
    if model in SUPERVISED_MODELS:
        return "supervised"
    raise ValueError(f"Unknown model: {model}")


# ---------------------------------------------------------------------------
# Data and feature configurations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventTable:
    events: list
    values: np.ndarray
    feature_mask: np.ndarray
    step_mask: np.ndarray
    labels: np.ndarray
    corridor: np.ndarray
    source: np.ndarray
    archive: np.ndarray
    split: np.ndarray
    feature_names: tuple[str, ...]

    def rows(self, **conditions: str) -> np.ndarray:
        selected = np.ones(len(self.labels), dtype=bool)
        for field, value in conditions.items():
            selected &= getattr(self, field) == value
        return np.flatnonzero(selected)


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    feature_names: tuple[str, ...]
    window: int
    use_masks: bool
    description: str


FULL_WINDOW = 10
FULL_CONFIG = FeatureConfig(
    "full_causal_window_L10",
    FEATURE_NAMES,
    FULL_WINDOW,
    True,
    "11 translation-invariant causal features, L = 10, feature and step validity masks",
)


def _without(names: Sequence[str]) -> tuple[str, ...]:
    return tuple(name for name in FEATURE_NAMES if name not in names)


ABLATION_CONFIGS = (
    (
        "window_length",
        FeatureConfig(
            "single_message_L1",
            FEATURE_NAMES,
            1,
            True,
            "Same 11 features, current message only (L = 1)",
        ),
    ),
    (
        "coordinate_representation",
        FeatureConfig(
            "raw_coordinates",
            _without(TRANSLATION_INVARIANT_KINEMATIC_FEATURES) + RAW_COORDINATE_FEATURE_NAMES,
            FULL_WINDOW,
            True,
            "Translation-invariant displacement, vertical step, speed change, and heading "
            "change replaced by absolute position x, y, z and absolute heading",
        ),
    ),
    (
        "timing_and_jitter",
        FeatureConfig(
            "without_timing_jitter",
            _without(TIMING_FEATURES),
            FULL_WINDOW,
            True,
            "Inter-arrival time, causal jitter, duplicate-timestamp, and long-gap "
            "indicators removed",
        ),
    ),
    (
        "local_density",
        FeatureConfig(
            "without_local_density",
            _without(("local_density_150m",)),
            FULL_WINDOW,
            True,
            "Local 150 m sender density removed",
        ),
    ),
    (
        "validity_masks",
        FeatureConfig(
            "without_validity_masks",
            FEATURE_NAMES,
            FULL_WINDOW,
            False,
            "Missing values imputed as the normalized benign median (zero) with no "
            "feature-validity or step-padding mask supplied to the model",
        ),
    ),
)


def build_event_table(events: list, archive_of_event: Sequence[str]) -> EventTable:
    """Causal windows and arrays for parsed events (windows never cross corridor_id)."""
    windows = extract_trajectory_windows(
        events, window_size=FULL_WINDOW, include_raw_coordinates=True
    )
    values = np.asarray([window.values for window in windows], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Trajectory feature extraction produced non-finite data")
    return EventTable(
        events=events,
        values=values,
        feature_mask=np.asarray([window.feature_mask for window in windows], dtype=bool),
        step_mask=np.asarray([window.time_mask for window in windows], dtype=bool),
        labels=np.asarray([event.label == "cyber" for event in events], dtype=np.int64),
        corridor=np.asarray([event.corridor_id for event in events]),
        source=np.asarray([event.source_id for event in events]),
        archive=np.asarray(archive_of_event),
        split=np.asarray([event.corridor_id.rsplit("-", 1)[-1] for event in events]),
        feature_names=FEATURE_NAMES + RAW_COORDINATE_FEATURE_NAMES,
    )


def load_event_table(data_dir: Path) -> tuple[EventTable, dict[str, int]]:
    archives = sorted(Path(data_dir).glob("*.zip"))
    if not archives:
        raise FileNotFoundError(
            f"No official VeReMi ZIP files found in {data_dir}. Run `corridor-ingest` first."
        )
    events = []
    archive_of_event = []
    record_counts = {}
    for archive in archives:
        _, category = veremi_archive_identity(archive)
        archive_events = list(veremi_events(archive))
        record_counts[archive.name] = len(archive_events)
        events.extend(archive_events)
        archive_of_event.extend([category] * len(archive_events))
    table = build_event_table(events, archive_of_event)
    observed = set(table.split.tolist())
    if not set(SPLITS).issubset(observed):
        raise ValueError(f"Official train, validation, and test splits required; saw {observed}")
    return table, record_counts


def _fit_robust_transform(
    train: np.ndarray, train_mask: np.ndarray
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


@dataclass(frozen=True)
class Representation:
    config: FeatureConfig
    normalized: np.ndarray
    feature_mask: np.ndarray
    step_mask: np.ndarray
    medians: np.ndarray
    scales: np.ndarray

    def last_step(self, rows: np.ndarray) -> np.ndarray:
        values = self.normalized[rows, -1]
        if not self.config.use_masks:
            return values
        return np.concatenate((values, self.feature_mask[rows, -1].astype(np.float32)), axis=1)

    def rule_input(self, rows: np.ndarray) -> np.ndarray:
        return self.normalized[rows, -1]

    def flat(self, rows: np.ndarray) -> np.ndarray:
        values = self.normalized[rows].reshape(len(rows), -1)
        if not self.config.use_masks:
            return values
        masks = self.feature_mask[rows].reshape(len(rows), -1).astype(np.float32)
        return np.concatenate((values, masks), axis=1)

    def sequence(self, rows: np.ndarray, labels: np.ndarray) -> SequenceSplit:
        return SequenceSplit(
            self.normalized[rows],
            self.feature_mask[rows],
            self.step_mask[rows],
            labels[rows],
        )

    def preprocessing_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps(self.config.__dict__, sort_keys=True).encode("utf-8"))
        digest.update(self.medians.astype(np.float64).tobytes())
        digest.update(self.scales.astype(np.float64).tobytes())
        return digest.hexdigest()


def build_representation(
    table: EventTable, config: FeatureConfig, benign_fit_rows: np.ndarray
) -> Representation:
    """Select columns and window length, then robust-scale on benign training rows only."""
    columns = [table.feature_names.index(name) for name in config.feature_names]
    values = table.values[:, -config.window :, :][:, :, columns]
    feature_mask = table.feature_mask[:, -config.window :, :][:, :, columns]
    step_mask = table.step_mask[:, -config.window :]
    medians, scales = _fit_robust_transform(
        values[benign_fit_rows, -1].astype(float), feature_mask[benign_fit_rows, -1]
    )
    normalized = ((values - medians) / scales).astype(np.float32)
    normalized[~feature_mask] = 0.0
    if not config.use_masks:
        feature_mask = np.ones_like(feature_mask)
        step_mask = np.ones_like(step_mask)
    return Representation(config, normalized, feature_mask, step_mask, medians, scales)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _pickle_sha256(model: Any) -> str:
    return hashlib.sha256(pickle.dumps(model, protocol=4)).hexdigest()


def _timed_scores(scorer: Callable[[np.ndarray], np.ndarray], matrix: np.ndarray):
    started = time.perf_counter()
    scores = np.asarray(scorer(matrix), dtype=float).reshape(-1)
    return scores, (time.perf_counter() - started) * 1000.0 / max(len(matrix), 1)


def fit_model(
    name: str,
    representation: Representation,
    labels: np.ndarray,
    split_rows: dict[str, np.ndarray],
    *,
    seed: int,
    epochs: int,
) -> dict[str, Any]:
    """Fit one model; return validation/test scores, checkpoint hash, and metadata."""
    train, validation, test = (split_rows[split] for split in SPLITS)
    benign_train = train[labels[train] == 0]
    result: dict[str, Any]
    if name == "robust_threshold_rules":
        validation_scores = rule_scores(representation.rule_input(validation))
        test_scores, inference = _timed_scores(rule_scores, representation.rule_input(test))
        result = {
            "validation_scores": validation_scores,
            "test_scores": test_scores,
            "inference_ms_per_event": inference,
            "training_seconds": 0.0,
            "training_status": "fixed_rule_threshold_3_on_benign_robust_scale",
            "training_records": int(len(benign_train)),
            "checkpoint_sha256": hashlib.sha256(b"robust_threshold_rules:3.0").hexdigest(),
        }
    elif name in {"isolation_forest", "one_class_svm"}:
        fit_rows = benign_train
        if name == "one_class_svm" and len(fit_rows) > 10_000:
            fit_rows = np.sort(np.random.default_rng(seed).choice(fit_rows, 10_000, replace=False))
        model = (
            IsolationForest(random_state=seed, n_estimators=100, contamination="auto", n_jobs=-1)
            if name == "isolation_forest"
            else OneClassSVM(gamma="scale", nu=0.05)
        )
        started = time.perf_counter()
        model.fit(representation.last_step(fit_rows))
        training_seconds = time.perf_counter() - started
        scorer = model.score_samples if name == "isolation_forest" else model.decision_function
        validation_scores = -np.asarray(scorer(representation.last_step(validation))).reshape(-1)
        test_scores, inference = _timed_scores(
            lambda matrix: -np.asarray(scorer(matrix)), representation.last_step(test)
        )
        result = {
            "validation_scores": validation_scores,
            "test_scores": test_scores,
            "inference_ms_per_event": inference,
            "training_seconds": training_seconds,
            "training_status": "fit_on_non_malicious_training_records_current_step",
            "training_records": int(len(fit_rows)),
            "checkpoint_sha256": _pickle_sha256(model),
        }
    elif name == "lstm_autoencoder":
        result = fit_lstm_autoencoder(
            representation.sequence(train, labels),
            representation.sequence(validation, labels),
            representation.sequence(test, labels),
            epochs=epochs,
            seed=seed,
        )
    elif name == "tabular_transformer":
        result = fit_tabular_transformer(
            representation.sequence(train, labels),
            representation.sequence(validation, labels),
            representation.sequence(test, labels),
            epochs=epochs,
            seed=seed,
        )
    elif name == "supervised_mlp":
        result = fit_supervised_mlp(
            (representation.flat(train), labels[train]),
            (representation.flat(validation), labels[validation]),
            (representation.flat(test), labels[test]),
            epochs=epochs,
            seed=seed,
        )
    elif name in {"logistic_regression", "random_forest", "hist_gradient_boosting", "xgboost"}:
        if name == "logistic_regression":
            model = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        elif name == "random_forest":
            model = RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            )
        elif name == "hist_gradient_boosting":
            model = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.1,
                early_stopping=False,
                class_weight="balanced",
                random_state=seed,
            )
        else:
            from xgboost import XGBClassifier

            positives = max(int(np.sum(labels[train] == 1)), 1)
            model = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=float(np.sum(labels[train] == 0)) / positives,
                random_state=seed,
                n_jobs=os.cpu_count(),
                tree_method="hist",
            )
        started = time.perf_counter()
        model.fit(representation.flat(train), labels[train])
        training_seconds = time.perf_counter() - started
        validation_scores = model.predict_proba(representation.flat(validation))[:, 1]
        test_scores, inference = _timed_scores(
            lambda matrix: model.predict_proba(matrix)[:, 1], representation.flat(test)
        )
        result = {
            "validation_scores": validation_scores,
            "test_scores": test_scores,
            "inference_ms_per_event": inference,
            "training_seconds": training_seconds,
            "training_status": "supervised_on_training_labels_flattened_causal_window",
            "training_records": int(len(train)),
            "checkpoint_sha256": _pickle_sha256(model),
        }
    else:
        raise ValueError(f"Unknown model: {name}")
    result["validation_auroc"] = float(
        roc_auc_score(labels[validation], result["validation_scores"])
    )
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size != 2:
        raise ValueError("Validation split must contain both non-malicious and attacker labels")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        raise ValueError("Validation scores do not provide a classification threshold")
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def far_budget_threshold(benign_scores: np.ndarray, budget: float) -> float:
    """Smallest threshold whose alert rate on ``benign_scores`` does not exceed budget.

    Alerts are ``score >= threshold``. Ties are resolved conservatively.
    """
    if not 0.0 <= budget < 1.0:
        raise ValueError("budget must be in [0, 1)")
    ordered = np.sort(np.asarray(benign_scores, dtype=float))[::-1]
    if ordered.size == 0:
        raise ValueError("FAR calibration requires non-malicious scores")
    allowed = int(np.floor(budget * ordered.size))
    if allowed >= ordered.size:
        return float(-np.inf)
    return float(np.nextafter(ordered[allowed], np.inf))


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


def evaluate_operating_points(
    validation_labels: np.ndarray,
    validation_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
) -> dict[str, Any]:
    """Validation-selected F1 threshold, AP, and matched-FAR operating points."""
    threshold = _best_f1_threshold(validation_labels, validation_scores)
    metrics: dict[str, Any] = {
        **_metrics(test_labels, test_scores, threshold),
        "average_precision": float(average_precision_score(test_labels, test_scores)),
        "validation_f1_threshold": threshold,
    }
    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    positive = test_labels == 1
    for budget in FAR_BUDGETS:
        key = f"far_{budget * 100:.1f}pct"
        budget_threshold = far_budget_threshold(validation_scores[validation_labels == 0], budget)
        alerts = test_scores >= budget_threshold
        metrics[f"test_roc_tpr_at_{key}"] = float(np.max(tpr[fpr <= budget + 1e-12]))
        metrics[f"validation_calibrated_recall_at_{key}"] = float(np.mean(alerts[positive]))
        metrics[f"validation_calibrated_realized_far_at_{key}"] = float(np.mean(alerts[~positive]))
    return metrics


def _emitter_recall_at_k(events, scores: np.ndarray, *, k: int) -> float | None:
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
    return sum(is_attacker for _, is_attacker in ranked[:k]) / positives


def _simulated_mttd(events, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    timelines: dict[tuple[str, str], list[tuple[int, int, bool]]] = defaultdict(list)
    for event, score in zip(events, scores, strict=True):
        if event.receive_time_ns is None:
            raise ValueError("VeReMi event is missing its integer simulator receive time")
        timelines[(event.corridor_id, event.source_id)].append(
            (event.receive_time_ns, int(event.label == "cyber"), bool(score >= threshold))
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


def _curve_data(labels: np.ndarray, scores: np.ndarray, max_points: int = 1500):
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    roc_index = np.linspace(0, len(fpr) - 1, min(max_points, len(fpr)), dtype=int)
    pr_index = np.linspace(0, len(precision) - 1, min(max_points, len(precision)), dtype=int)
    return (
        {"fpr": fpr[roc_index].tolist(), "tpr": tpr[roc_index].tolist()},
        {"precision": precision[pr_index].tolist(), "recall": recall[pr_index].tolist()},
    )


def summarize_seeds(per_seed: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean and sample standard deviation (ddof = 1) of every numeric metric."""
    summary: dict[str, dict[str, float]] = {}
    keys = [
        key
        for key, value in per_seed[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    for key in keys:
        values = np.asarray([row[key] for row in per_seed if row.get(key) is not None], dtype=float)
        if values.size == 0:
            continue
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "n_seeds": int(values.size),
        }
    return summary


# ---------------------------------------------------------------------------
# Composite sender-cluster bootstrap
# ---------------------------------------------------------------------------


def composite_cluster_codes(
    corridor_ids: Sequence[str], source_ids: Sequence[str]
) -> tuple[np.ndarray, int]:
    """Integer cluster code per row for the composite key (corridor_id, source_id)."""
    lookup: dict[tuple[str, str], int] = {}
    codes = np.empty(len(source_ids), dtype=np.int64)
    for index, key in enumerate(zip(corridor_ids, source_ids, strict=True)):
        codes[index] = lookup.setdefault((str(key[0]), str(key[1])), len(lookup))
    return codes, len(lookup)


class _WeightedScorer:
    """Exact weighted F1, AUROC (ties count one half), and FAR for one score vector."""

    def __init__(self, labels: np.ndarray, scores: np.ndarray, threshold: float) -> None:
        self.positive = labels.astype(float)
        self.negative = 1.0 - self.positive
        _, self.groups = np.unique(scores, return_inverse=True)
        self.group_count = int(self.groups.max()) + 1
        self.prediction = (scores >= threshold).astype(float)

    def __call__(self, weights: np.ndarray) -> dict[str, float] | None:
        positive_weight = weights * self.positive
        negative_weight = weights * self.negative
        total_positive = positive_weight.sum()
        total_negative = negative_weight.sum()
        if total_positive <= 0 or total_negative <= 0:
            return None
        group_positive = np.bincount(self.groups, positive_weight, self.group_count)
        group_negative = np.bincount(self.groups, negative_weight, self.group_count)
        negatives_below = np.cumsum(group_negative) - group_negative
        auroc = float(
            np.sum(group_positive * (negatives_below + 0.5 * group_negative))
            / (total_positive * total_negative)
        )
        true_positive = float(np.sum(positive_weight * self.prediction))
        false_positive = float(np.sum(negative_weight * self.prediction))
        false_negative = float(total_positive - true_positive)
        denominator = 2 * true_positive + false_positive + false_negative
        return {
            "f1_cyber_vs_benign": 2 * true_positive / denominator if denominator else 0.0,
            "auroc": auroc,
            "false_alarm_rate_non_malicious": false_positive / float(total_negative),
        }


def composite_cluster_bootstrap(
    labels: np.ndarray,
    corridor_ids: Sequence[str],
    source_ids: Sequence[str],
    scores_by_model: dict[str, np.ndarray],
    thresholds: dict[str, float],
    *,
    reference: str | None,
    seed: int,
    samples: int,
    preview_rows: int = 0,
) -> dict[str, Any]:
    """Paired nonparametric bootstrap over composite (corridor_id, source_id) clusters.

    Every model is scored on the same cluster resample. A replicate without both
    classes is skipped and counted. Paired differences are ``reference - model``.
    """
    codes, cluster_count = composite_cluster_codes(corridor_ids, source_ids)
    if cluster_count < 2:
        raise ValueError("Sender-cluster bootstrap requires at least two composite clusters")
    scorers = {
        name: _WeightedScorer(labels, scores, thresholds[name])
        for name, scores in scores_by_model.items()
    }
    rng = np.random.default_rng(seed)
    observations: dict[str, dict[str, list[float]]] = {name: defaultdict(list) for name in scorers}
    differences: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in scorers if name != reference
    }
    valid = 0
    skipped = 0
    preview = []
    for replicate in range(samples):
        draws = rng.integers(0, cluster_count, size=cluster_count)
        weights = np.bincount(draws, minlength=cluster_count)[codes].astype(float)
        results = {name: scorer(weights) for name, scorer in scorers.items()}
        if any(result is None for result in results.values()):
            skipped += 1
            continue
        valid += 1
        for name, metrics in results.items():
            for metric, value in metrics.items():
                observations[name][metric].append(value)
            if replicate < preview_rows:
                preview.append(
                    {
                        "replicate": replicate,
                        "resampling_unit": "composite_corridor_id_source_id_cluster",
                        "model": name,
                        **metrics,
                    }
                )
        if reference is not None and reference in results:
            for name in differences:
                for metric in ("f1_cyber_vs_benign", "auroc"):
                    differences[name][metric].append(
                        results[reference][metric] - results[name][metric]
                    )
    if valid == 0:
        raise ValueError("No valid sender-cluster bootstrap replicates were produced")

    def interval(values: list[float]) -> list[float]:
        return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]

    return {
        "method": (
            "paired nonparametric bootstrap resampling composite (corridor_id, source_id) "
            "test clusters with replacement; percentile 95% intervals"
        ),
        "composite_clusters": cluster_count,
        "requested_replicates": samples,
        "valid_replicates": valid,
        "skipped_replicates": skipped,
        "skip_rule": "replicate skipped when it lacks either class for any model",
        "reference_model": reference,
        "intervals_95pct": {
            name: {metric: interval(values) for metric, values in metrics.items()}
            for name, metrics in observations.items()
        },
        "paired_differences_reference_minus_model": {
            name: {
                f"delta_{metric}": {
                    "mean": float(np.mean(values)),
                    "ci_95pct": interval(values),
                    "fraction_of_replicates_reference_better": float(
                        np.mean(np.asarray(values) > 0)
                    ),
                }
                for metric, values in metrics.items()
            }
            for name, metrics in differences.items()
        },
        "_preview_rows": preview,
    }


# ---------------------------------------------------------------------------
# Evaluation contexts
# ---------------------------------------------------------------------------


def split_eligibility(labels: np.ndarray, split_rows: dict[str, np.ndarray]) -> str | None:
    """Return a NOT_EVALUABLE reason, or None when train/validation/test are usable."""
    for split in SPLITS:
        rows = split_rows.get(split)
        if rows is None or len(rows) == 0:
            return f"{split} partition is empty"
        classes = set(np.unique(labels[rows]).tolist())
        if classes != {0, 1}:
            return f"{split} partition lacks both attacker classes (observed {sorted(classes)})"
    return None


def run_models(
    table: EventTable,
    split_rows: dict[str, np.ndarray],
    models: Sequence[str],
    *,
    config: FeatureConfig,
    seeds: Sequence[int],
    epochs: int,
    label: str,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], Representation]:
    """Fit every model for every seed; keep raw scores per seed for later analysis."""
    train = split_rows["train"]
    representation = build_representation(table, config, train[table.labels[train] == 0])
    labels = table.labels
    validation_labels = labels[split_rows["validation"]]
    test_labels = labels[split_rows["test"]]
    summaries: dict[str, Any] = {}
    raw: dict[str, list[dict[str, Any]]] = {}
    for name in models:
        per_seed = []
        raw[name] = []
        for seed in seeds:
            _log(f"{label}: {name} seed {seed}")
            fitted = fit_model(name, representation, labels, split_rows, seed=seed, epochs=epochs)
            metrics = evaluate_operating_points(
                validation_labels,
                fitted["validation_scores"],
                test_labels,
                fitted["test_scores"],
            )
            record = {
                "seed": seed,
                **metrics,
                "validation_auroc": fitted["validation_auroc"],
                "checkpoint_sha256": fitted["checkpoint_sha256"],
                "training_seconds": fitted["training_seconds"],
                "inference_ms_per_event": fitted["inference_ms_per_event"],
                "training_status": fitted["training_status"],
                "training_records": fitted["training_records"],
            }
            for key in ("epochs_run", "selected_epoch", "selected_validation_auroc"):
                if key in fitted:
                    record[key] = fitted[key]
            per_seed.append(record)
            raw[name].append(fitted)
        summaries[name] = {
            "supervision_regime": regime(name),
            "display_name": DISPLAY_NAMES[name],
            "per_seed": per_seed,
            "mean_std_across_training_seeds": summarize_seeds(per_seed),
        }
    return summaries, raw, representation


def _split_counts(labels: np.ndarray, split_rows: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        split: {
            "records": int(len(rows)),
            "attacker_0_non_malicious": int(np.sum(labels[rows] == 0)),
            "attacker_1_misbehavior": int(np.sum(labels[rows] == 1)),
        }
        for split, rows in split_rows.items()
    }


def sender_disjoint_partition(table: EventTable) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Deterministic 60/20/20 partition of bare source_ids, stratified by attacker status.

    Partitioning by bare ``source_id`` across both archives and all official splits
    implies zero composite ``(corridor_id, source_id)`` overlap between partitions.
    """
    attacker_ids = set(table.source[table.labels == 1].tolist())
    all_ids = sorted(set(table.source.tolist()))
    assignment: dict[str, str] = {}
    for stratum in (
        [sid for sid in all_ids if sid in attacker_ids],
        [sid for sid in all_ids if sid not in attacker_ids],
    ):
        ordered = sorted(
            stratum,
            key=lambda sid: hashlib.sha256(f"{SENDER_HOLDOUT_SALT}:{sid}".encode()).hexdigest(),
        )
        train_cut = int(round(SENDER_HOLDOUT_FRACTIONS[0] * len(ordered)))
        validation_cut = train_cut + int(round(SENDER_HOLDOUT_FRACTIONS[1] * len(ordered)))
        for position, sid in enumerate(ordered):
            assignment[sid] = (
                "train"
                if position < train_cut
                else "validation"
                if position < validation_cut
                else "test"
            )
    partition = np.asarray([assignment[sid] for sid in table.source.tolist()])
    split_rows = {split: np.flatnonzero(partition == split) for split in SPLITS}
    bare = {split: set(table.source[rows].tolist()) for split, rows in split_rows.items()}
    composite = {
        split: set(zip(table.corridor[rows].tolist(), table.source[rows].tolist(), strict=True))
        for split, rows in split_rows.items()
    }
    overlap = {
        "bare_source_id_train_test": len(bare["train"] & bare["test"]),
        "bare_source_id_train_validation": len(bare["train"] & bare["validation"]),
        "bare_source_id_validation_test": len(bare["validation"] & bare["test"]),
        "composite_train_test": len(composite["train"] & composite["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Sender-disjoint partition overlaps: {overlap}")
    return split_rows, {
        "partition_rule": (
            "sha256(salt:source_id) order within attacker-ever and never-attacker strata; "
            "60/20/20 train/validation/test"
        ),
        "salt": SENDER_HOLDOUT_SALT,
        "senders_by_partition": {split: len(ids) for split, ids in bare.items()},
        "composite_clusters_by_partition": {split: len(ids) for split, ids in composite.items()},
        "overlap": overlap,
        "temporal_note": (
            "Features are still computed causally within each official archive split, but "
            "the partition pools official time splits, so it tests unseen senders rather "
            "than future time."
        ),
    }


def _context_bootstrap(
    table: EventTable,
    split_rows: dict[str, np.ndarray],
    raw: dict[str, list[dict[str, Any]]],
    summaries: dict[str, Any],
    *,
    samples: int,
    preview_rows: int = 0,
) -> dict[str, Any]:
    test = split_rows["test"]
    scores = {name: runs[0]["test_scores"] for name, runs in raw.items()}
    thresholds = {name: summaries[name]["per_seed"][0]["validation_f1_threshold"] for name in raw}
    result = composite_cluster_bootstrap(
        table.labels[test],
        table.corridor[test],
        table.source[test],
        scores,
        thresholds,
        reference=REFERENCE_MODEL if REFERENCE_MODEL in scores else None,
        seed=BOOTSTRAP_SEED,
        samples=samples,
        preview_rows=preview_rows,
    )
    result["scores_from_training_seed"] = summaries[next(iter(raw))]["per_seed"][0]["seed"]
    return result


def run_benchmark(
    *,
    data_dir: Path = Path("data/raw/veremi"),
    output_dir: Path = Path("results"),
    train_seeds: int = 5,
    bootstrap_samples: int = 1000,
    bootstrap_preview_rows: int = 20,
    epochs: int = 5,
    run_generalization: bool = True,
    run_ablations: bool = True,
) -> dict[str, Any]:
    if train_seeds < 1:
        raise ValueError("train_seeds must be at least one")
    if bootstrap_samples < 1 or bootstrap_preview_rows < 0:
        raise ValueError("bootstrap_samples must be positive; preview rows non-negative")
    if bootstrap_preview_rows > bootstrap_samples:
        raise ValueError("bootstrap_preview_rows cannot exceed bootstrap_samples")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    import torch

    torch_threads = os.cpu_count() or 1
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    seeds = [SEED_BASE + offset for offset in range(train_seeds)]
    xgboost_available = _xgboost_available()
    models = [*UNSUPERVISED_MODELS] + [
        name for name in SUPERVISED_MODELS if name != "xgboost" or xgboost_available
    ]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    _log("Loading official VeReMi archives and extracting causal windows")
    table, record_counts = load_event_table(data_dir)
    labels = table.labels
    official = {split: table.rows(split=split) for split in SPLITS}
    reason = split_eligibility(labels, official)
    if reason:
        raise ValueError(f"Official splits are not evaluable: {reason}")

    # --- composite sender-cluster audit -------------------------------------
    archives = sorted(set(table.archive.tolist()))
    clusters_by_split = {}
    collisions_by_split = {}
    for split in SPLITS:
        rows = official[split]
        clusters_by_split[split] = len(
            set(zip(table.corridor[rows].tolist(), table.source[rows].tolist(), strict=True))
        )
        owners: dict[str, set[str]] = defaultdict(set)
        for archive, source in zip(table.archive[rows], table.source[rows], strict=True):
            owners[str(source)].add(str(archive))
        collisions_by_split[split] = sum(len(found) > 1 for found in owners.values())
    overlap_by_archive = {}
    for archive in archives:
        senders = {
            split: set(table.source[table.rows(split=split, archive=archive)].tolist())
            for split in SPLITS
        }
        overlap_by_archive[archive] = {
            "train_test_source_id_overlap": len(senders["train"] & senders["test"]),
            "train_validation_source_id_overlap": len(senders["train"] & senders["validation"]),
        }
    cluster_audit = {
        "cluster_identity": "(corridor_id, source_id); corridor_id = <archive>-<split>",
        "unique_composite_clusters_by_split": clusters_by_split,
        "cross_archive_source_id_collisions_by_split": collisions_by_split,
        "train_test_sender_overlap_by_archive": overlap_by_archive,
    }

    # --- main official-split comparison -----------------------------------
    main_summary, main_raw, main_representation = run_models(
        table, official, models, config=FULL_CONFIG, seeds=seeds, epochs=epochs, label="main"
    )
    test_rows = official["test"]
    test_events = [table.events[index] for index in test_rows]
    for name in models:
        localization = []
        for run, record in zip(main_raw[name], main_summary[name]["per_seed"], strict=True):
            localization.append(
                {
                    "attacker_emitter_recall_at_1": _emitter_recall_at_k(
                        test_events, run["test_scores"], k=1
                    ),
                    "attacker_emitter_recall_at_3": _emitter_recall_at_k(
                        test_events, run["test_scores"], k=3
                    ),
                    **{
                        key: value
                        for key, value in _simulated_mttd(
                            test_events, run["test_scores"], record["validation_f1_threshold"]
                        ).items()
                        if key != "time_basis"
                    },
                }
            )
        main_summary[name]["emitter_ranking_and_simulated_latency"] = {
            "per_seed": localization,
            "mean_std_across_training_seeds": summarize_seeds(localization),
            "note": (
                "retrospective sender ranking by maximum message score and simulator-time "
                "detection delay; neither is causal RCA nor field MTTD"
            ),
        }
        roc, pr = _curve_data(labels[test_rows], main_raw[name][0]["test_scores"])
        main_summary[name]["curves_training_seed"] = seeds[0]
        main_summary[name]["roc"] = roc
        main_summary[name]["precision_recall"] = pr

    _log("main: composite sender-cluster bootstrap")
    main_bootstrap = _context_bootstrap(
        table,
        official,
        main_raw,
        main_summary,
        samples=bootstrap_samples,
        preview_rows=bootstrap_preview_rows,
    )
    preview = main_bootstrap.pop("_preview_rows")

    # --- per-attack slices of the pooled official model ---------------------
    per_attack = {}
    for archive in archives:
        archive_validation = official["validation"][
            table.archive[official["validation"]] == archive
        ]
        archive_test_mask = table.archive[test_rows] == archive
        per_attack[archive] = {
            "test_records": int(archive_test_mask.sum()),
            "test_attackers": int(labels[test_rows][archive_test_mask].sum()),
            "models": {},
        }
        for name in models:
            per_seed = []
            for run in main_raw[name]:
                validation_mask = table.archive[official["validation"]] == archive
                per_seed.append(
                    evaluate_operating_points(
                        labels[archive_validation],
                        run["validation_scores"][validation_mask],
                        labels[test_rows][archive_test_mask],
                        run["test_scores"][archive_test_mask],
                    )
                )
            per_attack[archive]["models"][name] = {
                "supervision_regime": regime(name),
                "mean_std_across_training_seeds": summarize_seeds(per_seed),
            }
        per_attack[archive]["protocol"] = (
            "pooled official-split models; threshold and FAR budget re-selected on the "
            "same attack category's validation records; test restricted to that category"
        )

    generalization: dict[str, Any] = {}
    if run_generalization:
        loao = {}
        for source in archives:
            for target in archives:
                if source == target:
                    continue
                direction = f"{source}->{target}"
                split_rows = {
                    "train": table.rows(split="train", archive=source),
                    "validation": table.rows(split="validation", archive=source),
                    "test": table.rows(split="test", archive=target),
                }
                reason = split_eligibility(labels, split_rows)
                if reason is None and table.values.shape[-1] != len(table.feature_names):
                    reason = "source and target feature schemas differ"
                if reason is not None:
                    loao[direction] = {"status": "NOT_EVALUABLE", "reason": reason}
                    continue
                summary, raw, _ = run_models(
                    table,
                    split_rows,
                    models,
                    config=FULL_CONFIG,
                    seeds=seeds,
                    epochs=epochs,
                    label=f"loao {direction}",
                )
                _log(f"loao {direction}: bootstrap")
                bootstrap = _context_bootstrap(
                    table, split_rows, raw, summary, samples=bootstrap_samples
                )
                bootstrap.pop("_preview_rows")
                loao[direction] = {
                    "status": "EVALUABLE",
                    "protocol": (
                        f"fit and threshold on official {source} train/validation; "
                        f"test on official {target} test; normalization from {source} "
                        "benign training records"
                    ),
                    "feature_schema": list(FULL_CONFIG.feature_names),
                    "class_counts": _split_counts(labels, split_rows),
                    "models": summary,
                    "composite_cluster_bootstrap": bootstrap,
                }
        generalization["leave_one_attack_type_out"] = loao

        holdout_rows, holdout_meta = sender_disjoint_partition(table)
        reason = split_eligibility(labels, holdout_rows)
        if reason is not None:
            generalization["sender_disjoint_holdout"] = {
                "status": "NOT_EVALUABLE",
                "reason": reason,
                **holdout_meta,
            }
        else:
            summary, raw, _ = run_models(
                table,
                holdout_rows,
                models,
                config=FULL_CONFIG,
                seeds=seeds,
                epochs=epochs,
                label="sender-disjoint",
            )
            _log("sender-disjoint: bootstrap")
            bootstrap = _context_bootstrap(
                table, holdout_rows, raw, summary, samples=bootstrap_samples
            )
            bootstrap.pop("_preview_rows")
            generalization["sender_disjoint_holdout"] = {
                "status": "EVALUABLE",
                **holdout_meta,
                "class_counts": _split_counts(labels, holdout_rows),
                "models": summary,
                "composite_cluster_bootstrap": bootstrap,
            }

    ablation_summary: dict[str, Any] | None = None
    if run_ablations:
        supervised_candidates = [
            name for name in models if regime(name) == "supervised" and name != REFERENCE_MODEL
        ]
        top_baseline = max(
            supervised_candidates,
            key=lambda name: main_summary[name]["mean_std_across_training_seeds"][
                "validation_auroc"
            ]["mean"],
        )
        ablation_models = [REFERENCE_MODEL, top_baseline]
        ablations = {}
        for family, config in ABLATION_CONFIGS:
            summary, _, _ = run_models(
                table,
                official,
                ablation_models,
                config=config,
                seeds=seeds,
                epochs=epochs,
                label=f"ablation {config.name}",
            )
            ablations[family] = {
                "variant": config.name,
                "description": config.description,
                "feature_names": list(config.feature_names),
                "window_length": config.window,
                "validity_masks": config.use_masks,
                "models": {},
            }
            for name in ablation_models:
                variant_stats = summary[name]["mean_std_across_training_seeds"]
                full_stats = main_summary[name]["mean_std_across_training_seeds"]
                ablations[family]["models"][name] = {
                    "per_seed": summary[name]["per_seed"],
                    "variant_mean_std_across_training_seeds": variant_stats,
                    "full_mean_std_across_training_seeds": {
                        key: full_stats[key] for key in variant_stats if key in full_stats
                    },
                    "variant_minus_full_mean": {
                        key: variant_stats[key]["mean"] - full_stats[key]["mean"]
                        for key in variant_stats
                        if key in full_stats and key not in TIMING_KEYS
                    },
                }
        ablation_summary = {
            "protocol": (
                "official train/validation/test splits; each variant changes one factor "
                "relative to the full configuration; same training seeds; full-configuration "
                "numbers are the main benchmark runs"
            ),
            "full_configuration": {
                "name": FULL_CONFIG.name,
                "feature_names": list(FULL_CONFIG.feature_names),
                "window_length": FULL_CONFIG.window,
                "validity_masks": FULL_CONFIG.use_masks,
            },
            "models": ablation_models,
            "top_supervised_baseline": top_baseline,
            "top_baseline_selection": (
                "highest mean validation AUROC among supervised models other than the "
                "Tabular Transformer; the test split is not used for this choice"
            ),
            "causal_jitter_definition": CAUSAL_JITTER_DEFINITION,
            "local_density_definition": LOCAL_DENSITY_DEFINITION,
            "training_seeds": seeds,
            "ablations": ablations,
        }
        (output_dir / "ablation_summary.json").write_text(
            json.dumps(ablation_summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    manifest_path = Path(data_dir).parent / "manifest.json"
    summary = {
        "title": (
            "Evidence-Limited Misbehavior Detection for Connected-Vehicle Telemetry Using "
            "Causal Trajectory Features and Sender-Cluster Evaluation"
        ),
        "study_type": "VeReMi NextGen simulated misbehavior detection",
        "dataset": "VeReMi NextGen InTAS highway 2",
        "dataset_doi": "https://doi.org/10.5281/zenodo.19665762",
        "observation_unit": "de-duplicated BSM receive event with a causal trajectory window",
        "feature_names": list(FEATURE_NAMES),
        "causal_jitter_definition": CAUSAL_JITTER_DEFINITION,
        "local_density_definition": LOCAL_DENSITY_DEFINITION,
        "archive_unique_record_counts": record_counts,
        "official_splits": _split_counts(labels, official),
        "composite_sender_cluster_audit": cluster_audit,
        "preprocessing": {
            "normalization_fit": "non-malicious training records only (current step)",
            "preprocessing_sha256": main_representation.preprocessing_sha256(),
            "missing_values": "validity masks retained; missing values set to normalized zero",
            "split_policy": "official train/validation/test paths; no random test subsampling",
            "receiver_copy_policy": (
                "same sender/messageID/sendTime receiver copies collapsed by parser; "
                "earliest receive_time_ns retained"
            ),
            "model_inputs": {
                "robust_threshold_rules": "normalized current-step features",
                "isolation_forest": "normalized current-step features and validity masks",
                "one_class_svm": (
                    "normalized current-step features and validity masks; seeded sample of "
                    "at most 10,000 non-malicious training records"
                ),
                "lstm_autoencoder": "L = 10 sequence, masked reconstruction loss",
                "tabular_transformer": "L = 10 sequence, causal attention, validity masks",
                "flattened_window_models": (
                    "logistic regression, random forest, histogram gradient boosting, "
                    "XGBoost when installed, and the supervised MLP use the flattened "
                    "L = 10 window of normalized features and validity masks"
                ),
            },
        },
        "protocol": {
            "training_seeds": seeds,
            "seed_variability": "mean and sample standard deviation (ddof = 1) across seeds",
            "threshold_selection": "maximum F1 on the validation split, per model and seed",
            "far_budgets": list(FAR_BUDGETS),
            "far_operating_points": (
                "test_roc_tpr_at_far: highest test TPR with test FPR at or below the budget "
                "(descriptive, uses test scores only to read the curve); "
                "validation_calibrated_recall_at_far: threshold set so that validation "
                "non-malicious alert rate is at or below the budget, then applied to test, "
                "with the realized test FAR reported alongside"
            ),
            "neural_checkpoint_selection": "best validation AUROC, early stopping patience 2",
            "epochs": epochs,
            "torch_threads": torch_threads,
            "torch_deterministic_algorithms": True,
            "bootstrap_uses_training_seed": seeds[0],
            "bootstrap_preview_rows": bootstrap_preview_rows,
        },
        "model_groups": {
            "unsupervised_or_one_class": [name for name in models if name in UNSUPERVISED_MODELS],
            "supervised": [name for name in models if name in SUPERVISED_MODELS],
        },
        "models_not_run": (
            {}
            if xgboost_available
            else {"xgboost": "NOT_RUN: the xgboost package is not installed"}
        ),
        "architecture_only_not_evaluated": {
            "corridor_log_transformer": "no synchronized controller-log text corpus",
            "static_gcn": "graph models not trained in this benchmark",
            "async_st_gnn": "graph models not trained in this benchmark",
        },
        "models": main_summary,
        "uncertainty": {
            "training_seed_variability": "see models.<name>.mean_std_across_training_seeds",
            "sender_cluster_bootstrap": main_bootstrap,
            "bootstrap_preview_rows_file": "results/bootstrap_preview.csv",
            "note": (
                "Seed variability reflects model fitting randomness; bootstrap intervals "
                "reflect resampling of composite test sender clusters for one training "
                "seed. They answer different questions and are not combined."
            ),
        },
        "per_attack_category": per_attack,
        "generalization": generalization,
        "ablation_summary_file": "results/ablation_summary.json" if run_ablations else None,
        "scope_limits": {
            "dataset_attack_labels": "simulated VeReMi attacker/misbehavior labels",
            "attacker_emitter_localization": "sender-level risk ranking only",
            "simulated_mttd": "message-level simulator delay; not field-incident MTTD",
            "independent_scenario_holdout": (
                "NOT_EVALUABLE: both acquired archives use InTAS highway 2"
            ),
            "work_zone_far": None,
            "causal_root_cause_accuracy": None,
            "cross_modal_fusion": None,
            "production_latency_sla": None,
        },
        "context_dataset_manifest": (
            json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_seed_csv(output_dir / "seed_metrics.csv", summary)
    with (output_dir / "bootstrap_preview.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "replicate",
            "resampling_unit",
            "model",
            "f1_cyber_vs_benign",
            "auroc",
            "false_alarm_rate_non_malicious",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(preview)
    from corridor_aiops.evaluation.tables import write_tables

    write_tables(summary, ablation_summary, output_dir)
    _log(f"benchmark finished in {summary['runtime_seconds']:.0f} s")
    return summary


def _write_seed_csv(path: Path, summary: dict[str, Any]) -> None:
    fields = [
        "context",
        "model",
        "supervision_regime",
        "seed",
        "f1_cyber_vs_benign",
        "auroc",
        "average_precision",
        "false_alarm_rate_non_malicious",
        "test_roc_tpr_at_far_1.0pct",
        "test_roc_tpr_at_far_5.0pct",
        "validation_calibrated_recall_at_far_1.0pct",
        "validation_calibrated_recall_at_far_5.0pct",
        "checkpoint_sha256",
    ]
    contexts = [("official_split", summary["models"])]
    for direction, block in summary["generalization"].get("leave_one_attack_type_out", {}).items():
        if block.get("status") == "EVALUABLE":
            contexts.append((f"loao:{direction}", block["models"]))
    holdout = summary["generalization"].get("sender_disjoint_holdout", {})
    if holdout.get("status") == "EVALUABLE":
        contexts.append(("sender_disjoint", holdout["models"]))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for context, models in contexts:
            for name, block in models.items():
                for row in block["per_seed"]:
                    writer.writerow(
                        {
                            "context": context,
                            "model": name,
                            "supervision_regime": block["supervision_regime"],
                            **row,
                        }
                    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate de-duplicated VeReMi events on official data splits."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/veremi"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--train-seeds", type=int, default=5, help="independent model-training seeds"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument(
        "--bootstrap-preview-rows",
        type=int,
        default=20,
        help="number of bootstrap replicates written to bootstrap_preview.csv",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--skip-generalization", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    args = parser.parse_args()
    result = run_benchmark(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_seeds=args.train_seeds,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_preview_rows=args.bootstrap_preview_rows,
        epochs=args.epochs,
        run_generalization=not args.skip_generalization,
        run_ablations=not args.skip_ablations,
    )
    print(
        json.dumps(
            {
                name: {
                    metric: block["mean_std_across_training_seeds"][metric]
                    for metric in ("f1_cyber_vs_benign", "auroc")
                }
                for name, block in result["models"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
