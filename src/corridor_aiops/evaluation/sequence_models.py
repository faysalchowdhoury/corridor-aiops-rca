from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

BATCH_SIZE = 128
EARLY_STOPPING_PATIENCE = 2
SELECTION_METRICS = ("validation_auroc", "benign_reconstruction_loss")


@dataclass(frozen=True)
class SequenceSplit:
    values: np.ndarray
    feature_mask: np.ndarray
    step_mask: np.ndarray
    labels: np.ndarray


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional extra
        raise RuntimeError(
            "Sequence baselines require PyTorch; install it with "
            '`python -m pip install -e ".[torch]"`.'
        ) from error
    return torch


def state_dict_sha256(state: dict[str, Any]) -> str:
    """Hash a model checkpoint by parameter name, dtype, shape, and raw bytes."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _batches(size: int, batch_size: int, rng: np.random.Generator):
    order = rng.permutation(size)
    for start in range(0, size, batch_size):
        yield order[start : start + batch_size]


def _sequence_tensors(data: SequenceSplit, indexes: np.ndarray, torch, device):
    return (
        torch.as_tensor(data.values[indexes], dtype=torch.float32, device=device),
        torch.as_tensor(data.feature_mask[indexes], dtype=torch.bool, device=device),
        torch.as_tensor(data.step_mask[indexes], dtype=torch.bool, device=device),
    )


def _predict(
    score_batch: Callable[[np.ndarray], Any],
    size: int,
    torch,
) -> tuple[np.ndarray, float]:
    scores: list[float] = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, size, BATCH_SIZE):
            indexes = np.arange(start, min(start + BATCH_SIZE, size))
            scores.extend(score_batch(indexes).cpu().numpy().tolist())
    elapsed_ms = (time.perf_counter() - started) * 1000.0 / max(size, 1)
    return np.asarray(scores, dtype=float), elapsed_ms


def _train_with_validation_checkpoint(
    *,
    model,
    epochs: int,
    seed: int,
    train_size: int,
    train_step: Callable[[np.ndarray], Any],
    validation_scores: Callable[[], np.ndarray],
    validation_labels: np.ndarray,
    optimizer,
    selection_metric: str = "validation_auroc",
) -> dict[str, Any]:
    """Train with AdamW; keep the best epoch on the selection metric (patience 2).

    ``validation_auroc`` maximizes attacker-versus-benign AUROC on the validation split
    and therefore uses attacker labels. ``benign_reconstruction_loss`` minimizes the mean
    score (masked reconstruction error) over validation records with label 0 only, so no
    attacker record and no attacker-versus-benign metric enters checkpoint selection.
    """
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(f"Unknown selection_metric: {selection_metric}")
    benign_validation = np.asarray(validation_labels) == 0
    if selection_metric == "benign_reconstruction_loss" and not benign_validation.any():
        raise ValueError("Label-free selection requires non-malicious validation records")
    rng = np.random.default_rng(seed)
    best_auc = -np.inf
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    epochs_run = 0
    for epoch in range(epochs):
        epochs_run = epoch + 1
        model.train()
        for indexes in _batches(train_size, BATCH_SIZE, rng):
            loss = train_step(indexes)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        if selection_metric == "validation_auroc":
            auc = float(roc_auc_score(validation_labels, validation_scores()))
        else:
            # higher is better: negative mean benign reconstruction error
            auc = -float(np.mean(validation_scores()[benign_validation]))
        if auc > best_auc:
            best_auc = auc
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= EARLY_STOPPING_PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("Model did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    selected = (
        {"selected_validation_auroc": best_auc}
        if selection_metric == "validation_auroc"
        else {
            "selection_metric": selection_metric,
            "selected_validation_benign_reconstruction_loss": -best_auc,
        }
    )
    return {
        "epochs_run": epochs_run,
        "selected_epoch": best_epoch,
        **selected,
        "checkpoint_sha256": state_dict_sha256(best_state),
    }


def fit_lstm_autoencoder(
    train: SequenceSplit,
    validation: SequenceSplit,
    test: SequenceSplit,
    *,
    epochs: int,
    seed: int,
    selection_metric: str = "validation_auroc",
) -> dict[str, Any]:
    """Train on non-malicious training sequences only.

    ``selection_metric="validation_auroc"`` (the main benchmark) selects the checkpoint
    with attacker labels on the validation split; ``"benign_reconstruction_loss"`` is
    the label-free control that selects on non-malicious validation records only.
    """
    torch = _torch()
    from corridor_aiops.models.lstm_autoencoder import (
        LSTMAutoencoder,
        masked_reconstruction_error,
    )

    device = torch.device("cpu")
    benign = np.flatnonzero(train.labels == 0)
    if not len(benign):
        raise ValueError("LSTM autoencoder training requires non-malicious training records")
    started = time.perf_counter()
    torch.manual_seed(seed)
    model = LSTMAutoencoder(train.values.shape[-1], hidden_size=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def score(data: SequenceSplit) -> Callable[[np.ndarray], Any]:
        def batch(indexes: np.ndarray):
            values, feature_mask, step_mask = _sequence_tensors(data, indexes, torch, device)
            valid = feature_mask & step_mask.unsqueeze(-1)
            return masked_reconstruction_error(values, model(values, valid), valid)

        return batch

    def train_step(indexes: np.ndarray):
        values, feature_mask, step_mask = _sequence_tensors(train, benign[indexes], torch, device)
        valid = feature_mask & step_mask.unsqueeze(-1)
        return masked_reconstruction_error(values, model(values, valid), valid).mean()

    selection = _train_with_validation_checkpoint(
        model=model,
        epochs=epochs,
        seed=seed,
        train_size=len(benign),
        train_step=train_step,
        validation_scores=lambda: _predict(score(validation), len(validation.labels), torch)[0],
        validation_labels=validation.labels,
        optimizer=optimizer,
        selection_metric=selection_metric,
    )
    training_seconds = time.perf_counter() - started
    validation_scores, _ = _predict(score(validation), len(validation.labels), torch)
    test_scores, inference_ms = _predict(score(test), len(test.labels), torch)
    return {
        "validation_scores": validation_scores,
        "test_scores": test_scores,
        "inference_ms_per_event": inference_ms,
        "training_seconds": training_seconds,
        "training_status": (
            "benign_only_training_with_validation_auroc_selection"
            if selection_metric == "validation_auroc"
            else "benign_only_training_with_label_free_benign_reconstruction_selection"
        ),
        "training_records": int(len(benign)),
        **selection,
    }


def fit_tabular_transformer(
    train: SequenceSplit,
    validation: SequenceSplit,
    test: SequenceSplit,
    *,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    torch = _torch()
    from corridor_aiops.models.transformer import TabularTransformer

    device = torch.device("cpu")
    positives = int(np.sum(train.labels == 1))
    negatives = int(np.sum(train.labels == 0))
    if not positives or not negatives:
        raise ValueError("Tabular Transformer training requires both training classes")
    started = time.perf_counter()
    torch.manual_seed(seed)
    model = TabularTransformer(
        train.values.shape[-1],
        hidden_size=32,
        heads=4,
        layers=1,
        max_steps=train.values.shape[1],
        dropout=0.1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, dtype=torch.float32, device=device)
    )

    def score(data: SequenceSplit) -> Callable[[np.ndarray], Any]:
        def batch(indexes: np.ndarray):
            values, feature_mask, step_mask = _sequence_tensors(data, indexes, torch, device)
            return torch.sigmoid(model(values, feature_mask, step_mask)["detection_logits"])

        return batch

    def train_step(indexes: np.ndarray):
        values, feature_mask, step_mask = _sequence_tensors(train, indexes, torch, device)
        labels = torch.as_tensor(train.labels[indexes], dtype=torch.float32, device=device)
        return loss_function(model(values, feature_mask, step_mask)["detection_logits"], labels)

    selection = _train_with_validation_checkpoint(
        model=model,
        epochs=epochs,
        seed=seed,
        train_size=len(train.labels),
        train_step=train_step,
        validation_scores=lambda: _predict(score(validation), len(validation.labels), torch)[0],
        validation_labels=validation.labels,
        optimizer=optimizer,
    )
    training_seconds = time.perf_counter() - started
    validation_scores, _ = _predict(score(validation), len(validation.labels), torch)
    test_scores, inference_ms = _predict(score(test), len(test.labels), torch)
    return {
        "validation_scores": validation_scores,
        "test_scores": test_scores,
        "inference_ms_per_event": inference_ms,
        "training_seconds": training_seconds,
        "training_status": "supervised_on_training_labels",
        "training_records": int(len(train.labels)),
        "training_positive_weight": negatives / positives,
        **selection,
    }


def fit_supervised_mlp(
    train: tuple[np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    *,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    """Two-hidden-layer MLP on flattened causal windows, same protocol as the Transformer."""
    torch = _torch()
    from torch import nn

    device = torch.device("cpu")
    train_x, train_y = train
    positives = int(np.sum(train_y == 1))
    negatives = int(np.sum(train_y == 0))
    if not positives or not negatives:
        raise ValueError("Supervised MLP training requires both training classes")
    started = time.perf_counter()
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(train_x.shape[1], 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, dtype=torch.float32, device=device)
    )

    def score(matrix: np.ndarray) -> Callable[[np.ndarray], Any]:
        def batch(indexes: np.ndarray):
            inputs = torch.as_tensor(matrix[indexes], dtype=torch.float32, device=device)
            return torch.sigmoid(model(inputs).squeeze(-1))

        return batch

    def train_step(indexes: np.ndarray):
        inputs = torch.as_tensor(train_x[indexes], dtype=torch.float32, device=device)
        labels = torch.as_tensor(train_y[indexes], dtype=torch.float32, device=device)
        return loss_function(model(inputs).squeeze(-1), labels)

    selection = _train_with_validation_checkpoint(
        model=model,
        epochs=epochs,
        seed=seed,
        train_size=len(train_y),
        train_step=train_step,
        validation_scores=lambda: _predict(score(validation[0]), len(validation[1]), torch)[0],
        validation_labels=validation[1],
        optimizer=optimizer,
    )
    training_seconds = time.perf_counter() - started
    validation_scores, _ = _predict(score(validation[0]), len(validation[1]), torch)
    test_scores, inference_ms = _predict(score(test[0]), len(test[1]), torch)
    return {
        "validation_scores": validation_scores,
        "test_scores": test_scores,
        "inference_ms_per_event": inference_ms,
        "training_seconds": training_seconds,
        "training_status": "supervised_on_training_labels",
        "training_records": int(len(train_y)),
        "training_positive_weight": negatives / positives,
        **selection,
    }


def fit_sequence_baselines(
    train: SequenceSplit,
    validation: SequenceSplit,
    test: SequenceSplit,
    *,
    epochs: int = 5,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    return {
        "lstm_autoencoder": fit_lstm_autoencoder(train, validation, test, epochs=epochs, seed=seed),
        "tabular_transformer": fit_tabular_transformer(
            train, validation, test, epochs=epochs, seed=seed
        ),
    }
