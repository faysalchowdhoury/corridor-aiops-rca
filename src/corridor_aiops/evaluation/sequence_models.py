from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class SequenceSplit:
    values: np.ndarray
    feature_mask: np.ndarray
    step_mask: np.ndarray
    labels: np.ndarray


def _batches(size: int, batch_size: int, rng: np.random.Generator):
    order = rng.permutation(size)
    for start in range(0, size, batch_size):
        yield order[start : start + batch_size]


def _tensor_batch(data: SequenceSplit, indexes: np.ndarray, torch, device):
    return (
        torch.as_tensor(data.values[indexes], dtype=torch.float32, device=device),
        torch.as_tensor(data.feature_mask[indexes], dtype=torch.bool, device=device),
        torch.as_tensor(data.step_mask[indexes], dtype=torch.bool, device=device),
    )


def _predict_autoencoder(model, data: SequenceSplit, torch, device, batch_size: int):
    model.eval()
    scores = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(data.labels), batch_size):
            indexes = np.arange(start, min(start + batch_size, len(data.labels)))
            values, feature_mask, step_mask = _tensor_batch(data, indexes, torch, device)
            valid = feature_mask & step_mask.unsqueeze(-1)
            reconstruction = model(values)
            errors = (reconstruction - values).square() * valid.to(values.dtype)
            scores.extend(
                (errors.sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp(min=1))
                .cpu()
                .numpy()
                .tolist()
            )
    elapsed_ms = (time.perf_counter() - started) * 1000.0 / max(len(data.labels), 1)
    return np.asarray(scores, dtype=float), elapsed_ms


def _predict_transformer(model, data: SequenceSplit, torch, device, batch_size: int):
    model.eval()
    scores = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(data.labels), batch_size):
            indexes = np.arange(start, min(start + batch_size, len(data.labels)))
            values, feature_mask, step_mask = _tensor_batch(data, indexes, torch, device)
            logits = model(values, feature_mask, step_mask)["detection_logits"]
            scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    elapsed_ms = (time.perf_counter() - started) * 1000.0 / max(len(data.labels), 1)
    return np.asarray(scores, dtype=float), elapsed_ms


def _fit_autoencoder(
    train: SequenceSplit,
    validation: SequenceSplit,
    test: SequenceSplit,
    *,
    epochs: int,
    seed: int,
    torch,
    device,
) -> dict[str, Any]:
    from corridor_aiops.models.lstm_autoencoder import LSTMAutoencoder

    training_started = time.perf_counter()
    benign_indexes = np.flatnonzero(train.labels == 0)
    if not len(benign_indexes):
        raise ValueError("LSTM autoencoder training requires non-malicious training records")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = LSTMAutoencoder(train.values.shape[-1], hidden_size=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    best_auc = -np.inf
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    batch_size = 128

    for epoch in range(epochs):
        model.train()
        for indexes in _batches(len(benign_indexes), batch_size, rng):
            selected = benign_indexes[indexes]
            values, feature_mask, step_mask = _tensor_batch(train, selected, torch, device)
            valid = feature_mask & step_mask.unsqueeze(-1)
            reconstruction = model(values)
            errors = (reconstruction - values).square() * valid.to(values.dtype)
            loss = errors.sum() / valid.sum().clamp(min=1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        validation_scores, _ = _predict_autoencoder(
            model, validation, torch, device, batch_size
        )
        validation_auc = float(roc_auc_score(validation.labels, validation_scores))
        if validation_auc > best_auc:
            best_auc = validation_auc
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 2:
                break

    if best_state is None:
        raise RuntimeError("LSTM autoencoder did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    training_seconds = time.perf_counter() - training_started
    validation_scores, _ = _predict_autoencoder(model, validation, torch, device, batch_size)
    test_scores, inference_ms = _predict_autoencoder(model, test, torch, device, batch_size)
    return {
        "validation_scores": validation_scores,
        "test_scores": test_scores,
        "inference_ms_per_event": inference_ms,
        "training_seconds": training_seconds,
        "training_status": "trained_on_non_malicious_official_training_sequences",
        "training_records": len(benign_indexes),
        "epochs_run": epoch + 1,
        "selected_epoch": best_epoch,
        "selected_validation_auroc": best_auc,
    }


def _fit_tabular_transformer(
    train: SequenceSplit,
    validation: SequenceSplit,
    test: SequenceSplit,
    *,
    epochs: int,
    seed: int,
    torch,
    device,
) -> dict[str, Any]:
    from corridor_aiops.models.transformer import TabularTransformer

    training_started = time.perf_counter()
    positives = int(np.sum(train.labels == 1))
    negatives = int(np.sum(train.labels == 0))
    if not positives or not negatives:
        raise ValueError("Tabular Transformer training requires both training classes")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TabularTransformer(
        train.values.shape[-1],
        hidden_size=32,
        heads=4,
        layers=1,
        max_steps=train.values.shape[1],
        dropout=0.1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    positive_weight = torch.tensor(negatives / positives, dtype=torch.float32, device=device)
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    best_auc = -np.inf
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    batch_size = 128

    for epoch in range(epochs):
        model.train()
        for indexes in _batches(len(train.labels), batch_size, rng):
            values, feature_mask, step_mask = _tensor_batch(train, indexes, torch, device)
            labels = torch.as_tensor(train.labels[indexes], dtype=torch.float32, device=device)
            logits = model(values, feature_mask, step_mask)["detection_logits"]
            loss = loss_function(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        validation_scores, _ = _predict_transformer(
            model, validation, torch, device, batch_size
        )
        validation_auc = float(roc_auc_score(validation.labels, validation_scores))
        if validation_auc > best_auc:
            best_auc = validation_auc
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 2:
                break

    if best_state is None:
        raise RuntimeError("Tabular Transformer did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    training_seconds = time.perf_counter() - training_started
    validation_scores, _ = _predict_transformer(model, validation, torch, device, batch_size)
    test_scores, inference_ms = _predict_transformer(model, test, torch, device, batch_size)
    return {
        "validation_scores": validation_scores,
        "test_scores": test_scores,
        "inference_ms_per_event": inference_ms,
        "training_seconds": training_seconds,
        "training_status": "supervised_on_official_training_labels",
        "training_records": len(train.labels),
        "epochs_run": epoch + 1,
        "selected_epoch": best_epoch,
        "selected_validation_auroc": best_auc,
        "training_positive_weight": negatives / positives,
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
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional extra
        raise RuntimeError(
            "Sequence baselines require PyTorch; install the optional dependencies with "
            "`pip install .[torch]`."
        ) from error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "lstm_autoencoder": _fit_autoencoder(
            train,
            validation,
            test,
            epochs=epochs,
            seed=seed,
            torch=torch,
            device=device,
        ),
        "tabular_transformer": _fit_tabular_transformer(
            train,
            validation,
            test,
            epochs=epochs,
            seed=seed,
            torch=torch,
            device=device,
        ),
    }
