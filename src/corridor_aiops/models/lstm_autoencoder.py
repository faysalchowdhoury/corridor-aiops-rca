from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as error:  # pragma: no cover - optional extra
    raise ImportError(
        'Install the optional PyTorch dependency with `python -m pip install -e ".[torch]"`.'
    ) from error


class LSTMAutoencoder(nn.Module):
    """Sequence reconstruction baseline; fit only on verified normal telemetry.

    When a validity mask is supplied, padded time steps and unobserved features are
    zeroed before encoding, so their raw contents cannot influence the
    reconstruction. :func:`masked_reconstruction_error` excludes them from the score.
    """

    def __init__(self, feature_count: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.encoder = nn.LSTM(feature_count, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, feature_count)

    def forward(
        self,
        sequence: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if valid is not None:
            if valid.shape != sequence.shape:
                raise ValueError("valid mask must have the same shape as sequence")
            sequence = sequence * valid.to(sequence.dtype)
        _, (hidden, cell) = self.encoder(sequence)
        repeated = hidden[-1].unsqueeze(1).expand(-1, sequence.size(1), -1)
        decoded, _ = self.decoder(repeated, (hidden, cell))
        return self.output(decoded)


def masked_reconstruction_error(
    sequence: torch.Tensor,
    reconstruction: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Per-sequence mean squared error over observed features of valid steps only."""
    if valid.shape != sequence.shape or reconstruction.shape != sequence.shape:
        raise ValueError("sequence, reconstruction, and valid must share one shape")
    weights = valid.to(sequence.dtype)
    errors = (reconstruction - sequence * weights).square() * weights
    return errors.sum(dim=(1, 2)) / weights.sum(dim=(1, 2)).clamp(min=1.0)
