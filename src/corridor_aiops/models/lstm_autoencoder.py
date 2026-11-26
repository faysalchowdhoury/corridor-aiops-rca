from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as error:  # pragma: no cover - optional extra
    raise ImportError("Install the optional ML dependencies with `pip install .[ml]`.") from error


class LSTMAutoencoder(nn.Module):
    """Sequence reconstruction baseline; fit only on verified normal telemetry."""

    def __init__(self, feature_count: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.encoder = nn.LSTM(feature_count, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, feature_count)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, (hidden, cell) = self.encoder(sequence)
        repeated = hidden[-1].unsqueeze(1).expand(-1, sequence.size(1), -1)
        decoded, _ = self.decoder(repeated, (hidden, cell))
        return self.output(decoded)
