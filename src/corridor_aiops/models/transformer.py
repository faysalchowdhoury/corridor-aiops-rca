from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
except ImportError as error:  # pragma: no cover - optional extra
    raise ImportError("Install the optional ML dependencies with `pip install .[ml]`.") from error


class CorridorLogTransformer(nn.Module):
    """Small multimodal event encoder for telemetry vectors and log token IDs."""

    def __init__(
        self,
        numeric_features: int,
        vocabulary_size: int,
        *,
        hidden_size: int = 64,
        heads: int = 4,
        layers: int = 2,
        classes: int = 2,
    ) -> None:
        super().__init__()
        self.numeric_projection = nn.Linear(numeric_features, hidden_size)
        self.text_embedding = nn.Embedding(vocabulary_size, hidden_size, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.classifier = nn.Linear(hidden_size, classes)
        self.mask_head = nn.Linear(hidden_size, vocabulary_size)

    def forward(
        self,
        telemetry: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        numeric = self.numeric_projection(telemetry)
        text = self.text_embedding(token_ids)
        encoded = self.sequence_encoder(numeric + text)
        return {
            "logits": self.classifier(encoded.mean(dim=1)),
            "masked_token_logits": self.mask_head(encoded),
            "hidden_states": encoded,
        }
