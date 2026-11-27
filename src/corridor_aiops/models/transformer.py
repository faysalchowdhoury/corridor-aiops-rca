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


class TabularTransformer(nn.Module):
    """Causal sequence classifier for structured telemetry without text inputs."""

    def __init__(
        self,
        feature_count: int,
        *,
        hidden_size: int = 64,
        heads: int = 4,
        layers: int = 2,
        max_steps: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if feature_count < 1:
            raise ValueError("feature_count must be positive")
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.input_projection = nn.Linear(feature_count * 2, hidden_size)
        self.position_embedding = nn.Embedding(max_steps, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(
        self,
        telemetry: torch.Tensor,
        feature_mask: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if telemetry.ndim != 3:
            raise ValueError("telemetry must have shape [batch, steps, features]")
        if feature_mask.shape != telemetry.shape:
            raise ValueError("feature_mask must have the same shape as telemetry")
        if step_mask.shape != telemetry.shape[:2]:
            raise ValueError("step_mask must have shape [batch, steps]")
        batch_size, step_count, _ = telemetry.shape
        if step_count < 1 or step_count > self.position_embedding.num_embeddings:
            raise ValueError("telemetry step count is outside the configured position range")

        observed_features = feature_mask.to(telemetry.dtype)
        values = telemetry * observed_features
        positions = torch.arange(step_count, device=telemetry.device)
        encoded = self.input_projection(
            torch.cat((values, observed_features), dim=-1)
        ) + self.position_embedding(positions).unsqueeze(0)
        valid_steps = step_mask.bool()
        safe_steps = valid_steps.clone()
        safe_steps[~valid_steps.any(dim=-1), 0] = True
        causal_mask = torch.triu(
            torch.ones(step_count, step_count, dtype=torch.bool, device=telemetry.device),
            diagonal=1,
        )
        states = self.encoder(
            encoded,
            mask=causal_mask,
            src_key_padding_mask=~safe_steps,
        )
        last_valid = torch.where(
            valid_steps, positions.unsqueeze(0), -1
        ).amax(dim=-1).clamp(min=0)
        pooled = states[torch.arange(batch_size, device=telemetry.device), last_valid]
        pooled = pooled * valid_steps.any(dim=-1, keepdim=True).to(pooled.dtype)
        return {
            "detection_logits": self.classifier(pooled).squeeze(-1),
            "hidden_states": states,
        }
