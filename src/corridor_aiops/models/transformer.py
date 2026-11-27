from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
except ImportError as error:  # pragma: no cover - optional extra
    raise ImportError(
        'Install the optional PyTorch dependency with `python -m pip install -e ".[torch]"`.'
    ) from error


class CorridorLogTransformer(nn.Module):
    """Small multimodal event encoder for telemetry vectors and log token IDs.

    ARCHITECTURE-ONLY: no synchronized controller-log text corpus is available in the
    acquired evidence, so this module is not trained or benchmarked in this repository.
    """

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
        self.sequence_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(hidden_size, classes)
        self.mask_head = nn.Linear(hidden_size, vocabulary_size)

    def forward(
        self,
        telemetry: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        causal: bool = True,
    ) -> dict[str, Any]:
        """Encode a padded sequence.

        ``attention_mask`` is True for valid positions; when omitted, positions whose
        token ID is the padding index 0 are treated as padding. Padding is passed to
        the encoder as ``src_key_padding_mask``; ``causal`` adds an upper-triangular
        mask so each position attends only to itself and earlier positions. Pooling
        averages valid positions only.
        """
        if telemetry.ndim != 3 or token_ids.ndim != 2:
            raise ValueError("telemetry must be [batch, steps, features]; tokens [batch, steps]")
        if token_ids.shape != telemetry.shape[:2]:
            raise ValueError("token_ids must align with telemetry steps")
        valid = token_ids.ne(0) if attention_mask is None else attention_mask.bool()
        if valid.shape != token_ids.shape:
            raise ValueError("attention_mask must have shape [batch, steps]")
        step_count = telemetry.size(1)
        safe_valid = valid.clone()
        safe_valid[~valid.any(dim=-1), 0] = True
        causal_mask = (
            torch.triu(
                torch.ones(step_count, step_count, dtype=torch.bool, device=telemetry.device),
                diagonal=1,
            )
            if causal
            else None
        )
        numeric = self.numeric_projection(telemetry)
        text = self.text_embedding(token_ids)
        encoded = self.sequence_encoder(
            numeric + text,
            mask=causal_mask,
            src_key_padding_mask=~safe_valid,
        )
        weights = valid.unsqueeze(-1).to(encoded.dtype)
        encoded = encoded * weights
        pooled = encoded.sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return {
            "logits": self.classifier(pooled),
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
