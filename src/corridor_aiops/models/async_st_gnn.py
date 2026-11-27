from __future__ import annotations

import math

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as error:  # pragma: no cover - optional extra
    raise ImportError("Install the optional ML dependencies with `pip install .[ml]`.") from error


class _GATv2Layer(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.edge_projection = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.attention = nn.Linear(hidden_size, 1, bias=False)
        self.message_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.normalization = nn.LayerNorm(hidden_size)

    def forward(
        self,
        node_states: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, node_count, hidden_size = node_states.shape
        valid_nodes = node_mask.bool()
        edges = adjacency.bool() & valid_nodes.unsqueeze(1) & valid_nodes.unsqueeze(2)
        identity = torch.eye(node_count, dtype=torch.bool, device=node_states.device)
        edges = edges | (identity.unsqueeze(0) & valid_nodes.unsqueeze(1))

        targets = node_states.unsqueeze(2).expand(-1, -1, node_count, -1)
        sources = node_states.unsqueeze(1).expand(-1, node_count, -1, -1)
        pair_features = torch.cat((targets, sources), dim=-1)
        scores = self.attention(
            F.leaky_relu(self.edge_projection(pair_features), negative_slope=0.2)
        ).squeeze(-1)
        scores = scores.masked_fill(~edges, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        messages = torch.bmm(weights, self.message_projection(node_states))
        updated = self.normalization(node_states + self.dropout(messages))
        return updated * valid_nodes.unsqueeze(-1).to(updated.dtype)


class _GraphConvolutionLayer(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.message_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.normalization = nn.LayerNorm(hidden_size)

    def forward(
        self,
        node_states: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_nodes = node_mask.bool()
        edges = adjacency.bool() & valid_nodes.unsqueeze(1) & valid_nodes.unsqueeze(2)
        node_count = node_states.size(1)
        identity = torch.eye(node_count, dtype=torch.bool, device=node_states.device)
        edges = edges | (identity.unsqueeze(0) & valid_nodes.unsqueeze(1))
        adjacency_values = edges.to(node_states.dtype)
        degree = adjacency_values.sum(dim=-1).clamp(min=1.0)
        inverse_sqrt_degree = degree.rsqrt()
        normalized_adjacency = (
            inverse_sqrt_degree.unsqueeze(2)
            * adjacency_values
            * inverse_sqrt_degree.unsqueeze(1)
        )
        messages = torch.bmm(
            normalized_adjacency,
            self.message_projection(node_states),
        )
        updated = self.normalization(node_states + self.dropout(messages))
        return updated * valid_nodes.unsqueeze(-1).to(updated.dtype)


class AsyncSTGNN(nn.Module):
    """Causal asynchronous BSM encoder with dynamic V2X graph attention heads."""

    def __init__(
        self,
        feature_count: int,
        *,
        hidden_size: int = 64,
        heads: int = 4,
        temporal_layers: int = 2,
        dropout: float = 0.1,
        time_frequencies: int = 4,
    ) -> None:
        super().__init__()
        if feature_count < 1:
            raise ValueError("feature_count must be positive")
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")
        if time_frequencies < 1:
            raise ValueError("time_frequencies must be positive")

        self.feature_count = feature_count
        self.time_frequencies = time_frequencies
        frequencies = torch.exp(
            torch.linspace(0.0, math.log(1000.0), steps=time_frequencies)
        )
        self.register_buffer("time_frequency", frequencies, persistent=True)
        self.telemetry_projection = nn.Linear(feature_count * 2, hidden_size)
        self.time_projection = nn.Linear(1 + time_frequencies * 2, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=temporal_layers,
            enable_nested_tensor=False,
        )
        self.graph_encoder = _GATv2Layer(hidden_size, dropout)
        self.detection_head = nn.Linear(hidden_size, 1)
        self.localization_head = nn.Linear(hidden_size, 1)

    def _time_encoding(self, delta_t: torch.Tensor) -> torch.Tensor:
        elapsed = torch.log1p(delta_t.clamp(min=0.0)).unsqueeze(-1)
        phase = elapsed * self.time_frequency
        return torch.cat((elapsed, torch.sin(phase), torch.cos(phase)), dim=-1)

    def forward(
        self,
        telemetry: torch.Tensor,
        feature_mask: torch.Tensor,
        delta_t: torch.Tensor,
        step_mask: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if telemetry.ndim != 4:
            raise ValueError("telemetry must have shape [batch, nodes, steps, features]")
        if telemetry.shape[-1] != self.feature_count:
            raise ValueError("telemetry feature dimension does not match feature_count")
        batch_size, node_count, step_count, _ = telemetry.shape
        if step_count < 1:
            raise ValueError("telemetry must contain at least one time step")
        if feature_mask.shape != telemetry.shape:
            raise ValueError("feature_mask must have the same shape as telemetry")
        if delta_t.shape != (batch_size, node_count, step_count):
            raise ValueError("delta_t must have shape [batch, nodes, steps]")
        if step_mask.shape != delta_t.shape:
            raise ValueError("step_mask must have shape [batch, nodes, steps]")
        if adjacency.shape != (batch_size, node_count, node_count):
            raise ValueError("adjacency must have shape [batch, nodes, nodes]")
        if node_mask.shape != (batch_size, node_count):
            raise ValueError("node_mask must have shape [batch, nodes]")

        observed_features = feature_mask.to(telemetry.dtype)
        inputs = torch.cat((telemetry * observed_features, observed_features), dim=-1)
        tokens = self.telemetry_projection(inputs) + self.time_projection(
            self._time_encoding(delta_t)
        )
        flattened = tokens.reshape(batch_size * node_count, step_count, -1)
        valid_steps = step_mask.bool().reshape(batch_size * node_count, step_count)
        safe_steps = valid_steps.clone()
        safe_steps[~valid_steps.any(dim=-1), 0] = True
        causal_mask = torch.triu(
            torch.ones(step_count, step_count, dtype=torch.bool, device=telemetry.device),
            diagonal=1,
        )
        temporal_states = self.temporal_encoder(
            flattened,
            mask=causal_mask,
            src_key_padding_mask=~safe_steps,
        ).reshape(batch_size, node_count, step_count, -1)

        positions = torch.arange(step_count, device=telemetry.device)
        last_valid = torch.where(
            step_mask.bool(), positions.view(1, 1, -1), -1
        ).amax(dim=-1).clamp(min=0)
        gather_index = last_valid[..., None, None].expand(-1, -1, 1, temporal_states.size(-1))
        node_states = temporal_states.gather(2, gather_index).squeeze(2)
        effective_node_mask = node_mask.bool() & step_mask.bool().any(dim=-1)
        node_states = node_states * effective_node_mask.unsqueeze(-1).to(node_states.dtype)
        graph_states = self.graph_encoder(node_states, adjacency, effective_node_mask)
        return {
            "detection_logits": self.detection_head(graph_states).squeeze(-1),
            "localization_logits": self.localization_head(graph_states).squeeze(-1),
            "hidden_states": graph_states,
            "temporal_states": temporal_states,
        }


class StaticGCN(nn.Module):
    """Topology-aware graph baseline that discards event-time ordering."""

    def __init__(
        self,
        feature_count: int,
        *,
        hidden_size: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if feature_count < 1:
            raise ValueError("feature_count must be positive")
        self.feature_count = feature_count
        self.node_projection = nn.Linear(feature_count * 2, hidden_size)
        self.graph_encoder = _GraphConvolutionLayer(hidden_size, dropout)
        self.detection_head = nn.Linear(hidden_size, 1)
        self.localization_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        telemetry: torch.Tensor,
        feature_mask: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if telemetry.ndim != 4:
            raise ValueError("telemetry must have shape [batch, nodes, steps, features]")
        if telemetry.shape[-1] != self.feature_count:
            raise ValueError("telemetry feature dimension does not match feature_count")
        if feature_mask.shape != telemetry.shape:
            raise ValueError("feature_mask must have the same shape as telemetry")
        observed = feature_mask.to(telemetry.dtype)
        counts = observed.sum(dim=2).clamp(min=1.0)
        mean_values = (telemetry * observed).sum(dim=2) / counts
        observed_fraction = observed.mean(dim=2)
        node_states = self.node_projection(
            torch.cat((mean_values, observed_fraction), dim=-1)
        )
        graph_states = self.graph_encoder(node_states, adjacency, node_mask)
        return {
            "detection_logits": self.detection_head(graph_states).squeeze(-1),
            "localization_logits": self.localization_head(graph_states).squeeze(-1),
            "hidden_states": graph_states,
        }
