from __future__ import annotations

import pytest


def _torch():
    return pytest.importorskip("torch")


def _inputs():
    torch = _torch()
    telemetry = torch.randn(2, 3, 5, 4)
    feature_mask = torch.ones_like(telemetry, dtype=torch.bool)
    delta_t = torch.full((2, 3, 5), 1.0)
    step_mask = torch.ones((2, 3, 5), dtype=torch.bool)
    adjacency = torch.zeros((2, 3, 3), dtype=torch.bool)
    node_mask = torch.tensor([[True, True, False], [True, True, True]])
    return telemetry, feature_mask, delta_t, step_mask, adjacency, node_mask


def test_async_st_gnn_outputs_detection_and_emitter_scores_with_empty_graphs() -> None:
    torch = _torch()
    from corridor_aiops.models.async_st_gnn import AsyncSTGNN

    model = AsyncSTGNN(feature_count=4, hidden_size=16, heads=4, temporal_layers=1)
    inputs = _inputs()

    output = model(*inputs)

    assert output["detection_logits"].shape == (2, 3)
    assert output["localization_logits"].shape == (2, 3)
    assert output["temporal_states"].shape == (2, 3, 5, 16)
    assert torch.isfinite(output["detection_logits"]).all()
    assert torch.isfinite(output["localization_logits"]).all()
    assert torch.equal(output["hidden_states"][0, 2], torch.zeros(16))


def test_async_st_gnn_temporal_attention_is_causal() -> None:
    torch = _torch()
    from corridor_aiops.models.async_st_gnn import AsyncSTGNN

    torch.manual_seed(7)
    model = AsyncSTGNN(feature_count=4, hidden_size=16, heads=4, temporal_layers=1)
    model.eval()
    inputs = list(_inputs())
    inputs[0] = inputs[0][:1, :1].clone()
    inputs[1] = inputs[1][:1, :1].clone()
    inputs[2] = inputs[2][:1, :1].clone()
    inputs[3] = inputs[3][:1, :1].clone()
    inputs[4] = inputs[4][:1, :1, :1].clone()
    inputs[5] = inputs[5][:1, :1].clone()
    changed = [tensor.clone() for tensor in inputs]
    changed[0][:, :, 4, :] += 1000.0

    with torch.no_grad():
        original_output = model(*inputs)
        changed_output = model(*changed)

    assert torch.allclose(
        original_output["temporal_states"][:, :, :4],
        changed_output["temporal_states"][:, :, :4],
        atol=1e-6,
        rtol=1e-6,
    )


def test_static_gcn_accepts_identity_fallback_and_masks_padding() -> None:
    torch = _torch()
    from corridor_aiops.models.async_st_gnn import StaticGCN

    model = StaticGCN(feature_count=4, hidden_size=16)
    telemetry, masks, _, _, adjacency, node_mask = _inputs()

    output = model(telemetry, masks, adjacency, node_mask)

    assert output["detection_logits"].shape == (2, 3)
    assert torch.isfinite(output["localization_logits"]).all()
    assert torch.equal(output["hidden_states"][0, 2], torch.zeros(16))


def test_async_st_gnn_handles_nodes_without_observed_steps() -> None:
    torch = _torch()
    from corridor_aiops.models.async_st_gnn import AsyncSTGNN

    model = AsyncSTGNN(feature_count=4, hidden_size=16, heads=4, temporal_layers=1)
    inputs = list(_inputs())
    inputs[1] = torch.zeros_like(inputs[1])
    inputs[3] = torch.zeros_like(inputs[3])
    inputs[5] = torch.ones_like(inputs[5])
    changed = [tensor.clone() for tensor in inputs]
    changed[0] = torch.full_like(changed[0], 10_000.0)

    with torch.no_grad():
        original_output = model(*inputs)
        changed_output = model(*changed)

    assert torch.isfinite(original_output["hidden_states"]).all()
    assert torch.equal(
        original_output["hidden_states"],
        torch.zeros_like(original_output["hidden_states"]),
    )
    assert torch.equal(
        original_output["hidden_states"],
        changed_output["hidden_states"],
    )


def test_tabular_transformer_is_causal_and_masks_unobserved_features() -> None:
    torch = _torch()
    from corridor_aiops.models.transformer import TabularTransformer

    torch.manual_seed(11)
    model = TabularTransformer(
        feature_count=4,
        hidden_size=16,
        heads=4,
        layers=1,
        max_steps=5,
    )
    model.eval()
    telemetry = torch.randn(1, 5, 4)
    feature_mask = torch.ones_like(telemetry, dtype=torch.bool)
    step_mask = torch.ones((1, 5), dtype=torch.bool)
    changed = telemetry.clone()
    changed[:, 4, :] += 1000.0
    missing_mask = feature_mask.clone()
    missing_mask[:, :, 0] = False
    missing_changed = telemetry.clone()
    missing_changed[:, :, 0] += 1000.0

    with torch.no_grad():
        original = model(telemetry, feature_mask, step_mask)
        future_changed = model(changed, feature_mask, step_mask)
        missing_original = model(telemetry, missing_mask, step_mask)
        missing_changed_output = model(missing_changed, missing_mask, step_mask)

    assert torch.allclose(
        original["hidden_states"][:, :4],
        future_changed["hidden_states"][:, :4],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        missing_original["detection_logits"],
        missing_changed_output["detection_logits"],
        atol=1e-6,
        rtol=1e-6,
    )
