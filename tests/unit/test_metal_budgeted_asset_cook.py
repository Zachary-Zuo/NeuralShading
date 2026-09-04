from __future__ import annotations

import pytest
import torch

from ncls.learning.models.metal_budgeted_asset import MetalBudgetedAssetCooker


def test_budgeted_asset_cook_modes_share_shape_and_keep_distinct_identity() -> None:
    encoded = torch.linspace(-0.4, 0.4, 4 * 3 * 2 * 4).reshape(4, 3, 2, 4)
    target = torch.full_like(encoded, 0.8)

    def objective(value: torch.Tensor) -> torch.Tensor:
        return torch.mean((value - target).square())

    encoder = MetalBudgetedAssetCooker.cook(
        encoded, mode="encoder-only@1"
    )
    refined = MetalBudgetedAssetCooker.cook(
        encoded,
        mode="bounded-refinement@1",
        objective=objective,
        refinement_steps=8,
        refinement_bound=0.1,
    )
    direct = MetalBudgetedAssetCooker.cook(
        encoded,
        mode="direct-control@1",
        objective=objective,
        refinement_steps=8,
    )
    assert encoder.values_snorm8.shape == refined.values_snorm8.shape == direct.values_snorm8.shape
    assert encoder.values_snorm8.dtype == refined.values_snorm8.dtype == direct.values_snorm8.dtype == torch.int8
    assert len({encoder.identity, refined.identity, direct.identity}) == 3
    assert float(torch.max(torch.abs(refined.values - encoded))) <= 0.1 + 1.0 / 127.0
    assert torch.mean((direct.values - target).square()) < torch.mean(target.square())


def test_budgeted_asset_cook_rejects_hidden_optimization_and_unbounded_refine() -> None:
    values = torch.zeros(2, 2, 4)
    with pytest.raises(ValueError, match="cannot optimize"):
        MetalBudgetedAssetCooker.cook(
            values,
            mode="encoder-only@1",
            objective=lambda value: value.square().mean(),
        )
    with pytest.raises(ValueError, match="must lie"):
        MetalBudgetedAssetCooker.cook(
            values,
            mode="bounded-refinement@1",
            objective=lambda value: value.square().mean(),
            refinement_steps=1,
            refinement_bound=0.75,
        )


def test_optimized_asset_cook_does_not_accumulate_captured_runtime_gradients() -> None:
    encoded = torch.zeros(2, 2, 4)
    runtime_weight = torch.nn.Parameter(torch.tensor(2.0))
    before = runtime_weight.detach().clone()

    MetalBudgetedAssetCooker.cook(
        encoded,
        mode="bounded-refinement@1",
        objective=lambda value: (runtime_weight * value - 1.0).square().mean(),
        refinement_steps=2,
        refinement_bound=0.1,
    )

    assert runtime_weight.grad is None
    torch.testing.assert_close(runtime_weight.detach(), before)
