from __future__ import annotations

import pytest
import torch

from ncls.learning.methods.metal.model import MetalBudgetedModel


pytestmark = pytest.mark.slangpy


def _conditioning(device: torch.device) -> dict[str, torch.Tensor]:
    batch = 2
    presence = torch.zeros((batch, 32), dtype=torch.int64, device=device)
    presence[:, :12] = 1
    return {
        "source_index": torch.arange(batch, dtype=torch.int64, device=device),
        "wo": torch.nn.functional.normalize(
            torch.tensor([[0.25, -0.1, 1.0], [-0.15, 0.2, 1.0]], device=device),
            dim=1,
        ),
        "uv": torch.tensor([[0.2, 0.7], [0.73, 0.11]], device=device),
        "uv_dx": torch.tensor([[1.0 / 4096.0, 0.0]], device=device).expand(batch, -1).clone(),
        "uv_dy": torch.tensor([[0.0, 1.0 / 4096.0]], device=device).expand(batch, -1).clone(),
        "mip_level": torch.tensor([0.35, 1.65], device=device),
        "metal_mip_fraction": torch.tensor([0.35, 0.65], device=device),
        "metal_budgeted_detail": torch.rand(batch, 4, device=device) * 2.0 - 1.0,
        "metal_budgeted_context": torch.rand(batch, 4, device=device) * 2.0 - 1.0,
        "metal_graph_index": torch.arange(batch, dtype=torch.int64, device=device),
        "metal_schema_index": torch.arange(batch, dtype=torch.int64, device=device),
        "metal_recipe_index": torch.arange(batch, dtype=torch.int64, device=device),
        "metal_identity_index": torch.arange(batch, dtype=torch.int64, device=device),
        "metal_finish_index": torch.arange(batch, dtype=torch.int64, device=device),
        "metal_asset_index": torch.arange(batch, dtype=torch.int64, device=device),
        "metal_typed_semantic_id": torch.arange(32, dtype=torch.int64, device=device)[None].expand(batch, -1).clone(),
        "metal_typed_type_id": torch.remainder(
            torch.arange(32, dtype=torch.int64, device=device), 8
        )[None].expand(batch, -1).clone(),
        "metal_typed_responsibility_id": torch.remainder(
            torch.arange(32, dtype=torch.int64, device=device), 6
        )[None].expand(batch, -1).clone(),
        "metal_typed_discrete": torch.remainder(
            torch.arange(32, dtype=torch.int64, device=device), 7
        )[None].expand(batch, -1).clone(),
        "metal_typed_continuous": torch.linspace(
            -1.0, 1.0, batch * 32 * 4, device=device
        ).reshape(batch, 32, 4),
        "metal_typed_presence": presence,
        "metal_canonical_optical": torch.linspace(
            0.1, 0.9, batch * 16, device=device
        ).reshape(batch, 16),
        "metal_access_state": torch.tensor(
            [
                [1.2, 0.8, 0.1, -0.2, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 0, 0],
                [0.9, 1.1, -0.1, 0.2, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 0, 0],
            ],
            device=device,
        ),
        "metal_frame_state": torch.zeros(batch, 8, device=device),
        "metal_distribution_id": torch.zeros(
            batch, dtype=torch.int64, device=device
        ),
    }


def test_budgeted_cuda_bfloat16_evaluate_sample_and_pdf_are_finite() -> None:
    pytest.importorskip("slangpy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda:0")
    torch.manual_seed(20260904)
    model = MetalBudgetedModel().to(device)
    values = _conditioning(device)
    wi = torch.nn.functional.normalize(
        torch.tensor(
            [
                [[0.1, 0.3, 1.0], [-0.4, 0.2, 0.8]],
                [[-0.2, 0.4, 1.0], [0.6, -0.1, 0.5]],
            ],
            device=device,
        ),
        dim=-1,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prepared = model.prepare(values)
        evaluated = model.evaluate_prepared(prepared, values["wo"], wi)
        sampled = model.sample_prepared(
            prepared,
            values["wo"],
            torch.tensor([[0.31, 0.77], [0.81, 0.19]], device=device),
        )
        independent = model.pdf_prepared(prepared, values["wo"], sampled.wi)
        loss = evaluated.f.mean() + sampled.weight.mean()
    loss.backward()
    assert bool(torch.isfinite(evaluated.f).all())
    assert bool((evaluated.f >= 0.0).all())
    assert bool(sampled.valid.all())
    assert bool(torch.isfinite(sampled.weight).all())
    torch.testing.assert_close(
        sampled.forward_pdf.float(), independent.forward.float(), rtol=2e-4, atol=2e-5
    )
