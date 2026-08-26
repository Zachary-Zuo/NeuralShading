"""SlangPy 锁定 NVIDIA baseline 的 inference→training callable 身份与 detach 边界。"""

from __future__ import annotations

import torch
import pytest

from ncls.learning.models import NvidiaNeuralAppearanceModel
from ncls.learning.pipelines.sampler_objective import sampler_cross_entropy


pytest.importorskip("slangpy")


def _queries() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state = torch.tensor([0], device="cuda")
    wo = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
    wi = torch.tensor(
        [[[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]]], device="cuda"
    )
    area = torch.ones((1, 2), device="cuda")
    return state, wo, wi, area


@pytest.mark.slangpy
def test_nvidia_evaluator_backward_survives_inference_warmup() -> None:
    model = NvidiaNeuralAppearanceModel(state_count=1).cuda()
    state, wo, wi, _ = _queries()
    with torch.no_grad():
        warm = model(state, wo, wi)
    assert torch.isfinite(warm).all()
    loss = model(state, wo, wi).mean()
    loss.backward()
    assert model.latent.grad is not None
    assert model.frame_w.grad is not None
    assert model.evaluate_w0.grad is not None
    assert torch.isfinite(model.latent.grad).all()
    assert torch.isfinite(model.frame_w.grad).all()
    assert torch.isfinite(model.evaluate_w0.grad).all()


@pytest.mark.slangpy
def test_nvidia_sampler_backward_survives_pdf_warmup_and_detaches_latent() -> None:
    model = NvidiaNeuralAppearanceModel(state_count=1).cuda()
    state, wo, wi, area = _queries()
    with torch.no_grad():
        warm = model.sampler_pdf(state, wo, wi, "nvidia-diffuse-ggx9")
    assert torch.isfinite(warm).all()
    target = model(state, wo, wi).detach()
    proposal, raw = model.sampler_pdf_with_head(
        state, wo, wi, "nvidia-diffuse-ggx9"
    )
    assert proposal.requires_grad and raw.requires_grad
    loss, _ = sampler_cross_entropy(target, wi, area, proposal)
    loss.backward()
    assert model.sampler_w0.grad is not None
    assert torch.isfinite(model.sampler_w0.grad).all()
    assert model.latent.grad is None
    assert model.frame_w.grad is None
    assert model.evaluate_w0.grad is None
