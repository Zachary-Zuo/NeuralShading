from __future__ import annotations

import pytest
import torch

from ncls.learning.methods.nvidia.model import NvidiaModel
from ncls.learning.objectives import sampler_forward_kl_score
from ncls.learning.training.engine import TrainingEngine


pytestmark = pytest.mark.slangpy


def _hemisphere(count: int, *, device: torch.device) -> torch.Tensor:
    values = torch.randn((count, 3), dtype=torch.float32, device=device)
    values[:, 2].abs_().add_(0.1)
    return torch.nn.functional.normalize(values, dim=-1)


def test_fp32_torch_training_core_matches_slang_functional_oracle() -> None:
    pytest.importorskip("slangpy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda:0")
    torch.manual_seed(71)
    torch.cuda.manual_seed_all(71)
    model = NvidiaModel(
        native_feature_count=5,
        latent_width=1,
        latent_height=1,
        latent_mip_count=1,
    ).to(device)
    latent = torch.randn((257, 8), dtype=torch.float32, device=device)
    wo = _hemisphere(257, device=device)
    wi = _hemisphere(257 * 3, device=device).reshape(257, 3, 3)

    with torch.no_grad():
        torch_f = model.evaluate_f(latent, wo, wi)
        slang_f = model.evaluate_f_slang(latent, wo, wi)
        raw = model.sampler_raw(latent, wo, detach_latent=True)
        torch_pdf = model._sampler_pdf_from_raw(raw, wo, wi)
        slang_pdf = model.sampler_pdf_slang(raw, wo, wi)

    torch.testing.assert_close(torch_f, slang_f, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(torch_pdf, slang_pdf, rtol=5e-5, atol=2e-6)


def test_fp32_torch_training_core_has_finite_joint_gradients() -> None:
    pytest.importorskip("slangpy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda:0")
    model = NvidiaModel(
        native_feature_count=5,
        latent_width=1,
        latent_height=1,
        latent_mip_count=1,
    ).to(device)
    latent = torch.randn((1024, 8), dtype=torch.float32, device=device, requires_grad=True)
    wo = _hemisphere(1024, device=device)
    wi = _hemisphere(1024, device=device)[:, None, :]
    evaluate_f = model.evaluate_f(latent, wo, wi).mean()
    raw = model.sampler_raw(latent, wo, detach_latent=True)
    pdf = model._sampler_pdf_from_raw(raw, wo, wi).mean()
    (evaluate_f + pdf).backward()

    assert latent.grad is not None and bool(torch.isfinite(latent.grad).all())
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_sampler_score_path_updates_only_sampler_head() -> None:
    pytest.importorskip("slangpy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda:0")
    torch.manual_seed(89)
    torch.cuda.manual_seed_all(89)
    model = NvidiaModel(
        native_feature_count=5,
        latent_width=1,
        latent_height=1,
        latent_mip_count=1,
    ).to(device)
    latent = torch.randn((2048, 8), dtype=torch.float32, device=device, requires_grad=True)
    wo = _hemisphere(2048, device=device)
    sample_u = torch.rand((2048, 2), dtype=torch.float32, device=device)
    wi, pdf, _, valid = model.sampler_sample_with_head(
        latent, wo, sample_u, "nvidia-diffuse-ggx9"
    )
    evaluator_f = torch.full((2048, 1, 3), 0.25, dtype=torch.float32, device=device)
    loss, valid_fraction = sampler_forward_kl_score(evaluator_f, wi, pdf, valid)
    assert float(valid_fraction) > 0.5
    loss.backward()

    assert latent.grad is None
    sampler_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("sampler_")
    ]
    other_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("sampler_")
    ]
    assert any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in sampler_parameters
    )
    assert all(parameter.grad is None for parameter in other_parameters)


def test_checkpoint_rng_restore_accepts_cuda_mapped_byte_tensors() -> None:
    pytest.importorskip("slangpy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    state = dict(TrainingEngine._rng_state())
    state["torch"] = state["torch"].cuda()
    state["cuda"] = [value.cuda() for value in state["cuda"]]
    TrainingEngine._restore_rng(state)
