"""锁定 NVIDIA frozen-evaluator LTC adaptation 的 warm-session 梯度隔离。"""

from __future__ import annotations

import pytest
import torch

from ncls.learning.models import (
    NvidiaNeuralAppearanceLtcAdaptationModel,
    NvidiaNeuralAppearanceModel,
)


pytest.importorskip("slangpy")


@pytest.mark.slangpy
def test_nvidia_matched_ltc_warm_inference_then_head_only_backward() -> None:
    if not torch.cuda.is_available():
        pytest.skip("SlangPy CUDA test requires a CUDA device")
    reproduction = NvidiaNeuralAppearanceModel(state_count=2).cuda()
    model = NvidiaNeuralAppearanceLtcAdaptationModel(reproduction).cuda()
    model.set_sampler_training("ltc-k2")
    state_index = torch.tensor([0, 1], device="cuda")
    wo = torch.tensor(
        [[0.0, 0.0, 1.0], [0.3, -0.2, 0.9327379]],
        device="cuda",
    )
    wi = torch.tensor(
        [
            [[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]],
            [[0.0, 0.6, 0.8], [-0.3, 0.4, 0.8660254]],
        ],
        device="cuda",
    )
    with torch.no_grad():
        warm = model.sampler_pdf(state_index, wo, wi, "ltc-k2")
    assert torch.all(torch.isfinite(warm))
    pdf, raw = model.sampler_pdf_with_head(state_index, wo, wi, "ltc-k2")
    assert pdf.requires_grad and raw.requires_grad
    (-torch.log(pdf).mean()).backward()
    target = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("ltc_sampler_")
    ]
    frozen = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("reproduction.")
    ]
    assert all(gradient is not None and torch.all(torch.isfinite(gradient)) for gradient in target)
    assert any(bool(torch.any(gradient != 0.0)) for gradient in target if gradient is not None)
    assert all(gradient is None for gradient in frozen)
