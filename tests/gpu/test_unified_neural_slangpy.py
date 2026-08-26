"""锁定SlangPy callable active-gradient mask的cold/warm sampler回归。"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import torch

from ncls.learning.models import UnifiedNeuralModel


PROBE = Path(__file__).with_name("unified_sampler_grad_probe.py")


@pytest.mark.parametrize("sampler", ("nvidia-diffuse-ggx9", "ltc-k2"))
@pytest.mark.parametrize("warm", (False, True), ids=("cold", "warm-evaluator-first"))
def test_sampler_only_gradients_survive_callable_order(sampler: str, warm: bool) -> None:
    command = [sys.executable, str(PROBE), "--sampler", sampler]
    if warm:
        command.append("--warm")
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.slangpy
def test_core_evaluator_gradients_survive_initial_validation_warm() -> None:
    if not torch.cuda.is_available():
        pytest.skip("SlangPy CUDA test requires a CUDA device")
    top = {
        "interface_kind": 3,
        "alpha": [0.2, 0.2],
        "relative_ior": 1.0,
        "eta": [0.0, 0.0, 0.0],
        "k": [0.0, 0.0, 0.0],
        "color": [0.5, 0.5, 0.5],
        "tangent_rotation": 0.0,
    }
    model = UnifiedNeuralModel(
        state_count=1,
        response_scale=[[1.0, 1.0, 1.0]],
        top_rows=[top],
        evaluator="core-frame-neural-v1",
        runtime_class="realtime",
    ).cuda()
    state = torch.zeros(1, dtype=torch.int64, device="cuda")
    wo = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
    wi = torch.tensor(
        [[[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]]], device="cuda"
    )
    with torch.no_grad():
        warm = model(state, wo, wi)
    assert torch.all(torch.isfinite(warm))
    prediction = model(state, wo, wi)
    assert prediction.requires_grad
    prediction.mean().backward()
    trainable = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert trainable and all(
        gradient is not None and torch.all(torch.isfinite(gradient))
        for gradient in trainable
    )
