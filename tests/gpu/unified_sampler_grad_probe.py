from __future__ import annotations

import argparse

import torch

from ncls.learning.models import UnifiedNeuralModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler", choices=("nvidia-diffuse-ggx9", "ltc-k2"), required=True)
    parser.add_argument("--warm", action="store_true")
    args = parser.parse_args()
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
        evaluator="nvidia-frame-two-lobe-v1",
        runtime_class="realtime",
    ).cuda()
    state = torch.zeros(1, dtype=torch.int64, device="cuda")
    wo = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
    wi = torch.tensor([[[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]]], device="cuda")
    if args.warm:
        with torch.no_grad():
            prediction = model(state, wo, wi)
            assert torch.isfinite(prediction).all()
    model.set_sampler_training(args.sampler)
    if args.warm:
        with torch.no_grad():
            warm_pdf = model.sampler_pdf(state, wo, wi, args.sampler)
        assert torch.isfinite(warm_pdf).all()
    pdf, head = model.sampler_pdf_with_head(state, wo, wi, args.sampler)
    assert head.requires_grad and pdf.requires_grad
    loss = -torch.log(torch.clamp(pdf, min=1e-12)).mean()
    loss.backward()
    selected_prefix = "nvidia_sampler_" if args.sampler == "nvidia-diffuse-ggx9" else "ltc_sampler_"
    selected = []
    for name, parameter in model.named_parameters():
        if name.startswith(selected_prefix):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            selected.append(parameter.grad)
        else:
            assert parameter.grad is None
    assert selected and any(bool(torch.any(gradient != 0.0)) for gradient in selected)


if __name__ == "__main__":
    main()
