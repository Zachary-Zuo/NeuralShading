from __future__ import annotations

import torch

from ncls.core.representations.legacy_ltc_k2.torch_eval import (
    LegacyLtcK2Tensors,
    decode_ltc_residual,
    evaluate_state_response_cos,
)


def predict_legacy_ltc_k2_response(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    lights: torch.Tensor,
) -> torch.Tensor:
    raw = model(
        batch["interface_kinds"].long(),
        batch["continuous"].float(),
        batch["interface_counts"].long(),
        batch["view"].float(),
    )
    amplitude, inverse_scale, shear, angle = decode_ltc_residual(raw)
    state = LegacyLtcK2Tensors(
        interface_kind=batch["top_kind"].long(),
        alpha=batch["top_alpha"].float(),
        relative_ior=batch["top_relative_ior"].float(),
        eta=batch["top_eta"].float(),
        k=batch["top_k"].float(),
        color=batch["top_color"].float(),
        tangent_rotation=batch["top_rotation"].float(),
        amplitude=amplitude,
        inverse_scale=inverse_scale,
        shear=shear,
        angle=angle,
    )
    return evaluate_state_response_cos(state, batch["view"].float(), lights)
