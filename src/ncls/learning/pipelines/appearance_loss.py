from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
from torch.nn import functional as F


def p1_appearance_loss(
    prediction_f: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    target_scale: torch.Tensor | Sequence[Sequence[float]],
) -> torch.Tensor:
    """`p1-appearance-v3`：log smooth-L1 + 0.25 线性 L1 + 0.10 能量 + 0.15 峰值。

    `prediction_f` 是线性 RGB `f`，非负性由模型参数化保证（softplus 输出、LTC lobe 幅值），
    这里不再 clamp，避免 `p1_audit.md` §4.2 的梯度死区。
    """

    target = torch.clamp(batch["mean"].float(), min=0.0)
    cosine = torch.abs(batch["wi"].float()[..., 2:3])
    prediction_y = prediction_f * cosine
    scales = torch.as_tensor(
        target_scale,
        dtype=target.dtype,
        device=target.device,
    )[batch["state_index"].long()][:, None, :]
    transformed = F.smooth_l1_loss(
        torch.log(prediction_y / scales + 1e-4),
        torch.log(target / scales + 1e-4),
    )

    weights = batch["solid_angle_weight"].float()[..., None]
    linear_numerator = torch.sum(torch.abs(prediction_y - target) * weights, dim=(1, 2))
    scale_energy_envelope = torch.sum(scales, dim=(1, 2)) * torch.sum(weights, dim=(1, 2))
    linear_denominator = torch.clamp(
        torch.sum(torch.abs(target) * weights, dim=(1, 2)),
        min=1e-8,
    )
    linear_denominator = torch.maximum(linear_denominator, 1e-3 * scale_energy_envelope)
    linear = torch.mean(linear_numerator / linear_denominator)

    predicted_energy = torch.sum(prediction_y * weights, dim=1)
    target_energy = torch.sum(target * weights, dim=1)
    energy_scale = torch.clamp(torch.amax(target_energy, dim=1, keepdim=True), min=1e-6)
    energy = F.smooth_l1_loss(
        torch.log1p(predicted_energy / energy_scale),
        torch.log1p(target_energy / energy_scale),
    )

    magnitude = torch.sum(target, dim=-1)
    peak_count = max(1, int(math.ceil(target.shape[1] * 0.05)))
    peak_indices = torch.topk(magnitude, peak_count, dim=1, sorted=False).indices
    peak_mask = torch.zeros_like(magnitude, dtype=torch.bool).scatter_(1, peak_indices, True)
    peak_error = torch.abs(prediction_y - target)[peak_mask]
    peak_target = torch.abs(target)[peak_mask]
    peak = torch.mean(
        peak_error / torch.clamp(peak_target + scales.expand_as(target)[peak_mask], min=1e-6)
    )
    return transformed + 0.25 * linear + 0.10 * energy + 0.15 * peak
