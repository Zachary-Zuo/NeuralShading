from __future__ import annotations

import math
from typing import Mapping

import torch


def reference_se_group_tail_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    standard_error: torch.Tensor,
    *,
    tail_fraction: float,
) -> torch.Tensor:
    """对齐正式 group-level error/reference-SE 指标的可微尾部损失。"""

    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")
    if prediction.shape != target.shape or target.shape != standard_error.shape:
        raise ValueError("prediction, target and standard_error must have identical shapes")
    if prediction.ndim != 3 or prediction.shape[0] == 0:
        raise ValueError("reference-SE group tail loss requires non-empty [group, wi, rgb]")
    group_ratio = torch.sum(torch.abs(prediction - target), dim=(1, 2)) / torch.clamp(
        torch.sum(torch.abs(standard_error), dim=(1, 2)), min=1e-8
    )
    tail_count = max(1, math.ceil(len(group_ratio) * tail_fraction))
    tail = torch.topk(group_ratio, tail_count, largest=True, sorted=False).values
    return torch.mean(torch.log1p(tail))


def energy_shape_terms(
    prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """把加权积分能量与归一化方向形状分开，供不同表示复用。"""

    target = torch.clamp(batch["mean"].float(), min=0.0)
    weights = batch["solid_angle_weight"].float()[..., None]
    predicted_contribution = torch.clamp(prediction, min=0.0) * weights
    target_contribution = target * weights
    predicted_energy = torch.sum(predicted_contribution, dim=1)
    target_energy = torch.sum(target_contribution, dim=1)
    energy_floor = 1e-5 * torch.amax(target_energy, dim=1, keepdim=True) + 1e-8
    energy_loss = torch.mean(torch.square(
        torch.log(predicted_energy + energy_floor)
        - torch.log(target_energy + energy_floor)
    ))
    predicted_distribution = predicted_contribution / torch.clamp(
        predicted_energy[:, None, :], min=1e-12
    )
    target_distribution = target_contribution / torch.clamp(
        target_energy[:, None, :], min=1e-12
    )
    shape_loss = torch.mean(torch.sum(torch.square(
        torch.sqrt(predicted_distribution + 1e-12)
        - torch.sqrt(target_distribution + 1e-12)
    ), dim=1))
    return energy_loss, shape_loss
