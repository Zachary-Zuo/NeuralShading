from __future__ import annotations

from typing import Mapping

import torch


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
