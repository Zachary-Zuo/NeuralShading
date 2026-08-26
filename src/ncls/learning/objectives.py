from __future__ import annotations

import math

import torch


def sampler_cross_entropy(
    evaluator_f: torch.Tensor,
    wi: torch.Tensor,
    solid_angle_weight: torch.Tensor,
    proposal_pdf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 evaluator 能量分布计算 matched sampler 的 cross entropy。"""

    cosine = torch.clamp(wi[..., 2], min=0.0)
    luminance = (
        0.2126 * evaluator_f[..., 0]
        + 0.7152 * evaluator_f[..., 1]
        + 0.0722 * evaluator_f[..., 2]
    )
    energy = torch.clamp(luminance * cosine, min=0.0)
    weighted_mass = energy * solid_angle_weight
    normalization = weighted_mass.sum(dim=1, keepdim=True)
    cosine_mass = cosine / math.pi * solid_angle_weight
    cosine_mass = cosine_mass / torch.clamp(cosine_mass.sum(dim=1, keepdim=True), min=1e-12)
    target_mass = torch.where(
        normalization > 1e-12,
        weighted_mass / torch.clamp(normalization, min=1e-12),
        cosine_mass,
    ).detach()
    cross_entropy = -(target_mass * torch.log(torch.clamp(proposal_pdf, min=1e-12))).sum(dim=1)
    target_entropy = -(target_mass * torch.log(torch.clamp(target_mass, min=1e-12))).sum(dim=1)
    return cross_entropy.mean(), (cross_entropy - target_entropy).mean()


__all__ = ["sampler_cross_entropy"]
