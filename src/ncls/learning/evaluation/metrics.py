from __future__ import annotations

import numpy as np
import torch


def response_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    standard_error: torch.Tensor,
) -> torch.Tensor:
    peak = torch.amax(target, dim=(1, 2), keepdim=True)
    floor = 1e-3 * peak + 1e-5
    confidence = torch.clamp(
        (target + floor) / (target + floor + standard_error),
        0.1,
        1.0,
    ).detach()
    log_delta = torch.log(prediction + floor) - torch.log(target + floor)
    log_loss = torch.sum(confidence * log_delta * log_delta) / torch.sum(confidence)
    smape = torch.sum(
        confidence * 2.0 * torch.abs(prediction - target) / (prediction + target + floor)
    ) / torch.sum(confidence)
    return log_loss + 0.05 * smape


def directional_relative_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(prediction - target), dim=(1, 2)) / torch.clamp(
        torch.sum(torch.abs(target), dim=(1, 2)), min=1e-8
    )


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "p95": float(np.quantile(values, 0.95)),
    }
