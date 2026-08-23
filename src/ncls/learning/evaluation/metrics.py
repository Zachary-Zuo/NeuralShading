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


def evaluator_metric_distributions(
    prediction: torch.Tensor,
    target: torch.Tensor,
    standard_error: torch.Tensor,
    solid_angle_weight: torch.Tensor,
    light_directions: torch.Tensor,
) -> dict[str, np.ndarray]:
    """返回按 query group 分布的线性、log、峰值、能量与合法性指标。"""

    finite = torch.isfinite(prediction)
    finite_rate = torch.mean(finite.float(), dim=(1, 2))
    safe_prediction = torch.where(finite, prediction, torch.zeros_like(prediction))
    weights = solid_angle_weight.float()[..., None]
    absolute_error = torch.abs(safe_prediction - target)
    target_absolute = torch.abs(target)
    weighted_l1 = torch.sum(absolute_error * weights, dim=(1, 2)) / torch.clamp(
        torch.sum(target_absolute * weights, dim=(1, 2)), min=1e-8
    )
    relative_l1 = directional_relative_l1(safe_prediction, target)

    peak = torch.amax(target_absolute, dim=(1, 2), keepdim=True)
    log_scale = 0.01 * peak + 1e-6
    log_l1 = torch.mean(
        torch.abs(
            torch.log1p(torch.clamp(safe_prediction, min=0.0) / log_scale)
            - torch.log1p(torch.clamp(target, min=0.0) / log_scale)
        ),
        dim=(1, 2),
    )

    target_energy_rgb = torch.sum(target * weights, dim=1)
    predicted_energy_rgb = torch.sum(safe_prediction * weights, dim=1)
    energy_relative_error = torch.sum(
        torch.abs(predicted_energy_rgb - target_energy_rgb), dim=1
    ) / torch.clamp(torch.sum(torch.abs(target_energy_rgb), dim=1), min=1e-8)

    target_magnitude = torch.sum(torch.abs(target), dim=-1)
    predicted_magnitude = torch.sum(torch.abs(safe_prediction), dim=-1)
    target_peak_index = torch.argmax(target_magnitude, dim=1)
    predicted_peak_index = torch.argmax(predicted_magnitude, dim=1)
    rows = torch.arange(len(target), device=target.device)
    target_peak = target_magnitude[rows, target_peak_index]
    predicted_peak = predicted_magnitude[rows, predicted_peak_index]
    peak_ratio = predicted_peak / torch.clamp(target_peak, min=1e-8)
    peak_ratio_log_error = torch.abs(torch.log(torch.clamp(peak_ratio, min=1e-8)))
    target_peak_direction = light_directions[rows, target_peak_index]
    predicted_peak_direction = light_directions[rows, predicted_peak_index]
    peak_angle = torch.rad2deg(torch.acos(torch.clamp(
        torch.sum(target_peak_direction * predicted_peak_direction, dim=1), -1.0, 1.0
    )))

    direction_count = target.shape[1]
    top_count = max(1, int(np.ceil(direction_count * 0.05)))
    target_contribution = target_magnitude * solid_angle_weight.float()
    predicted_contribution = predicted_magnitude * solid_angle_weight.float()
    target_top = torch.topk(target_contribution, top_count, dim=1).indices
    predicted_top = torch.topk(predicted_contribution, top_count, dim=1).indices
    target_top_mask = torch.zeros_like(target_contribution, dtype=torch.bool)
    predicted_top_mask = torch.zeros_like(predicted_contribution, dtype=torch.bool)
    target_top_mask.scatter_(1, target_top, True)
    predicted_top_mask.scatter_(1, predicted_top, True)
    recalled_energy = torch.sum(
        target_contribution * target_top_mask * predicted_top_mask, dim=1
    )
    top_energy_recall = recalled_energy / torch.clamp(
        torch.sum(target_contribution * target_top_mask, dim=1), min=1e-8
    )

    error_over_reference_se = torch.sum(absolute_error, dim=(1, 2)) / torch.clamp(
        torch.sum(torch.abs(standard_error), dim=(1, 2)), min=1e-8
    )
    nonnegative_rate = torch.mean((safe_prediction >= 0.0).float(), dim=(1, 2))
    return {
        "solid_angle_normalized_l1": weighted_l1.detach().cpu().numpy(),
        "linear_relative_l1": relative_l1.detach().cpu().numpy(),
        "log_l1": log_l1.detach().cpu().numpy(),
        "energy_relative_error": energy_relative_error.detach().cpu().numpy(),
        "peak_ratio": peak_ratio.detach().cpu().numpy(),
        "peak_ratio_log_error": peak_ratio_log_error.detach().cpu().numpy(),
        "peak_angle_degrees": peak_angle.detach().cpu().numpy(),
        "top_5_percent_energy_recall": top_energy_recall.detach().cpu().numpy(),
        "model_error_over_reference_standard_error": error_over_reference_se.detach().cpu().numpy(),
        "finite_rate": finite_rate.detach().cpu().numpy(),
        "nonnegative_rate": nonnegative_rate.detach().cpu().numpy(),
    }


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "p5": float(np.quantile(values, 0.05)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }
