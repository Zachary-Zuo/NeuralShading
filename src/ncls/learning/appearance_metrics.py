from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


_RGB_NAMES = ("r", "g", "b")


def _require_tensor(condition: torch.Tensor, message: str) -> None:
    if condition.device.type == "cuda":
        torch._assert_async(condition)
    elif not bool(condition):
        raise ValueError(message)


@dataclass(frozen=True)
class AppearanceMetricCalibration:
    """由 train-only reference calibration 冻结的 RGB metric 尺度。"""

    scale_rgb: torch.Tensor
    peak_rgb: torch.Tensor
    energy_epsilon: float
    robust_epsilon: float = 1e-3

    def tensors_like(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.scale_rgb.shape != (3,) or self.peak_rgb.shape != (3,):
            raise ValueError("appearance calibration RGB tensors must have shape [3]")
        if not self.scale_rgb.is_floating_point() or not self.peak_rgb.is_floating_point():
            raise ValueError("appearance calibration RGB tensors must be floating point")
        if not self.energy_epsilon > 0.0 or not self.robust_epsilon > 0.0:
            raise ValueError("appearance calibration epsilons must be positive")
        scale = self.scale_rgb.to(device=value.device, dtype=value.dtype)
        peak = self.peak_rgb.to(device=value.device, dtype=value.dtype)
        _require_tensor(
            torch.isfinite(scale).all() & (scale > 0.0).all(),
            "appearance calibration scale_rgb must be finite and positive",
        )
        _require_tensor(
            torch.isfinite(peak).all() & (peak >= 0.0).all(),
            "appearance calibration peak_rgb must be finite and nonnegative",
        )
        return scale, peak


def _require_rgb_pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.shape != target.shape or prediction.ndim < 2 or prediction.shape[-1] != 3:
        raise ValueError("appearance prediction and target must share [...,3] shape")
    if prediction.device != target.device or prediction.dtype != target.dtype:
        raise ValueError("appearance prediction and target must share device and dtype")
    if not prediction.is_floating_point():
        raise ValueError("appearance prediction and target must be floating point")


def _channel_metrics(
    result: dict[str, torch.Tensor], name: str, values: torch.Tensor
) -> None:
    result[name] = values.mean()
    for index, channel in enumerate(_RGB_NAMES):
        result[f"{name}/{channel}"] = values[..., index].mean()


def _masked_channel_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    numerator = torch.sum(values * weights[..., None], dim=tuple(range(values.ndim - 1)))
    denominator = torch.clamp(weights.sum(), min=1.0)
    return numerator / denominator


def appearance_error_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    wi: torch.Tensor,
    calibration: AppearanceMetricCalibration,
    *,
    paired_prediction: torch.Tensor | None = None,
    paired_target: torch.Tensor | None = None,
) -> Mapping[str, torch.Tensor]:
    """计算逐通道 HDR、色度、峰值和可选 paired-UV 空间差分误差。"""

    _require_rgb_pair(prediction, target)
    if wi.shape != prediction.shape:
        raise ValueError("appearance wi must match prediction shape")
    scale, peak = calibration.tensors_like(prediction)
    _require_tensor(
        torch.isfinite(prediction).all() & (prediction >= 0.0).all(),
        "appearance prediction must be finite and nonnegative",
    )
    _require_tensor(
        torch.isfinite(target).all() & (target >= 0.0).all(),
        "appearance target must be finite and nonnegative",
    )

    log_prediction = torch.log1p(prediction / scale)
    log_target = torch.log1p(target / scale)
    log_difference = log_prediction - log_target
    robust = torch.sqrt(
        log_difference.square() + calibration.robust_epsilon**2
    ) - calibration.robust_epsilon
    cosine = torch.clamp(torch.abs(wi[..., 2]), max=1.0)
    linear = torch.abs(prediction - target) * cosine[..., None]

    result: dict[str, torch.Tensor] = {}
    _channel_metrics(result, "appearance/log_rgb", robust)
    _channel_metrics(result, "appearance/linear_rgb", linear)

    prediction_chroma = log_prediction - log_prediction.mean(dim=-1, keepdim=True)
    target_chroma = log_target - log_target.mean(dim=-1, keepdim=True)
    chroma_error = torch.abs(prediction_chroma - target_chroma)
    energy_valid = target.sum(dim=-1) > calibration.energy_epsilon
    chroma_channels = _masked_channel_mean(chroma_error, energy_valid)
    result["appearance/chroma"] = chroma_channels.mean()
    for index, channel in enumerate(_RGB_NAMES):
        result[f"appearance/chroma/{channel}"] = chroma_channels[index]

    peak_support = (target >= peak) & (target > 0.0)
    peak_weights = peak_support.to(target.dtype)
    peak_numerator = torch.sum(
        robust * peak_weights, dim=tuple(range(robust.ndim - 1))
    )
    peak_denominator = torch.clamp(
        peak_weights.sum(dim=tuple(range(peak_weights.ndim - 1))), min=1.0
    )
    peak_channels = peak_numerator / peak_denominator
    result["appearance/peak_rgb"] = peak_channels.mean()
    for index, channel in enumerate(_RGB_NAMES):
        result[f"appearance/peak_rgb/{channel}"] = peak_channels[index]

    if (paired_prediction is None) != (paired_target is None):
        raise ValueError("paired appearance prediction and target must be provided together")
    if paired_prediction is not None and paired_target is not None:
        _require_rgb_pair(paired_prediction, paired_target)
        if paired_prediction.shape != prediction.shape:
            raise ValueError("paired appearance tensors must match the primary shape")
        _require_tensor(
            torch.isfinite(paired_prediction).all()
            & (paired_prediction >= 0.0).all(),
            "paired appearance prediction must be finite and nonnegative",
        )
        _require_tensor(
            torch.isfinite(paired_target).all() & (paired_target >= 0.0).all(),
            "paired appearance target must be finite and nonnegative",
        )
        predicted_gradient = (
            torch.log1p(paired_prediction / scale) - log_prediction
        )
        target_gradient = torch.log1p(paired_target / scale) - log_target
        spatial = torch.abs(predicted_gradient - target_gradient)
        _channel_metrics(result, "appearance/spatial_gradient", spatial)

    return result


__all__ = ["AppearanceMetricCalibration", "appearance_error_metrics"]
