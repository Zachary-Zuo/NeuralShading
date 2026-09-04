from __future__ import annotations

import pytest
import torch

from ncls.learning.appearance_metrics import (
    AppearanceMetricCalibration,
    appearance_error_metrics,
)


def _calibration() -> AppearanceMetricCalibration:
    return AppearanceMetricCalibration(
        scale_rgb=torch.tensor([0.5, 1.0, 2.0]),
        peak_rgb=torch.tensor([1.5, 2.5, 3.5]),
        energy_epsilon=1e-6,
    )


def test_appearance_metrics_expose_per_channel_chroma_peak_and_spatial_error() -> None:
    target = torch.tensor(
        [[[2.0, 3.0, 4.0], [0.25, 0.5, 1.0]]], dtype=torch.float32
    )
    prediction = target * torch.tensor([1.1, 0.8, 1.25])
    paired_target = target + torch.tensor([[[0.4, 0.2, 0.1], [0.1, 0.1, 0.2]]])
    paired_prediction = prediction + torch.tensor(
        [[[0.2, 0.4, 0.05], [0.05, 0.2, 0.1]]]
    )
    wi = torch.tensor([[[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]]])

    metrics = appearance_error_metrics(
        prediction,
        target,
        wi,
        _calibration(),
        paired_prediction=paired_prediction,
        paired_target=paired_target,
    )

    for family in (
        "appearance/log_rgb",
        "appearance/linear_rgb",
        "appearance/chroma",
        "appearance/peak_rgb",
        "appearance/spatial_gradient",
    ):
        assert family in metrics
        assert all(f"{family}/{channel}" in metrics for channel in ("r", "g", "b"))
        assert torch.isfinite(metrics[family])
    assert metrics["appearance/chroma"] > 0.0
    assert metrics["appearance/spatial_gradient"] > 0.0


def test_appearance_metrics_reject_partial_pair_and_nonfinite_prediction() -> None:
    value = torch.ones((2, 1, 3))
    wi = torch.tensor([[[0.0, 0.0, 1.0]]]).expand(2, 1, 3)
    with pytest.raises(ValueError, match="provided together"):
        appearance_error_metrics(
            value, value, wi, _calibration(), paired_prediction=value
        )
    invalid = value.clone()
    invalid[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        appearance_error_metrics(invalid, value, wi, _calibration())


def test_zero_calibrated_peak_does_not_turn_zero_energy_samples_into_peak_tail() -> None:
    target = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]])
    prediction = torch.tensor([[[10.0, 20.0, 30.0], [1.0, 1.0, 1.0]]])
    wi = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
    calibration = AppearanceMetricCalibration(
        scale_rgb=torch.ones(3),
        peak_rgb=torch.zeros(3),
        energy_epsilon=1e-6,
    )
    metrics = appearance_error_metrics(prediction, target, wi, calibration)
    torch.testing.assert_close(metrics["appearance/peak_rgb"], torch.zeros(()))
