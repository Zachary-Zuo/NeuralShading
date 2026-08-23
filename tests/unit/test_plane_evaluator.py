from __future__ import annotations

import torch
import pytest

from ncls.learning.models.plane_evaluator import (
    PlaneFactorizedModelConfig,
    PlaneFactorizedNeuralEvaluator,
)


def test_plane_bilinear_lookup_matches_corners_and_center() -> None:
    plane = torch.tensor([[[0.0, 2.0], [4.0, 6.0]]])
    coordinates = torch.tensor([
        [-1.0, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
    ])
    values = PlaneFactorizedNeuralEvaluator._bilinear(plane, coordinates)
    torch.testing.assert_close(values[:, 0], torch.tensor([0.0, 2.0, 4.0, 6.0, 3.0]))


def test_plane_v1_limits_each_texel_to_one_rgba_feature_vector() -> None:
    with pytest.raises(ValueError, match="RGBA-width"):
        PlaneFactorizedModelConfig(plane_feature_dimension=5)
