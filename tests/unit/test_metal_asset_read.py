import math

import pytest
import torch

from ncls.learning.methods.metal.asset_read import (
    EncodedAssetTile, plan_asset_read, read_bilinear, snorm8_ste,
)


def test_jacobian_uses_both_axes_and_zero_footprint_is_point():
    uv = torch.tensor([[0.2, 0.3]]).repeat(3, 1)
    affine = torch.tensor([[[0.0, -2.0, 0.1], [0.5, 0.0, -0.2]]]).repeat(3, 1, 1)
    dx = torch.tensor([[0.0, 0.0], [0.04, 0.0], [0.0, 0.015]])
    dy = torch.zeros_like(dx)
    plan = plan_asset_read(uv, dx, dy, affine, uv.new_tensor([[100, 20]]).repeat(3, 1),
                           uv.new_full((3,), 6), uv.new_tensor([0.4, 0.4, 0.2]))
    torch.testing.assert_close(plan.uv, uv.new_tensor([[-0.5, -0.1]]).repeat(3, 1))
    torch.testing.assert_close(plan.lod, uv.new_tensor([0, 0, math.log2(3)]))
    assert plan.level.tolist() == [0, 0, 2]
    features = plan.query_features(torch.tensor([[0., 0., 1.]]).repeat(3, 1))
    assert features.shape == (3, 8)
    assert not torch.equal(features[0], features[1])


@pytest.mark.parametrize("mode,expected", [("wrap", 0.3), ("clamp", 0.1)])
def test_bilinear_seam_and_negative_repeat(mode, expected):
    value = torch.tensor([[[[0.1, 0.5]]]])
    tile = EncodedAssetTile(value, (1, 2))
    result = read_bilinear(tile, torch.tensor([[0., 0.5]]), address_mode=mode, qat=False, full_level=True)
    torch.testing.assert_close(result, torch.tensor([[expected]]))
    if mode == "wrap":
        negative = read_bilinear(tile, torch.tensor([[-1., 0.5]]), address_mode=mode, qat=False, full_level=True)
        torch.testing.assert_close(negative, result)


def test_quantization_precedes_interpolation_and_keeps_gradients():
    source = torch.tensor([[[[0.001, 0.009]]]], requires_grad=True)
    tile = EncodedAssetTile(source, (1, 2))
    uv = torch.tensor([[0.4, 0.5]], requires_grad=True)
    result = read_bilinear(tile, uv, address_mode="clamp", full_level=True)
    expected = 0.7 * snorm8_ste(source)[0, 0, 0, 0] + 0.3 * snorm8_ste(source)[0, 0, 0, 1]
    torch.testing.assert_close(result[0, 0], expected)
    assert not torch.allclose(result[0, 0], snorm8_ste(0.7 * source[0, 0, 0, 0] + 0.3 * source[0, 0, 0, 1]))
    result.sum().backward()
    torch.testing.assert_close(source.grad, torch.tensor([[[[0.7, 0.3]]]]))
    assert uv.grad[0, 0] > 0


def test_tile_read_does_not_silently_clamp_missing_halo():
    with pytest.raises(ValueError, match="four read neighbours"):
        read_bilinear(EncodedAssetTile(torch.zeros(1, 4, 2, 2), (8, 8), (2, 2)),
                      torch.tensor([[0., 0.]]), address_mode="wrap")
