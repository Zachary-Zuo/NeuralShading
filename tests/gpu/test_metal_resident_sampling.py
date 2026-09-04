from __future__ import annotations

import pytest
import torch

from ncls.learning.mdl_metal_assets import (
    MdlMetalNativeAssetCollection,
    _ResidentMipPyramid,
)


pytestmark = pytest.mark.slangpy


def test_resident_mip_sampling_keeps_indices_and_patches_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda:0")
    level0 = torch.arange(3 * 8 * 8, dtype=torch.float32, device=device).reshape(
        3, 8 * 8
    )
    level1 = torch.arange(3 * 4 * 4, dtype=torch.float32, device=device).reshape(
        3, 4 * 4
    )
    pyramid = _ResidentMipPyramid(
        torch.cat((level0, level1), dim=1),
        torch.tensor(((8, 8), (4, 4)), dtype=torch.int64, device=device),
        torch.tensor((0, 64), dtype=torch.int64, device=device),
    )
    uv = torch.tensor(((0.5, 0.5), (0.99, 0.01)), device=device)
    requested = torch.tensor((0, 1), dtype=torch.int64, device=device)

    result = MdlMetalNativeAssetCollection._sample_gpu_pyramid(
        pyramid, uv, requested, 8, "wrap"
    )

    assert result.device == device
    assert result.shape == (2, 3, 8, 8)
    assert bool(torch.isfinite(result).all())
