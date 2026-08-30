from __future__ import annotations

import pytest
import torch

from ncls.learning.metal_asset_cook import MetalAssetCooker
from ncls.learning.models.metal_fused import (
    METAL_FUSED_REQUIRED_CONTEXT,
    MetalFusedNeuralMaterialModel,
)
from ncls.learning.source_adaptation import DenseNativeAssetCollection, NativeAssetRole


pytestmark = pytest.mark.slangpy


def _collection() -> DenseNativeAssetCollection:
    generator = torch.Generator().manual_seed(97)
    levels = []
    shape = 8
    while True:
        levels.append(torch.rand((shape, shape, 4), generator=generator))
        if shape == 1:
            break
        shape //= 2
    return DenseNativeAssetCollection(
        (tuple(levels),),
        ("cook-fixture",),
        "cook-schema",
        "surface-uv",
        "surface-uv",
        "wrap",
        (
            NativeAssetRole(
                "packed",
                "packed-correlated",
                0,
                4,
                "linear",
                "box-mip",
            ),
        ),
    )


def test_three_asset_cook_paths_share_decoder_profile_and_keep_distinct_identity() -> None:
    pytest.importorskip("slangpy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    model = MetalFusedNeuralMaterialModel.from_context(
        METAL_FUSED_REQUIRED_CONTEXT
    ).to("cuda:0")
    cooker = MetalAssetCooker(
        model,
        _collection(),
        max_core_texels=1024,
        encoder_halo=32,
    )
    encoder = cooker.cook_asset(0, mode="encoder-only")
    refined = cooker.cook_asset(
        0,
        mode="encoder-bounded-refinement",
        refinement_steps=1,
        refinement_bound=0.25,
    )
    direct = cooker.cook_asset(
        0,
        mode="direct-optimized-control",
        refinement_steps=1,
        refinement_bound=0.25,
    )
    assert len(encoder.records) == len(refined.records) == len(direct.records) == 4
    assert encoder.high_grid_int8.shape == refined.high_grid_int8.shape == direct.high_grid_int8.shape
    assert encoder.low_grid_int8.shape == refined.low_grid_int8.shape == direct.low_grid_int8.shape
    assert encoder.grid_scales.shape == refined.grid_scales.shape == direct.grid_scales.shape
    assert len({encoder.identity, refined.identity, direct.identity}) == 3
    assert refined.refinement_bound == 0.25
    assert encoder.refinement_steps == 0
