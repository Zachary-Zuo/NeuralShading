from __future__ import annotations

import numpy as np
import pytest
import torch

from ncls.learning.methods.metal.native_assets import (
    MdlMetalNativeAssetCollection,
    _ResidentMipPyramid,
    _canonicalize_decoded_channels,
    _mip_shapes,
)
from ncls.learning.source_adaptation import (
    NativeAssetDescriptor,
    NativeAssetDomain,
    NativeAssetRole,
    NativeAssetTileRequest,
)


def _lazy_collection(
    values: np.ndarray,
    slot: dict[str, object],
    domain: NativeAssetDomain,
) -> MdlMetalNativeAssetCollection:
    collection = MdlMetalNativeAssetCollection.__new__(MdlMetalNativeAssetCollection)
    collection.descriptors = (NativeAssetDescriptor("asset", "schema", (domain,)),)
    collection._base_array = lambda _asset_index, _slot_index: (
        values,
        {"shape": "2d", "data_origin": "top_left"},
        slot,
    )
    return collection


def test_lazy_tile_broadcasts_scalar_payload_to_registry_rgb_channels() -> None:
    values = np.arange(16, dtype=np.uint8).reshape(4, 4, 1)
    slot = {
        "channels": {"RGB": "mask"},
        "transfer": "identity-linear",
    }
    domain = NativeAssetDomain(
        "slot-1",
        "surface-uv",
        "clamp",
        _mip_shapes(4, 4),
        (NativeAssetRole("mask", "mask", 0, 3, "identity-linear", "box"),),
    )
    request = NativeAssetTileRequest(
        0,
        "asset",
        "schema",
        "slot-1",
        domain.role_layout_id,
        0,
        (1, 1),
        (2, 2),
        1,
    )
    tile = _lazy_collection(values, slot, domain)._load_lazy_tile(
        request, domain, torch.device("cpu")
    )

    assert tile.shape == (4, 4, 3)
    expected = torch.arange(16, dtype=torch.float32).reshape(4, 4, 1) / 255.0
    assert torch.equal(tile, expected.expand(4, 4, 3))


def test_lazy_tile_mip_reduction_preserves_expanded_channel_layout() -> None:
    values = np.arange(64, dtype=np.uint8).reshape(8, 8, 1)
    slot = {
        "channels": {"RGB": "mask"},
        "transfer": "identity-linear",
    }
    domain = NativeAssetDomain(
        "slot-1",
        "surface-uv",
        "clamp",
        _mip_shapes(8, 8),
        (NativeAssetRole("mask", "mask", 0, 3, "identity-linear", "box"),),
    )
    request = NativeAssetTileRequest(
        0,
        "asset",
        "schema",
        "slot-1",
        domain.role_layout_id,
        1,
        (1, 1),
        (2, 2),
        1,
    )
    tile = _lazy_collection(values, slot, domain)._load_lazy_tile(
        request, domain, torch.device("cpu")
    )

    assert tile.shape == (4, 4, 3)
    assert torch.equal(tile[..., 0], tile[..., 1])
    assert torch.equal(tile[..., 1], tile[..., 2])


def test_decoded_channel_adapter_rejects_unsupported_partial_layout() -> None:
    with pytest.raises(ValueError, match="semantic channel contract"):
        _canonicalize_decoded_channels(
            np.zeros((2, 2, 2), dtype=np.float32), (("mask", 3),)
        )


def test_mip_patch_sampling_uses_canonical_channels_for_scalar_payload() -> None:
    values = np.arange(64, dtype=np.uint8).reshape(8, 8, 1)
    slot = {
        "channels": {"RGB": "mask"},
        "transfer": "identity-linear",
    }
    domain = NativeAssetDomain(
        "slot-1",
        "surface-uv",
        "clamp",
        _mip_shapes(8, 8),
        (NativeAssetRole("mask", "mask", 0, 3, "identity-linear", "box"),),
    )
    collection = _lazy_collection(values, slot, domain)
    patches = collection._sample_mip_patches(
        0, 1, 1, np.asarray([[0.5, 0.5]], dtype=np.float64), 2
    )

    assert patches.shape == (1, 2, 2, 3)
    assert np.array_equal(patches[..., 0], patches[..., 1])
    assert np.array_equal(patches[..., 1], patches[..., 2])


def test_resident_mip_atlas_matches_random_access_patch_sampling() -> None:
    values = np.arange(64, dtype=np.uint8).reshape(8, 8, 1)
    slot = {
        "channels": {"RGB": "mask"},
        "transfer": "identity-linear",
    }
    domain = NativeAssetDomain(
        "slot-1",
        "surface-uv",
        "clamp",
        _mip_shapes(8, 8),
        (NativeAssetRole("mask", "mask", 0, 3, "identity-linear", "box"),),
    )
    collection = _lazy_collection(values, slot, domain)
    levels = collection._decode_mip_pyramid(0, 1, domain)
    counts = tuple(level.shape[0] * level.shape[1] for level in levels)
    offsets = np.cumsum((0, *counts[:-1]), dtype=np.int64)
    pyramid = _ResidentMipPyramid(
        torch.cat(
            tuple(
                torch.from_numpy(level.transpose(2, 0, 1).copy()).reshape(3, -1)
                for level in levels
            ),
            dim=1,
        ),
        torch.tensor([level.shape[:2] for level in levels], dtype=torch.int64),
        torch.from_numpy(offsets),
    )
    uv = torch.tensor([[0.5, 0.5], [0.99, 0.01]], dtype=torch.float32)
    requested = torch.tensor([0, 2], dtype=torch.int64)
    actual = collection._sample_gpu_pyramid(
        pyramid, uv, requested, 2, domain.address_mode
    )
    expected = np.concatenate(
        tuple(
            collection._sample_mip_patches(
                0,
                1,
                int(level),
                uv[row : row + 1].numpy().astype(np.float64),
                2,
            ).transpose(0, 3, 1, 2)
            for row, level in enumerate(requested)
        ),
        axis=0,
    )
    torch.testing.assert_close(actual, torch.from_numpy(expected))
