from __future__ import annotations

import numpy as np
import pytest

from ncls.bundle.typed_texture import (
    RGBA16F_DDS_DTYPE,
    RGBA8_SNORM_DDS_DTYPE,
    encode_rgba16f_dds,
    encode_rgba8_snorm_dds,
    inspect_rgba16f_dds,
    inspect_rgba8_snorm_dds,
    validate_typed_resource,
)


def test_rgba16f_dds_roundtrip_and_tamper_matrix() -> None:
    levels = (
        np.arange(4 * 8 * 4, dtype=np.float32).reshape(4, 8, 4) / 32.0,
        np.ones((2, 4, 4), dtype=np.float32),
        np.zeros((1, 2, 4), dtype=np.float32),
    )
    payload = encode_rgba16f_dds(levels)
    assert inspect_rgba16f_dds(payload) == (8, 4, 3)
    descriptor = {
        "dtype": RGBA16F_DDS_DTYPE,
        "shape": [8, 4, 3, 4],
        "stride": 8,
        "alignment": 16,
        "usage": "gTexture",
    }
    validate_typed_resource(payload, descriptor)
    for changed in (
        {**descriptor, "shape": [4, 8, 3, 4]},
        {**descriptor, "stride": 4},
        {**descriptor, "usage": ""},
    ):
        with pytest.raises(ValueError):
            validate_typed_resource(payload, changed)
    with pytest.raises(ValueError):
        inspect_rgba16f_dds(payload[:-1])


def test_rgba16f_dds_rejects_noncanonical_mip_extents() -> None:
    with pytest.raises(ValueError, match="mip"):
        encode_rgba16f_dds((
            np.zeros((4, 4, 4), dtype=np.float32),
            np.zeros((3, 2, 4), dtype=np.float32),
        ))


def test_rgba8_snorm_dds_roundtrip_and_validation() -> None:
    levels = (
        np.arange(8 * 4 * 4, dtype=np.int8).reshape(4, 8, 4),
        np.full((2, 4, 4), -37, dtype=np.int8),
        np.full((1, 2, 4), 91, dtype=np.int8),
    )
    payload = encode_rgba8_snorm_dds(levels)
    assert inspect_rgba8_snorm_dds(payload) == (8, 4, 3)
    validate_typed_resource(
        payload,
        {
            "dtype": RGBA8_SNORM_DDS_DTYPE,
            "shape": [8, 4, 3, 4],
            "stride": 4,
            "alignment": 16,
            "usage": "gDetail",
        },
    )
    with pytest.raises(ValueError, match="payload length"):
        inspect_rgba8_snorm_dds(payload[:-1])
