from __future__ import annotations

import numpy as np

from ncls.bundle import encode_rgba8_snorm_dds
from tools.viewer.validate_metal_budgeted_package_gpu import _dds_levels


def test_gpu_package_validator_decodes_rgba8_snorm_mips() -> None:
    levels = (
        np.arange(8 * 4 * 4, dtype=np.int8).reshape(4, 8, 4),
        np.arange(4 * 2 * 4, dtype=np.int8).reshape(2, 4, 4),
        np.arange(2 * 1 * 4, dtype=np.int8).reshape(1, 2, 4),
        np.arange(4, dtype=np.int8).reshape(1, 1, 4),
    )
    descriptor = {
        "dtype": "texture2d-rgba8-snorm-dds@1",
        "shape": [8, 4, 4, 4],
    }

    decoded = _dds_levels(encode_rgba8_snorm_dds(levels), descriptor)

    assert len(decoded) == len(levels)
    for actual, expected in zip(decoded, levels):
        assert actual.flags.c_contiguous
        assert actual.flags.writeable
        np.testing.assert_array_equal(actual, expected)
