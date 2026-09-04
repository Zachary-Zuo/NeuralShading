from __future__ import annotations

import struct
from typing import Any, Mapping, Sequence

import numpy as np


RGBA16F_DDS_DTYPE = "texture2d-rgba16float-dds@1"
RGBA8_SNORM_DDS_DTYPE = "texture2d-rgba8-snorm-dds@1"
_DDS_MAGIC = b"DDS "
_DX10_FOURCC = b"DX10"
_DXGI_FORMAT_R16G16B16A16_FLOAT = 10
_DXGI_FORMAT_R8G8B8A8_SNORM = 31
_D3D10_RESOURCE_DIMENSION_TEXTURE2D = 3


def _encode_rgba_dds(
    levels: Sequence[np.ndarray],
    *,
    dtype: np.dtype[Any],
    bytes_per_texel: int,
    dxgi_format: int,
    label: str,
) -> bytes:
    if not levels:
        raise ValueError(f"{label} DDS requires at least one mip level")
    normalized = tuple(np.asarray(level, dtype=dtype) for level in levels)
    height, width = normalized[0].shape[:2]
    if height < 1 or width < 1:
        raise ValueError(f"{label} DDS base extent must be positive")
    expected_height, expected_width = height, width
    for level in normalized:
        if level.shape != (expected_height, expected_width, 4):
            raise ValueError(f"{label} DDS mip extents/channels are inconsistent")
        if np.issubdtype(dtype, np.floating) and not np.isfinite(level).all():
            raise ValueError(f"{label} DDS texels must be finite")
        expected_width = max(1, expected_width // 2)
        expected_height = max(1, expected_height // 2)

    header = bytearray(148)
    header[:4] = _DDS_MAGIC
    flags = 0x1 | 0x2 | 0x4 | 0x8 | 0x20000
    struct.pack_into(
        "<7I", header, 4,
        124, flags, height, width, width * bytes_per_texel, 0, len(normalized),
    )
    struct.pack_into("<II4s", header, 76, 32, 0x4, _DX10_FOURCC)
    caps = 0x1000 | (0x8 | 0x400000 if len(normalized) > 1 else 0)
    struct.pack_into("<I", header, 108, caps)
    struct.pack_into(
        "<5I", header, 128,
        dxgi_format,
        _D3D10_RESOURCE_DIMENSION_TEXTURE2D,
        0,
        1,
        0,
    )
    return bytes(header) + b"".join(
        np.ascontiguousarray(level).tobytes() for level in normalized
    )


def encode_rgba16f_dds(levels: Sequence[np.ndarray]) -> bytes:
    return _encode_rgba_dds(
        levels,
        dtype=np.dtype("<f2"),
        bytes_per_texel=8,
        dxgi_format=_DXGI_FORMAT_R16G16B16A16_FLOAT,
        label="RGBA16F",
    )


def encode_rgba8_snorm_dds(levels: Sequence[np.ndarray]) -> bytes:
    return _encode_rgba_dds(
        levels,
        dtype=np.dtype(np.int8),
        bytes_per_texel=4,
        dxgi_format=_DXGI_FORMAT_R8G8B8A8_SNORM,
        label="RGBA8 SNORM",
    )


def _inspect_rgba_dds(
    payload: bytes,
    *,
    bytes_per_texel: int,
    expected_dxgi_format: int,
    label: str,
) -> tuple[int, int, int]:
    if len(payload) < 148 or payload[:4] != _DDS_MAGIC:
        raise ValueError("typed texture is not a DDS file")
    size, _, height, width, pitch, depth, mip_count = struct.unpack_from("<7I", payload, 4)
    pixel_size, pixel_flags, fourcc = struct.unpack_from("<II4s", payload, 76)
    dxgi_format, dimension, misc, array_size, misc2 = struct.unpack_from(
        "<5I", payload, 128
    )
    if (
        size != 124 or pixel_size != 32 or pixel_flags != 0x4 or fourcc != _DX10_FOURCC
        or dxgi_format != expected_dxgi_format
        or dimension != _D3D10_RESOURCE_DIMENSION_TEXTURE2D
        or misc != 0 or array_size != 1 or misc2 != 0 or depth != 0
        or width < 1 or height < 1 or mip_count < 1
        or pitch != width * bytes_per_texel
    ):
        raise ValueError(f"{label} DDS header disagrees with the typed texture contract")
    expected_bytes = 148
    mip_width, mip_height = width, height
    for _ in range(mip_count):
        expected_bytes += mip_width * mip_height * bytes_per_texel
        mip_width, mip_height = max(1, mip_width // 2), max(1, mip_height // 2)
    if len(payload) != expected_bytes:
        raise ValueError(f"{label} DDS payload length disagrees with its mip chain")
    return width, height, mip_count


def inspect_rgba16f_dds(payload: bytes) -> tuple[int, int, int]:
    return _inspect_rgba_dds(
        payload,
        bytes_per_texel=8,
        expected_dxgi_format=_DXGI_FORMAT_R16G16B16A16_FLOAT,
        label="RGBA16F",
    )


def inspect_rgba8_snorm_dds(payload: bytes) -> tuple[int, int, int]:
    return _inspect_rgba_dds(
        payload,
        bytes_per_texel=4,
        expected_dxgi_format=_DXGI_FORMAT_R8G8B8A8_SNORM,
        label="RGBA8 SNORM",
    )


def validate_typed_resource(payload: bytes, descriptor: Mapping[str, Any]) -> None:
    dtype = descriptor.get("dtype")
    if dtype == RGBA16F_DDS_DTYPE:
        width, height, mip_count = inspect_rgba16f_dds(payload)
        stride = 8
        label = "RGBA16F"
    elif dtype == RGBA8_SNORM_DDS_DTYPE:
        width, height, mip_count = inspect_rgba8_snorm_dds(payload)
        stride = 4
        label = "RGBA8 SNORM"
    else:
        raise ValueError(
            f"unsupported ScatteringPackage typed resource dtype {dtype!r}"
        )
    if descriptor.get("shape") != [width, height, mip_count, 4]:
        raise ValueError(f"{label} DDS descriptor shape disagrees with file contents")
    if (
        int(descriptor.get("stride", 0)) != stride
        or int(descriptor.get("alignment", 0)) < 16
    ):
        raise ValueError(f"{label} DDS descriptor stride/alignment is invalid")
    usage = descriptor.get("usage")
    if not isinstance(usage, str) or not usage:
        raise ValueError(f"{label} DDS descriptor requires a shader usage")


__all__ = [
    "RGBA16F_DDS_DTYPE",
    "RGBA8_SNORM_DDS_DTYPE",
    "encode_rgba16f_dds",
    "encode_rgba8_snorm_dds",
    "inspect_rgba16f_dds",
    "inspect_rgba8_snorm_dds",
    "validate_typed_resource",
]
