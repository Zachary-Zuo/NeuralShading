from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol

import numpy as np
from PIL import Image
import pyexr
import torch

from ncls.core.identity import sha256_json
from ncls.core.material import MAX_INTERFACES, MAX_MEDIA, HomogeneousMedium, LayerStackIR, pack_layer_interface
from ncls.core.material.abi_layout import INTERFACE_STRUCT


@dataclass(frozen=True)
class NativeFeatureField:
    name: str
    channels: int
    filter_rule: str

    def __post_init__(self) -> None:
        if not self.name or self.channels < 1 or not self.filter_rule:
            raise ValueError("native feature field identity is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "channels": self.channels,
            "filter_rule": self.filter_rule,
        }


@dataclass(frozen=True)
class NativeFeatureLayout:
    family_id: str
    source_contract_version: int
    fields: tuple[NativeFeatureField, ...]
    spatial: bool

    def __post_init__(self) -> None:
        if not self.family_id or self.source_contract_version < 1 or not self.fields:
            raise ValueError("native feature layout identity is invalid")
        if len({field.name for field in self.fields}) != len(self.fields):
            raise ValueError("native feature layout fields must be unique")
        object.__setattr__(self, "fields", tuple(self.fields))

    @property
    def channel_count(self) -> int:
        return sum(field.channels for field in self.fields)

    @property
    def layout_id(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "source_contract_version": self.source_contract_version,
            "fields": [field.to_dict() for field in self.fields],
            "spatial": self.spatial,
        }


class NativeFeaturePyramid(Protocol):
    """method source adaptation 的 tile 化输入，避免展开整张 K-channel 纹理。"""

    feature_count: int
    level_shapes: tuple[tuple[int, int], ...]

    def iter_level_tiles(
        self,
        level: int,
        max_texels: int,
        device: torch.device,
    ) -> Iterator[tuple[int, torch.Tensor]]: ...


@dataclass(frozen=True)
class DenseNativeFeaturePyramid:
    levels: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if not self.levels or any(level.ndim != 3 for level in self.levels):
            raise ValueError("dense native feature levels must have shape [height,width,feature]")
        feature_count = int(self.levels[0].shape[2])
        if feature_count < 1 or any(int(level.shape[2]) != feature_count for level in self.levels):
            raise ValueError("dense native feature pyramid channels must agree")
        if any(not bool(torch.isfinite(level).all()) for level in self.levels):
            raise ValueError("dense native feature pyramid must be finite")
        object.__setattr__(self, "levels", tuple(self.levels))

    @property
    def feature_count(self) -> int:
        return int(self.levels[0].shape[2])

    @property
    def level_shapes(self) -> tuple[tuple[int, int], ...]:
        return tuple((int(level.shape[0]), int(level.shape[1])) for level in self.levels)

    def iter_level_tiles(
        self,
        level: int,
        max_texels: int,
        device: torch.device,
    ) -> Iterator[tuple[int, torch.Tensor]]:
        if max_texels < 1:
            raise ValueError("native feature materialization tile must be positive")
        flat = self.levels[level].reshape(-1, self.feature_count)
        for offset in range(0, len(flat), max_texels):
            yield offset, flat[offset : offset + max_texels].to(device=device, dtype=torch.float32)


def layer_stack_native_feature_layout() -> NativeFeatureLayout:
    fields = [
        NativeFeatureField("interface-counts", 2, "constant"),
    ]
    for index in range(MAX_INTERFACES):
        fields.append(NativeFeatureField(f"interface-{index}", 19, "constant"))
    for index in range(MAX_MEDIA):
        fields.append(NativeFeatureField(f"medium-{index}", 9, "constant"))
    return NativeFeatureLayout("ncls.layer-stack@1", 1, tuple(fields), False)


def materialx_native_feature_layout() -> NativeFeatureLayout:
    return NativeFeatureLayout(
        "materialx.document@1.39.4",
        1,
        (
            NativeFeatureField("resolved-standard-surface-inputs", 24, "constant"),
            NativeFeatureField("base-color-linear", 3, "srgb-decode-then-box-mip"),
            NativeFeatureField("specular-roughness", 1, "box-mip"),
            NativeFeatureField("metalness", 1, "box-mip"),
            NativeFeatureField("normal-first-moment", 3, "lean-box-mip"),
            NativeFeatureField("normal-second-moment", 6, "lean-box-mip"),
        ),
        True,
    )


def _downsample_features(values: np.ndarray) -> np.ndarray:
    height, width = values.shape[:2]
    if height == 1 and width == 1:
        return values
    source = values.astype(np.float32, copy=False)
    if height == 1:
        source = np.concatenate((source, source), axis=0)
    elif height % 2:
        source = source[:-1]
    if width == 1:
        source = np.concatenate((source, source), axis=1)
    elif width % 2:
        source = source[:, :-1]
    result = 0.25 * (
        source[0::2, 0::2] + source[1::2, 0::2]
        + source[0::2, 1::2] + source[1::2, 1::2]
    )
    return result.astype(np.float16)


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    return np.where(
        values <= 0.04045,
        values / 12.92,
        np.power((values + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


@dataclass(frozen=True)
class MaterialXNativeFeaturePyramid:
    """MaterialX spatial fields的紧凑 filtered pyramid；常量不按 texel 重复存储。"""

    constants: np.ndarray
    spatial_levels: tuple[np.ndarray, ...]
    layout_id: str
    _torch_cache: dict[str, tuple[torch.Tensor, tuple[torch.Tensor, ...]]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        constants = np.asarray(self.constants, dtype=np.float32)
        if constants.shape != (24,) or not np.isfinite(constants).all():
            raise ValueError("MaterialX resolved constants must contain 24 finite values")
        levels = tuple(np.asarray(level, dtype=np.float16) for level in self.spatial_levels)
        if not levels or any(level.ndim != 3 or level.shape[2] != 14 for level in levels):
            raise ValueError("MaterialX spatial feature mips must have 14 channels")
        if any(not np.isfinite(level).all() for level in levels):
            raise ValueError("MaterialX spatial feature mips must be finite")
        layout = materialx_native_feature_layout()
        if self.layout_id != layout.layout_id:
            raise ValueError("MaterialX native feature layout identity mismatch")
        object.__setattr__(self, "constants", constants)
        object.__setattr__(self, "spatial_levels", levels)

    @classmethod
    def from_textures(
        cls,
        constants: np.ndarray,
        *,
        base_color: Path | None,
        roughness: Path | None,
        metalness: Path | None,
        normal: Path | None,
    ) -> "MaterialXNativeFeaturePyramid":
        inputs = np.asarray(constants, dtype=np.float32)
        loaded: dict[str, np.ndarray | None] = {
            "base": None if base_color is None else (
                np.asarray(Image.open(base_color).convert("RGB"), dtype=np.float32) / 255.0
            ),
            "roughness": None if roughness is None else pyexr.read(str(roughness)).astype(np.float32)[..., :1],
            "metalness": None if metalness is None else pyexr.read(str(metalness)).astype(np.float32)[..., :1],
            "normal": None if normal is None else pyexr.read(str(normal)).astype(np.float32)[..., :3],
        }
        extents = {value.shape[:2] for value in loaded.values() if value is not None}
        if len(extents) > 1:
            raise ValueError("MaterialX training textures must share one native extent")
        height, width = next(iter(extents), (1, 1))
        spatial = np.empty((height, width, 14), dtype=np.float16)
        if loaded["base"] is None:
            spatial[..., 0:3] = inputs[1:4]
        else:
            spatial[..., 0:3] = _srgb_to_linear(loaded["base"])
        spatial[..., 3] = inputs[12] if loaded["roughness"] is None else loaded["roughness"][..., 0]
        spatial[..., 4] = inputs[5] if loaded["metalness"] is None else loaded["metalness"][..., 0]
        if loaded["normal"] is None:
            normal_values = np.zeros((height, width, 3), dtype=np.float32)
            normal_values[..., 2] = 1.0
        else:
            normal_values = loaded["normal"] * 2.0 - 1.0
            normal_values[..., :2] *= float(inputs[17])
            lengths = np.linalg.norm(normal_values, axis=2, keepdims=True)
            normal_values /= np.maximum(lengths, 1e-12)
        spatial[..., 5:8] = normal_values
        spatial[..., 8] = normal_values[..., 0] * normal_values[..., 0]
        spatial[..., 9] = normal_values[..., 0] * normal_values[..., 1]
        spatial[..., 10] = normal_values[..., 0] * normal_values[..., 2]
        spatial[..., 11] = normal_values[..., 1] * normal_values[..., 1]
        spatial[..., 12] = normal_values[..., 1] * normal_values[..., 2]
        spatial[..., 13] = normal_values[..., 2] * normal_values[..., 2]
        levels = [spatial]
        while levels[-1].shape[0] > 1 or levels[-1].shape[1] > 1:
            levels.append(_downsample_features(levels[-1]))
        return cls(inputs, tuple(levels), materialx_native_feature_layout().layout_id)

    @property
    def feature_count(self) -> int:
        return 38

    @property
    def level_shapes(self) -> tuple[tuple[int, int], ...]:
        return tuple((int(level.shape[0]), int(level.shape[1])) for level in self.spatial_levels)

    def iter_level_tiles(
        self,
        level: int,
        max_texels: int,
        device: torch.device,
    ) -> Iterator[tuple[int, torch.Tensor]]:
        if max_texels < 1:
            raise ValueError("native feature materialization tile must be positive")
        constants, levels = self._device_data(device)
        spatial = levels[level].reshape(-1, 14)
        for offset in range(0, len(spatial), max_texels):
            tile = spatial[offset : offset + max_texels].to(dtype=torch.float32)
            yield offset, torch.cat((constants.expand(len(tile), -1), tile), dim=1)

    def _device_data(
        self, device: torch.device
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        key = str(device)
        cached = self._torch_cache.get(key)
        if cached is None:
            cached = (
                torch.as_tensor(self.constants, dtype=torch.float32, device=device),
                tuple(
                    torch.as_tensor(level, dtype=torch.float16, device=device)
                    for level in self.spatial_levels
                ),
            )
            self._torch_cache[key] = cached
        return cached

    @staticmethod
    def _bilinear_wrap(level: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        height, width = int(level.shape[0]), int(level.shape[1])
        wrapped = torch.remainder(uv, 1.0)
        x = wrapped[:, 0] * width - 0.5
        y = wrapped[:, 1] * height - 0.5
        x0, y0 = torch.floor(x).long(), torch.floor(y).long()
        tx, ty = (x - x0).unsqueeze(1), (y - y0).unsqueeze(1)
        x0w, x1w = torch.remainder(x0, width), torch.remainder(x0 + 1, width)
        y0w, y1w = torch.remainder(y0, height), torch.remainder(y0 + 1, height)
        v00 = level[y0w, x0w].to(dtype=torch.float32)
        v10 = level[y0w, x1w].to(dtype=torch.float32)
        v01 = level[y1w, x0w].to(dtype=torch.float32)
        v11 = level[y1w, x1w].to(dtype=torch.float32)
        return (
            (v00 * (1.0 - tx) + v10 * tx) * (1.0 - ty)
            + (v01 * (1.0 - tx) + v11 * tx) * ty
        )

    def sample_torch(
        self, uv: torch.Tensor, mip_level: torch.Tensor
    ) -> torch.Tensor:
        if uv.ndim != 2 or uv.shape[1] != 2 or mip_level.shape != (len(uv),):
            raise ValueError("MaterialX native feature samples require aligned uv and mip level")
        constants, levels = self._device_data(uv.device)
        selected = torch.clamp(
            torch.round(mip_level).long(), 0, len(levels) - 1
        )
        result = torch.empty(
            (len(uv), self.feature_count), dtype=torch.float32, device=uv.device
        )
        result[:, :24] = constants
        for index, level in enumerate(levels):
            mask = selected == index
            result[mask, 24:] = self._bilinear_wrap(level, uv[mask])
        return result

    def sample(self, uv: np.ndarray, mip_level: np.ndarray) -> np.ndarray:
        coordinates = np.asarray(uv, dtype=np.float32)
        selected = np.clip(
            np.rint(np.asarray(mip_level, dtype=np.float32)).astype(np.int64),
            0,
            len(self.spatial_levels) - 1,
        )
        if coordinates.ndim != 2 or coordinates.shape[1] != 2 or selected.shape != (len(coordinates),):
            raise ValueError("MaterialX native feature samples require aligned uv and mip level")
        result = np.empty((len(coordinates), self.feature_count), dtype=np.float32)
        result[:, :24] = self.constants
        for index, level in enumerate(self.spatial_levels):
            mask = selected == index
            if not np.any(mask):
                continue
            height, width = level.shape[:2]
            wrapped = np.remainder(coordinates[mask], 1.0)
            x = wrapped[:, 0] * width - 0.5
            y = wrapped[:, 1] * height - 0.5
            x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
            tx, ty = (x - x0)[:, None], (y - y0)[:, None]
            x0w, x1w = np.remainder(x0, width), np.remainder(x0 + 1, width)
            y0w, y1w = np.remainder(y0, height), np.remainder(y0 + 1, height)
            values = level.astype(np.float32, copy=False)
            sampled = (
                (values[y0w, x0w] * (1.0 - tx) + values[y0w, x1w] * tx) * (1.0 - ty)
                + (values[y1w, x0w] * (1.0 - tx) + values[y1w, x1w] * tx) * ty
            )
            result[mask, 24:] = sampled
        return result


def _medium_values(medium: HomogeneousMedium) -> tuple[float, ...]:
    return (*medium.sigma_a, *medium.sigma_s, medium.g, medium.thickness)


def encode_layer_stack_native_features(stack: LayerStackIR) -> np.ndarray:
    """把 LayerStack 原生 tagged records 编成有 mask/one-hot 的 1×1 encoder 输入。"""

    values: list[float] = [
        len(stack.interfaces) / float(MAX_INTERFACES),
        len(stack.media) / float(MAX_MEDIA),
    ]
    for index in range(MAX_INTERFACES):
        if index >= len(stack.interfaces):
            values.extend([0.0] * 19)
            continue
        record = INTERFACE_STRUCT.unpack(pack_layer_interface(stack.interfaces[index]))
        kind = int(record[0])
        one_hot = [1.0 if kind == candidate else 0.0 for candidate in range(4)]
        values.extend((1.0, *one_hot, *map(float, record[2:])))
    for index in range(MAX_MEDIA):
        if index >= len(stack.media):
            values.extend([0.0] * 9)
            continue
        values.extend((1.0, *_medium_values(stack.media[index])))
    result = np.asarray(values, dtype=np.float32)
    layout = layer_stack_native_feature_layout()
    if result.shape != (layout.channel_count,) or not np.isfinite(result).all():
        raise AssertionError("LayerStack native feature layout disagrees with encoder payload")
    return result
