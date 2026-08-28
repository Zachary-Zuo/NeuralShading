from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from ncls.core.identity import sha256_file, sha256_json
from ncls.core.material import LayerStackIR, MaterialProgram, canonicalize_layer_stack
from ncls.core.source import SourceSnapshot
from ncls.learning.source_adaptation import (
    DenseNativeFeaturePyramid,
    MaterialXNativeFeaturePyramid,
    NativeFeaturePyramid,
    encode_layer_stack_native_features,
    layer_stack_native_feature_layout,
    materialx_native_feature_layout,
)


class MethodSourceAdapter(ABC):
    method_key: str
    family_id: str
    source_contract_version: int
    adapter_id: str
    implementation_sha256: str

    def __init__(
        self, snapshots: Sequence[SourceSnapshot], device: torch.device
    ) -> None:
        values = tuple(snapshots)
        if not values:
            raise ValueError("method source adapter requires source snapshots")
        if any(
            snapshot.family_id != self.family_id
            or snapshot.source_contract_version != self.source_contract_version
            for snapshot in values
        ):
            raise ValueError("method source adapter received an incompatible snapshot")
        self.snapshots = values
        self.device = device

    @property
    def identity(self) -> str:
        return sha256_json(
            {
                "adapter_id": self.adapter_id,
                "implementation_sha256": self.implementation_sha256,
                "source_snapshot_ids": [value.snapshot_id for value in self.snapshots],
            }
        )

    @abstractmethod
    def sample_tensors(
        self,
        source_index: torch.Tensor,
        generator: torch.Generator,
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def materialization_features(self) -> NativeFeaturePyramid:
        raise NotImplementedError


class NvidiaLayerStackSourceAdapter(MethodSourceAdapter):
    method_key = "nvidia-neural-appearance"
    family_id = "ncls.layer-stack@1"
    source_contract_version = 1
    adapter_id = "nvidia.layer-stack-native@2"
    implementation_sha256 = sha256_file(Path(__file__))

    def __init__(
        self, snapshots: Sequence[SourceSnapshot], device: torch.device
    ) -> None:
        super().__init__(snapshots, device)
        stacks = []
        for snapshot in self.snapshots:
            if isinstance(snapshot.native_object, LayerStackIR):
                stacks.append(snapshot.native_object)
            else:
                stacks.append(
                    canonicalize_layer_stack(
                        MaterialProgram.from_json(snapshot.native_payload.decode("utf-8"))
                    )
                )
        values = np.stack([encode_layer_stack_native_features(stack) for stack in stacks])
        self._feature_table = torch.as_tensor(
            values, dtype=torch.float32, device=device
        )
        self._layout_id = layer_stack_native_feature_layout().layout_id

    def sample_tensors(
        self,
        source_index: torch.Tensor,
        generator: torch.Generator,
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, str]]:
        del generator
        count = int(source_index.shape[0])
        return (
            {
                "uv": torch.zeros((count, 2), dtype=torch.float32, device=self.device),
                "uv_dx": torch.zeros((count, 2), dtype=torch.float32, device=self.device),
                "uv_dy": torch.zeros((count, 2), dtype=torch.float32, device=self.device),
                "mip_level": torch.zeros(count, dtype=torch.float32, device=self.device),
                "native_features": self._feature_table[source_index],
            },
            {"native_feature_layout_id": self._layout_id},
        )

    def materialization_features(self) -> NativeFeaturePyramid:
        if len(self.snapshots) != 1:
            raise RuntimeError("NVIDIA materialization trains one source snapshot per run")
        return DenseNativeFeaturePyramid((self._feature_table[0:1, None, :].cpu(),))


class NvidiaMaterialXSourceAdapter(MethodSourceAdapter):
    method_key = "nvidia-neural-appearance"
    family_id = "materialx.document@1.39.4"
    source_contract_version = 1
    adapter_id = "nvidia.materialx-standard-surface-spatial@2"
    implementation_sha256 = sha256_file(Path(__file__))

    def __init__(
        self, snapshots: Sequence[SourceSnapshot], device: torch.device
    ) -> None:
        super().__init__(snapshots, device)
        if len(self.snapshots) != 1:
            raise RuntimeError("NVIDIA materialization trains one source snapshot per run")
        snapshot = self.snapshots[0]
        inputs = snapshot.editor_metadata.get("resolved_inputs")
        paths = snapshot.editor_metadata.get("resource_paths")
        if not isinstance(inputs, bytes) or not isinstance(paths, Mapping):
            raise ValueError("MaterialX snapshot is missing canonical runtime bindings")
        constants = np.frombuffer(inputs, dtype=np.float32).copy()
        self._pyramid = MaterialXNativeFeaturePyramid.from_textures(
            constants,
            base_color=_path(paths.get("base-color")),
            roughness=_path(paths.get("roughness")),
            metalness=_path(paths.get("metalness")),
            normal=_path(paths.get("normal")),
        )
        self._layout_id = materialx_native_feature_layout().layout_id

    def sample_tensors(
        self,
        source_index: torch.Tensor,
        generator: torch.Generator,
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, str]]:
        count = int(source_index.shape[0])
        uv = torch.rand((count, 2), generator=generator, device=self.device)
        maximum_mip = len(self._pyramid.level_shapes) - 1
        exponential = -torch.log(
            torch.clamp(
                1.0 - torch.rand(count, generator=generator, device=self.device),
                min=1e-7,
            )
        )
        mip_level = torch.clamp(exponential, max=float(maximum_mip))
        texel_extent = max(self._pyramid.level_shapes[0])
        footprint = torch.pow(2.0, mip_level) / float(texel_extent)
        uv_dx = torch.stack((footprint, torch.zeros_like(footprint)), dim=1)
        uv_dy = torch.stack((torch.zeros_like(footprint), footprint), dim=1)
        return (
            {
                "uv": uv,
                "uv_dx": uv_dx,
                "uv_dy": uv_dy,
                "mip_level": mip_level,
                "native_features": self._pyramid.sample_torch(uv, mip_level),
            },
            {"native_feature_layout_id": self._layout_id},
        )

    def materialization_features(self) -> NativeFeaturePyramid:
        return self._pyramid


def _path(value: object) -> Path | None:
    return None if value is None else Path(str(value)).resolve()


_ADAPTERS: dict[
    tuple[str, str, int],
    Callable[[Sequence[SourceSnapshot], torch.device], MethodSourceAdapter],
] = {
    (
        NvidiaLayerStackSourceAdapter.method_key,
        NvidiaLayerStackSourceAdapter.family_id,
        NvidiaLayerStackSourceAdapter.source_contract_version,
    ): NvidiaLayerStackSourceAdapter,
    (
        NvidiaMaterialXSourceAdapter.method_key,
        NvidiaMaterialXSourceAdapter.family_id,
        NvidiaMaterialXSourceAdapter.source_contract_version,
    ): NvidiaMaterialXSourceAdapter,
}


def create_method_source_adapter(
    method_key: str,
    snapshots: Sequence[SourceSnapshot],
    device: torch.device,
) -> MethodSourceAdapter:
    values = tuple(snapshots)
    if not values:
        raise ValueError("method source adapter requires snapshots")
    source_keys = {
        (snapshot.family_id, snapshot.source_contract_version) for snapshot in values
    }
    if len(source_keys) != 1:
        raise ValueError("one online producer cannot mix source contracts")
    family_id, version = next(iter(source_keys))
    try:
        factory = _ADAPTERS[(method_key, family_id, version)]
    except KeyError as error:
        raise ValueError(
            f"method {method_key!r} has no source adaptation for {family_id}@{version}"
        ) from error
    return factory(values, device)


__all__ = [
    "MethodSourceAdapter",
    "create_method_source_adapter",
]
