from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch
from ncls.core.identity import sha256_file
from ncls.core.material import LayerStackIR, MaterialProgram, canonicalize_layer_stack
from ncls.core.source import SourceSnapshot
from ncls.learning.source_adaptation import DenseNativeAssetCollection, MaterialXNativeAssetCollection, NativeAssetCollection, NativeAssetRole, encode_layer_stack_native_features, encode_mdl_fixed_native_features, layer_stack_native_feature_layout, materialx_native_feature_layout, mdl_fixed_native_feature_layout
from ncls.source_materials.mdl import MdlMaterialSource

from ncls.learning.source_adapters import MethodSourceAdapter


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
        self._native_assets = DenseNativeAssetCollection(
            tuple(
                (torch.from_numpy(values[index : index + 1, None, :]).clone(),)
                for index in range(len(self.snapshots))
            ),
            tuple(snapshot.snapshot_id for snapshot in self.snapshots),
            self._layout_id,
            "constant",
            "constant",
            "clamp",
            (
                NativeAssetRole(
                    "encoder-input",
                    "layer-stack-native-records",
                    0,
                    self._feature_table.shape[1],
                    "linear",
                    "constant",
                ),
            ),
        )

    def sample_tensors(
        self,
        source_index: torch.Tensor,
        generator: torch.Generator,
        options: Mapping[str, Any],
        *,
        execution_source_indices: Sequence[int] | None = None,
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, str]]:
        del generator, options, execution_source_indices
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

    def native_assets(self) -> NativeAssetCollection:
        return self._native_assets


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
        self._assets = MaterialXNativeAssetCollection.from_textures(
            constants,
            base_color=_path(paths.get("base-color")),
            roughness=_path(paths.get("roughness")),
            metalness=_path(paths.get("metalness")),
            normal=_path(paths.get("normal")),
            asset_id=snapshot.snapshot_id,
        )
        self._layout_id = materialx_native_feature_layout().layout_id

    def sample_tensors(
        self,
        source_index: torch.Tensor,
        generator: torch.Generator,
        options: Mapping[str, Any],
        *,
        execution_source_indices: Sequence[int] | None = None,
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, str]]:
        del options, execution_source_indices
        count = int(source_index.shape[0])
        uv = torch.rand((count, 2), generator=generator, device=self.device)
        level_shapes = self._assets.descriptors[0].domain("surface-uv").level_shapes
        maximum_mip = len(level_shapes) - 1
        exponential = -torch.log(
            torch.clamp(
                1.0 - torch.rand(count, generator=generator, device=self.device),
                min=1e-7,
            )
        )
        mip_level = torch.clamp(exponential, max=float(maximum_mip))
        texel_extent = max(level_shapes[0])
        footprint = torch.pow(2.0, mip_level) / float(texel_extent)
        uv_dx = torch.stack((footprint, torch.zeros_like(footprint)), dim=1)
        uv_dy = torch.stack((torch.zeros_like(footprint), footprint), dim=1)
        return (
            {
                "uv": uv,
                "uv_dx": uv_dx,
                "uv_dy": uv_dy,
                "mip_level": mip_level,
                "native_features": self._assets.sample_torch(uv, mip_level),
            },
            {"native_feature_layout_id": self._layout_id},
        )

    def native_assets(self) -> NativeAssetCollection:
        return self._assets


class NvidiaMdlFixedSourceAdapter(MethodSourceAdapter):
    method_key = "nvidia-neural-appearance"
    family_id = "mdl.program@1"
    source_contract_version = 1
    adapter_id = "nvidia.mdl-fixed-uniform@1"
    implementation_sha256 = sha256_file(Path(__file__))

    def __init__(
        self, snapshots: Sequence[SourceSnapshot], device: torch.device
    ) -> None:
        super().__init__(snapshots, device)
        if len(self.snapshots) != 1:
            raise RuntimeError("NVIDIA MDL fixed-uniform training requires one snapshot")
        from ncls.source_materials.mdl import MdlMaterialSource

        source = MdlMaterialSource.from_snapshot(self.snapshots[0])
        values, schema_identity = encode_mdl_fixed_native_features(source.arguments)
        self._feature_table = torch.as_tensor(
            values[None, :], dtype=torch.float32, device=device
        )
        self._layout_id = mdl_fixed_native_feature_layout().layout_id
        self._schema_identity = schema_identity
        self._native_assets = DenseNativeAssetCollection(
            ((torch.from_numpy(values[None, None, :]).clone(),),),
            (self.snapshots[0].snapshot_id,),
            self._layout_id,
            "constant",
            "constant",
            "clamp",
            (
                NativeAssetRole(
                    "encoder-input",
                    "mdl-typed-parameters",
                    0,
                    self._feature_table.shape[1],
                    "signed-bounded",
                    "constant",
                ),
            ),
        )

    def sample_tensors(
        self,
        source_index: torch.Tensor,
        generator: torch.Generator,
        options: Mapping[str, Any],
        *,
        execution_source_indices: Sequence[int] | None = None,
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, str]]:
        del generator, options, execution_source_indices
        if bool((source_index != 0).any()):
            raise ValueError("MDL fixed-uniform adapter accepts only source index zero")
        count = int(source_index.shape[0])
        zeros = torch.zeros((count, 2), dtype=torch.float32, device=self.device)
        return (
            {
                "uv": zeros,
                "uv_dx": zeros.clone(),
                "uv_dy": zeros.clone(),
                "mip_level": torch.zeros(
                    count, dtype=torch.float32, device=self.device
                ),
                "native_features": self._feature_table.expand(count, -1),
            },
            {
                "native_feature_layout_id": self._layout_id,
                "mdl_parameter_schema_identity": self._schema_identity,
            },
        )

    def native_assets(self) -> NativeAssetCollection:
        return self._native_assets


def _path(value: object) -> Path | None:
    return None if value is None else Path(str(value)).resolve()
