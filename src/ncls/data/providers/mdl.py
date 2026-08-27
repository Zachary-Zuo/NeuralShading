from __future__ import annotations

from dataclasses import dataclass
import gc
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from ncls.core.identity import sha256_file, sha256_json
from ncls.data.collector import CollectionConfig
from ncls.data.contract import (
    EvaluatedBlock,
    PositionKind,
    QueryPlan,
    ReferenceDescriptor,
    SourceState,
    SurfaceSample,
)
from ncls.data.falcor import create_falcor_device, direction_rows, import_falcor, output_buffer, structured_buffer
from ncls.paths import PROJECT_ROOT
from ncls.references.mdl import MDL_SDK_DIRECTORY, MdlCompiledArtifact, MdlSdkCompilerBridge
from ncls.source_materials.mdl import snapshot_from_mdl_artifact

from .base import BaseProvider, assign_group_splits, implementation_hash


@dataclass(frozen=True)
class MdlAssetSpec:
    asset_id: str
    module: str
    material: str
    arguments: Mapping[str, Any] | None = None
    pack_id: str = "project.fixtures"
    pack_version: str = "1"

    def __post_init__(self) -> None:
        if not self.asset_id or not self.module.startswith("::") or not self.material:
            raise ValueError("MDL asset identity is incomplete")
        object.__setattr__(self, "arguments", dict(self.arguments or {}))


@dataclass(frozen=True)
class MdlProviderConfig:
    module_root: Path = PROJECT_ROOT / "tests/fixtures/mdl"
    assets: tuple[MdlAssetSpec, ...] = (
        MdlAssetSpec("constant-diffuse", "::constant_diffuse", "constant_diffuse"),
    )
    sdk_root: Path = PROJECT_ROOT / "external" / MDL_SDK_DIRECTORY
    bridge_executable: Path = (
        PROJECT_ROOT / "build/mdl-sdk-bridge/Release/ncls_mdl_sdk_bridge.exe"
    )
    cache_root: Path = PROJECT_ROOT / "build/mdl-reference/cache"
    asset_manifest: Path | None = None

    def __post_init__(self) -> None:
        if not self.assets or len({item.asset_id for item in self.assets}) != len(self.assets):
            raise ValueError("MDL provider assets must be nonempty and unique")

    @classmethod
    def from_vmaterials2(cls, asset_ids: Sequence[str]) -> "MdlProviderConfig":
        manifest_path = PROJECT_ROOT / "references/mdl-vmaterials2-v1/assets.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_name") != "ncls.mdl-vmaterials-assets"
            or manifest.get("schema_version") != 1
            or manifest.get("pack_id") != "nvidia.vmaterials@2.4.0"
        ):
            raise ValueError("unsupported vMaterials asset manifest")
        by_id = {str(item["asset_id"]): item for item in manifest.get("assets", [])}
        requested = tuple(map(str, asset_ids))
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("vMaterials asset IDs must be nonempty and unique")
        missing = sorted(set(requested) - set(by_id))
        if missing:
            raise ValueError(f"unknown vMaterials asset IDs: {missing}")
        specs = tuple(
            MdlAssetSpec(
                asset_id,
                str(by_id[asset_id]["module"]),
                str(by_id[asset_id]["export"]),
                pack_id="nvidia.vmaterials",
                pack_version="2.4.0",
            )
            for asset_id in requested
        )
        return cls(
            module_root=(
                PROJECT_ROOT
                / "assets/source-materials/mdl-vmaterials2/2.4.0"
                / str(manifest["module_root"])
            ),
            assets=specs,
            asset_manifest=manifest_path,
        )


@dataclass(frozen=True)
class _MdlRuntimeState:
    artifact: MdlCompiledArtifact


@dataclass
class _MdlGpuQuerySlot:
    views: object
    lights: object
    positions: object
    uv: object
    gradients: object
    output: object
    output_tensor: torch.Tensor


class MdlGpuQueryRuntime:
    """当前 Falcor 8 上的唯一正式 MDL GPU executor。"""

    def __init__(
        self,
        artifact: MdlCompiledArtifact,
        *,
        sdk_root: Path,
        query_capacity: int | None = None,
        slot_count: int = 2,
    ) -> None:
        if any(
            texture.get("shape") not in {"2d", "bsdf_data"}
            for texture in artifact.manifest.get("textures", [])
        ):
            raise ValueError("MDL runtime accepts 2D and SDK BSDF-data textures only")
        if len(artifact.manifest.get("ro_data", [])) > 1:
            raise ValueError("MDL V1 supports at most one read-only data segment")
        if query_capacity is not None and (query_capacity < 1 or slot_count < 2):
            raise ValueError("MDL shared query runtime requires positive capacity and at least two slots")
        self.artifact = artifact
        self.sdk_root = sdk_root.resolve()
        self.query_capacity = None if query_capacity is None else int(query_capacity)
        self.slot_count = int(slot_count)
        self._falcor = None
        self._device = None
        self._compute = None
        self._textures: list[tuple[int, str, Any]] = []
        self._argument_data = None
        self._ro_data = None
        self._sampler = None
        self._slots: tuple[_MdlGpuQuerySlot, ...] = ()

    def _texture_2d(self, descriptor: Mapping[str, Any]):
        falcor, device = self._falcor, self._device
        if falcor is None or device is None:
            raise RuntimeError("MDL GPU device is not initialized")
        width = int(descriptor["width"])
        height = int(descriptor["height"])
        data = descriptor.get("data")
        if data is not None:
            payload = (self.artifact.root / str(data)).read_bytes()
            pixel_type = str(descriptor.get("pixel_type", ""))
            if pixel_type == "Sint8":
                encoded = np.frombuffer(payload, dtype=np.uint8).reshape(height, width).copy()
                is_scalar = True
            elif pixel_type in {"Rgb", "Rgba"}:
                channels = 3 if pixel_type == "Rgb" else 4
                source = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, channels)
                if channels == 3:
                    encoded = np.empty((height, width, 4), dtype=np.uint8)
                    encoded[..., :3] = source
                    encoded[..., 3] = 255
                else:
                    encoded = source.copy()
                is_scalar = False
            else:
                raise ValueError(f"unsupported MDL decoded texture pixel type: {pixel_type}")
            data_origin = descriptor.get("data_origin")
            if data_origin == "lower_left":
                # Falcor's file loader stores scanlines top-to-bottom.  Keep
                # that GPU layout so the renderer's MDL lower-left V
                # conversion remains identical for every decoded resource.
                encoded = encoded[::-1].copy()
            elif data_origin != "top_left":
                raise ValueError("MDL decoded texture has an unsupported row origin")
        else:
            with Image.open(str(descriptor["path"])) as source_image:
                is_scalar = source_image.mode in {"1", "L", "I", "I;16", "F"}
                encoded = np.array(
                    source_image.convert("L" if is_scalar else "RGBA"),
                    dtype=np.uint8,
                    copy=True,
                )
        if is_scalar:
            resource_format = falcor.ResourceFormat.R8Unorm
        else:
            resource_format = (
                falcor.ResourceFormat.RGBA8UnormSrgb
                if descriptor.get("gamma") == "srgb"
                else falcor.ResourceFormat.RGBA8Unorm
            )
        if encoded.shape[:2] != (height, width):
            raise ValueError("MDL 2D texture dimensions differ from the SDK artifact")
        texture = device.create_texture(
            width=encoded.shape[1],
            height=encoded.shape[0],
            format=resource_format,
            mip_levels=1,
            bind_flags=falcor.ResourceBindFlags.ShaderResource,
        )
        texture.from_numpy(np.ascontiguousarray(encoded))
        return texture

    def _texture_3d(self, descriptor: Mapping[str, Any]):
        falcor, device = self._falcor, self._device
        if falcor is None or device is None:
            raise RuntimeError("MDL GPU device is not initialized")
        width = int(descriptor["width"])
        height = int(descriptor["height"])
        depth = int(descriptor["depth"])
        data_path = self.artifact.root / str(descriptor["data"])
        values = np.frombuffer(data_path.read_bytes(), dtype=np.float32)
        if values.size != width * height * depth:
            raise ValueError("MDL BSDF-data texture payload has the wrong size")
        scalar = values.reshape(depth, height, width).copy()
        texture = device.create_texture(
            width=width,
            height=height,
            depth=depth,
            format=falcor.ResourceFormat.R32Float,
            mip_levels=1,
            bind_flags=falcor.ResourceBindFlags.ShaderResource,
        )
        texture.from_numpy(np.ascontiguousarray(scalar))
        return texture

    def _shared_buffer(self, *, writable: bool = False):
        if self.query_capacity is None or self._falcor is None or self._device is None:
            raise RuntimeError("MDL shared query runtime is not initialized")
        flags = self._falcor.ResourceBindFlags.ShaderResource | self._falcor.ResourceBindFlags.Shared
        if writable:
            flags |= self._falcor.ResourceBindFlags.UnorderedAccess
        return self._device.create_structured_buffer(
            struct_size=16,
            element_count=self.query_capacity,
            bind_flags=flags,
        )

    def _slot(self) -> _MdlGpuQuerySlot:
        if self._falcor is None:
            raise RuntimeError("MDL shared query runtime is not initialized")
        views = self._shared_buffer()
        lights = self._shared_buffer()
        positions = self._shared_buffer()
        uv = self._shared_buffer()
        gradients = self._shared_buffer()
        output = self._shared_buffer(writable=True)
        return _MdlGpuQuerySlot(
            views,
            lights,
            positions,
            uv,
            gradients,
            output,
            output.to_torch([self.query_capacity, 4], self._falcor.float32),
        )

    @staticmethod
    def _query_input(
        values: torch.Tensor | np.ndarray,
        count: int,
        channels: int,
        device: torch.device,
    ) -> torch.Tensor:
        source = torch.as_tensor(values, dtype=torch.float32, device=device)
        if source.ndim != 2 or source.shape != (count, channels):
            raise ValueError("MDL flat query input has the wrong shape")
        result = torch.zeros((count, 4), dtype=torch.float32, device=device)
        result[:, :channels].copy_(source)
        return result

    def _argument_rows(self) -> np.ndarray:
        argument_bytes = self.artifact.argument_block
        padded_size = max(16, ((len(argument_bytes) + 15) // 16) * 16)
        padded_arguments = argument_bytes + bytes(padded_size - len(argument_bytes))
        return (
            np.frombuffer(padded_arguments, dtype=np.uint32)
            .view(np.float32)
            .reshape(-1, 4)
            .copy()
        )

    def _bind_static_resources(self, compute: Any) -> None:
        if self._argument_data is None:
            raise RuntimeError("MDL argument block is not initialized")
        compute.globals.gMdlArgumentBlock = self._argument_data
        if self._ro_data is not None:
            compute.globals.gMdlRoData = self._ro_data
        if self._textures:
            for index, shape, texture in self._textures:
                dimension = "2D" if shape == "2d" else "3D"
                compute.globals[f"gMdlTexture{dimension}{index - 1}"] = texture
            compute.globals.gMdlTextureSampler = self._sampler

    def _runtime(self):
        if self._compute is None:
            self._falcor = import_falcor()
            self._device = create_falcor_device(self._falcor)
            types = (
                self.sdk_root
                / "examples/mdl_sdk/dxr/content/mdl_target_code_types.hlsl"
            ).read_text(encoding="utf-8")
            renderer = (
                PROJECT_ROOT / "shaders/ncls/reference_backends/mdl_runtime.slangh"
            ).read_text(encoding="utf-8")
            generated_source = "\n".join(
                (
                    "#define MDL_NUM_TEXTURE_RESULTS 16",
                    "#define MDL_DF_HANDLE_SLOT_MODE -1",
                    "struct NclsMdlRendererState { float3 view_direction; };",
                    "#define RENDERER_STATE_TYPE NclsMdlRendererState",
                    f"#define NCLS_MDL_TEXTURE_COUNT {max(1, len(self.artifact.manifest.get('textures', [])))}",
                    types,
                    renderer,
                    self.artifact.hlsl,
                )
            )
            desc = self._falcor.ProgramDesc()
            desc.add_shader_module("NclsMdlGenerated").add_string(
                generated_source, self.artifact.root / "ncls_mdl_generated.slang"
            )
            query_path = PROJECT_ROOT / "shaders/ncls/reference_backends/mdl_query.slang"
            desc.add_shader_module("NclsMdlQuery").add_string(
                query_path.read_text(encoding="utf-8"), query_path
            )
            desc.cs_entry("main")
            self._compute = self._falcor.ComputePass(self._device, desc)
            self._argument_data = structured_buffer(
                self._device,
                self._falcor,
                self._argument_rows(),
                16,
            )
            texture_descriptors = self.artifact.manifest.get("textures", [])
            self._textures = []
            for descriptor in texture_descriptors:
                shape = str(descriptor["shape"])
                texture = (
                    self._texture_2d(descriptor)
                    if shape == "2d"
                    else self._texture_3d(descriptor)
                )
                self._textures.append((int(descriptor["index"]), shape, texture))
            ro_segments = self.artifact.manifest.get("ro_data", [])
            if ro_segments:
                descriptor = ro_segments[0]
                payload = (self.artifact.root / str(descriptor["path"])).read_bytes()
                padded_size = max(16, ((len(payload) + 15) // 16) * 16)
                rows = (
                    np.frombuffer(payload + bytes(padded_size - len(payload)), dtype=np.uint32)
                    .view(np.float32)
                    .reshape(-1, 4)
                    .copy()
                )
                self._ro_data = structured_buffer(self._device, self._falcor, rows, 16)
            if self._textures:
                self._sampler = self._device.create_sampler(
                    mag_filter=self._falcor.TextureFilteringMode.Linear,
                    min_filter=self._falcor.TextureFilteringMode.Linear,
                    mip_filter=self._falcor.TextureFilteringMode.Linear,
                    max_anisotropy=1,
                    address_mode_u=self._falcor.TextureAddressingMode.Wrap,
                    address_mode_v=self._falcor.TextureAddressingMode.Wrap,
                    address_mode_w=self._falcor.TextureAddressingMode.Wrap,
                )
            if self.query_capacity is not None:
                self._slots = tuple(self._slot() for _ in range(self.slot_count))
        return self._falcor, self._device, self._compute

    def evaluate(
        self,
        surfaces: Sequence[SurfaceSample],
        plan: QueryPlan,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not surfaces:
            raise ValueError("MDL evaluation requires at least one surface sample")
        for surface in surfaces:
            if surface.position_kind not in {PositionKind.CONSTANT, PositionKind.UV}:
                raise ValueError("MDL V1 supports constant/UV surface positions only")
            if surface.geometric_normal != (0.0, 0.0, 1.0) or surface.geometric_tangent != (1.0, 0.0, 0.0):
                raise ValueError("MDL V1 query frame is fixed to +Z normal and +X tangent")
        falcor, device, compute = self._runtime()
        view_rows, light_rows = direction_rows(plan.view_directions, plan.light_directions, len(surfaces))
        repeat_count = len(plan.view_directions) * plan.direction_count
        positions = np.repeat(
            np.asarray([(*surface.position, 0.0) for surface in surfaces], dtype=np.float32),
            repeat_count,
            axis=0,
        )
        uv = np.repeat(
            np.asarray([(*surface.uv, 0.0, 0.0) for surface in surfaces], dtype=np.float32),
            repeat_count,
            axis=0,
        )
        gradients = np.repeat(
            np.asarray([(*surface.uv_dx, *surface.uv_dy) for surface in surfaces], dtype=np.float32),
            repeat_count,
            axis=0,
        )
        output = output_buffer(device, falcor, len(view_rows))
        compute.globals.gViews = structured_buffer(device, falcor, view_rows, 16)
        compute.globals.gLights = structured_buffer(device, falcor, light_rows, 16)
        compute.globals.gPositions = structured_buffer(device, falcor, positions, 16)
        compute.globals.gUv = structured_buffer(device, falcor, uv, 16)
        self._bind_static_resources(compute)
        compute.globals.gOutput = output
        compute.globals.gQueryCount = len(view_rows)
        compute.execute(threads_x=len(view_rows))
        result = output.to_numpy().view(np.float32).reshape(len(view_rows), 4).copy()
        device.end_frame()
        if not np.all(np.isfinite(result)) or np.any(result[:, 3] < 0.0):
            raise RuntimeError("MDL GPU reference produced invalid evaluate/PDF output")
        shape = (len(surfaces), len(plan.view_directions), plan.direction_count)
        return result[:, :3].reshape(*shape, 3), result[:, 3].reshape(shape)

    def evaluate_torch(
        self,
        slot_index: int,
        views: torch.Tensor | np.ndarray,
        lights: torch.Tensor | np.ndarray,
        uv: torch.Tensor | np.ndarray,
        gradients: torch.Tensor | np.ndarray,
        positions: torch.Tensor | np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.query_capacity is None:
            raise RuntimeError("MDL runtime was not created with shared query capacity")
        count = len(views)
        if not 1 <= count <= self.query_capacity or any(
            len(values) != count for values in (lights, uv, gradients)
        ):
            raise ValueError("MDL flat query arrays must be nonempty, aligned, and within capacity")
        _, device, compute = self._runtime()
        if not 0 <= slot_index < len(self._slots):
            raise ValueError("MDL query slot index is out of range")
        slot = self._slots[slot_index]
        tensor_device = slot.output_tensor.device
        position_values: torch.Tensor | np.ndarray = (
            torch.zeros((count, 3), dtype=torch.float32, device=tensor_device)
            if positions is None
            else positions
        )
        slot.views.from_torch(self._query_input(views, count, 3, tensor_device))
        slot.lights.from_torch(self._query_input(lights, count, 3, tensor_device))
        slot.positions.from_torch(self._query_input(position_values, count, 3, tensor_device))
        slot.uv.from_torch(self._query_input(uv, count, 2, tensor_device))
        slot.gradients.from_torch(self._query_input(gradients, count, 4, tensor_device))
        device.render_context.wait_for_cuda()
        self._bind_static_resources(compute)
        compute.globals.gViews = slot.views
        compute.globals.gLights = slot.lights
        compute.globals.gPositions = slot.positions
        compute.globals.gUv = slot.uv
        compute.globals.gOutput = slot.output
        compute.globals.gQueryCount = count
        compute.execute(threads_x=count)
        device.render_context.wait_for_falcor()
        result = slot.output_tensor[:count]
        return result[:, :3], result[:, 3]

    def close(self) -> None:
        self._slots = ()
        self._compute = None
        self._textures = []
        self._argument_data = None
        self._ro_data = None
        self._sampler = None
        self._device = None
        self._falcor = None
        gc.collect()


class MdlProvider(BaseProvider):
    def __init__(
        self,
        collection: CollectionConfig,
        config: MdlProviderConfig = MdlProviderConfig(),
    ) -> None:
        super().__init__(collection)
        self.provider_config = config
        self.descriptor = ReferenceDescriptor(
            "mdl.program@1",
            "ncls.mdl-vmaterials2@1",
            "ncls.mdl-source@1",
            query_contract="mdl-local-frame-f-cos",
            incident_domain="upper-hemisphere",
            position_kind=PositionKind.UV,
            deterministic=True,
            capabilities=("evaluate", "spatial"),
            implementation_sha256=implementation_hash(
                (
                    Path(__file__),
                    PROJECT_ROOT / "src/ncls/references/mdl.py",
                    PROJECT_ROOT / "src/ncls/source_materials/mdl.py",
                    PROJECT_ROOT / "shaders/ncls/reference_backends/mdl_runtime.slangh",
                    PROJECT_ROOT / "shaders/ncls/reference_backends/mdl_query.slang",
                )
            ),
        )
        self._bridge = MdlSdkCompilerBridge(
            config.module_root,
            sdk_root=config.sdk_root,
            executable=config.bridge_executable,
            cache_root=config.cache_root,
        )
        inspection_root = PROJECT_ROOT / "build/mdl-reference/inspection"
        splits = assign_group_splits([asset.asset_id for asset in config.assets], collection.seed)
        states = []
        for asset in config.assets:
            inspection_key = sha256_json(
                {
                    "module": asset.module,
                    "material": asset.material,
                    "arguments": asset.arguments,
                    "module_root": str(config.module_root.resolve()),
                    "bridge_executable_sha256": sha256_file(self._bridge.executable),
                }
            )
            inspection = inspection_root / inspection_key
            artifact = (
                MdlCompiledArtifact.load(inspection)
                if inspection.exists()
                else self._bridge.inspect(
                    asset.module,
                    asset.material,
                    asset.arguments,
                    output=inspection,
                )
            )
            snapshot = snapshot_from_mdl_artifact(
                artifact,
                config.module_root,
                pack_id=asset.pack_id,
                pack_version=asset.pack_version,
            )
            compiled = self._bridge.compile_snapshot(snapshot)
            states.append(
                SourceState(
                    snapshot=snapshot,
                    reference_id=self.descriptor.reference_id,
                    asset_id=asset.asset_id,
                    split_group_id=asset.asset_id,
                    source_uri=f"mdl:{asset.material if asset.material.startswith('::') else asset.module + '::' + asset.material}",
                    split=splits[asset.asset_id],
                    structure_family_id=asset.module,
                    difficulty_class="unclassified",
                    difficulty_tags=(),
                    evaluation_cohort="workflow",
                    runtime_state=_MdlRuntimeState(compiled),
                )
            )
        self._states = tuple(states)
        self._runtimes: dict[str, MdlGpuQueryRuntime] = {}

    def source_states(self) -> Sequence[SourceState]:
        return self._states

    def evaluate(
        self,
        state: SourceState,
        surfaces: Sequence[SurfaceSample],
        plan: QueryPlan,
    ) -> EvaluatedBlock:
        runtime_state: _MdlRuntimeState = state.runtime_state
        key = runtime_state.artifact.artifact_sha256
        runtime = self._runtimes.get(key)
        if runtime is None:
            runtime = MdlGpuQueryRuntime(runtime_state.artifact, sdk_root=self.provider_config.sdk_root)
            self._runtimes[key] = runtime
        response, pdf = runtime.evaluate(surfaces, plan)
        return EvaluatedBlock.deterministic(response, reference_pdf=pdf)

    def metadata(self) -> Mapping[str, Any]:
        return {
            **super().metadata(),
            "material_count": len(self._states),
            "mdl_sdk": "2025.0.0-387700.1252",
            "formal_executor": "Falcor 8.0 / current project pin",
            "falcor2_role": "external validation oracle only",
            "host_readback": "offline EvaluatedBlock boundary only",
            "asset_manifest": (
                None
                if self.provider_config.asset_manifest is None
                else self.provider_config.asset_manifest.relative_to(PROJECT_ROOT).as_posix()
            ),
        }

    def close(self) -> None:
        for runtime in self._runtimes.values():
            runtime.close()
        self._runtimes.clear()
