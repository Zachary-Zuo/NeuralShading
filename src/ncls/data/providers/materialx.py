from __future__ import annotations

from dataclasses import dataclass
import gc
import math
from pathlib import Path
from typing import Sequence
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import pyexr

from ncls.data.collector import CollectionConfig
from ncls.data.contract import EvaluatedBlock, PositionKind, QueryPlan, QueryRole, ReferenceDescriptor, SourceState, SurfaceSample, make_state_id
from ncls.data.directions import MIXTURE_QUERY_PROFILE_ID, peak_grazing_mixture_query
from ncls.source_materials import MaterialXReference, MaterialXSourceMaterial
from ncls.paths import SOURCE_MATERIAL_ROOT
from ncls.source_materials.identity import materialx_asset_sha256

from .base import BaseProvider, PROJECT_ROOT, assign_group_splits, implementation_hash
from .falcor import direction_rows, import_falcor, output_buffer, structured_buffer


@dataclass(frozen=True)
class MaterialXProviderConfig:
    asset_ids: tuple[str, ...] = ()
    materialx_root: Path = PROJECT_ROOT / "external/MaterialX"
    asset_root: Path = SOURCE_MATERIAL_ROOT / "materialx-polyhaven/v1"
    asset_manifest: Path = PROJECT_ROOT / "references/materialx-polyhaven-v1/assets.json"


@dataclass(frozen=True)
class _MaterialXRuntimeState:
    source: MaterialXSourceMaterial
    inputs: np.ndarray
    base_color: Path | None
    roughness: Path | None
    metalness: Path | None
    normal: Path | None
    displacement: Path | None


def _value3(text: str) -> tuple[float, float, float]:
    values = tuple(float(item.strip()) for item in text.split(","))
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        raise ValueError(f"MaterialX value is not a finite float3: {text!r}")
    return values


def _parse_surface(document_path: Path, source: MaterialXSourceMaterial) -> _MaterialXRuntimeState:
    root = ET.parse(document_path).getroot()
    if root.tag != "materialx" or root.get("version") != "1.38":
        raise ValueError("MaterialX Falcor subset requires a 1.38 source document")
    material = next((node for node in root if node.tag == "surfacematerial"), None)
    if material is None:
        raise ValueError(f"MaterialX document has no surfacematerial: {document_path}")
    shader_binding = next((node for node in material if node.tag == "input" and node.get("name") == "surfaceshader"), None)
    if shader_binding is None or not shader_binding.get("nodename"):
        raise ValueError("MaterialX surfacematerial has no standard_surface binding")
    surface = next(
        (node for node in root if node.tag == "standard_surface" and node.get("name") == shader_binding.get("nodename")),
        None,
    )
    if surface is None:
        raise ValueError("MaterialX surface binding does not resolve to standard_surface")
    inputs_by_name = {node.get("name"): node for node in surface if node.tag == "input"}
    values = np.zeros(24, dtype=np.float32)
    values[0], values[1:4], values[8], values[9:12], values[12], values[14] = (
        1.0, (0.8, 0.8, 0.8), 1.0, (1.0, 1.0, 1.0), 0.2, 1.5,
    )
    values[17], values[20:23], values[23] = 1.0, (1.0, 1.0, 1.0), 1.0

    def scalar(name: str, fallback: float) -> float:
        node = inputs_by_name.get(name)
        if node is None:
            return fallback
        if node.get("value") is None or node.get("nodegraph") or node.get("nodename"):
            raise ValueError(f"MaterialX Falcor subset requires constant {name}")
        result = float(node.get("value", ""))
        if not math.isfinite(result):
            raise ValueError(f"MaterialX input {name} is non-finite")
        return result

    def color(name: str, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
        node = inputs_by_name.get(name)
        if node is None:
            return fallback
        if node.get("value") is None or node.get("nodegraph") or node.get("nodename"):
            raise ValueError(f"MaterialX Falcor subset requires constant {name}")
        return _value3(node.get("value", ""))

    values[0] = scalar("base", float(values[0]))
    values[4] = scalar("diffuse_roughness", 0.0)
    values[8] = scalar("specular", float(values[8]))
    values[9:12] = color("specular_color", (1.0, 1.0, 1.0))
    values[14] = scalar("specular_IOR", float(values[14]))
    values[15] = scalar("specular_anisotropy", 0.0)
    values[16] = scalar("specular_rotation", 0.0)
    values[19] = scalar("emission", 0.0)
    values[20:23] = color("emission_color", (1.0, 1.0, 1.0))
    opacity = inputs_by_name.get("opacity")
    if opacity is not None:
        if opacity.get("value") is None or opacity.get("nodegraph") or opacity.get("nodename"):
            raise ValueError("MaterialX Falcor subset requires constant opacity")
        if opacity.get("type") == "color3":
            opacity_rgb = _value3(opacity.get("value", ""))
            if max(opacity_rgb) - min(opacity_rgb) > 1e-8:
                raise ValueError("MaterialX Falcor subset requires achromatic opacity")
            values[23] = opacity_rgb[0]
        else:
            values[23] = float(opacity.get("value", ""))
    for name in (
        "transmission", "transmission_scatter_anisotropy", "transmission_dispersion",
        "transmission_extra_roughness", "subsurface", "subsurface_anisotropy", "sheen",
        "coat", "thin_film_thickness",
    ):
        if abs(scalar(name, 0.0)) > 1e-8:
            raise ValueError(f"MaterialX Falcor surface-response subset does not support nonzero {name}")
    thin_walled = inputs_by_name.get("thin_walled")
    if thin_walled is not None and thin_walled.get("value", "false") != "false":
        raise ValueError("MaterialX Falcor surface-response subset requires thin_walled=false")

    graphs = {node.get("name"): node for node in root if node.tag == "nodegraph"}

    def connected_node(input_node: ET.Element, expected: str) -> ET.Element:
        graph_name, output_name = input_node.get("nodegraph"), input_node.get("output")
        if not graph_name or not output_name or graph_name not in graphs:
            raise ValueError(f"MaterialX input {input_node.get('name')} has an invalid graph output")
        graph = graphs[graph_name]
        output = next((node for node in graph if node.tag == "output" and node.get("name") == output_name), None)
        if output is None or not output.get("nodename"):
            raise ValueError("MaterialX graph output has no node")
        result = next(
            (node for node in graph if node.tag == expected and node.get("name") == output.get("nodename")), None
        )
        if result is None:
            raise ValueError(f"MaterialX graph output does not resolve to {expected}")
        return result

    def image_path(
        image: ET.Element,
        expected_type: str,
        expected_color_space: str | None = None,
    ) -> Path:
        if image.get("type") != expected_type:
            raise ValueError(f"MaterialX image {image.get('name')} has the wrong type")
        file_node = next((node for node in image if node.tag == "input" and node.get("name") == "file"), None)
        if file_node is None or file_node.get("type") != "filename" or not file_node.get("value"):
            raise ValueError("MaterialX image has no filename")
        color_space = file_node.get("colorspace", "")
        if expected_color_space is not None and color_space != expected_color_space:
            raise ValueError(f"MaterialX image {image.get('name')} has the wrong colorspace")
        if expected_color_space is None and color_space:
            raise ValueError(f"MaterialX raw image {image.get('name')} has an unexpected colorspace")
        texcoord = next((node for node in image if node.tag == "input" and node.get("name") == "texcoord"), None)
        if texcoord is None or texcoord.get("type") != "vector2" or not texcoord.get("nodename"):
            raise ValueError(f"MaterialX image {image.get('name')} must use an explicit texcoord node")
        path = (document_path.parent / file_node.get("value", "")).resolve()
        path.relative_to(document_path.parent.resolve())
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    textures: dict[str, Path | None] = {"base_color": None, "roughness": None, "metalness": None, "normal": None}
    for name, value_offset, flag_offset in (("base_color", 1, 6), ("specular_roughness", 12, 13), ("metalness", 5, 7)):
        node = inputs_by_name.get(name)
        if node is None:
            raise ValueError(f"MaterialX standard_surface has no {name}")
        if node.get("value") is not None:
            if name == "base_color":
                values[value_offset : value_offset + 3] = _value3(node.get("value", ""))
            else:
                values[value_offset] = float(node.get("value", ""))
        else:
            key = "roughness" if name == "specular_roughness" else name
            image = connected_node(node, "image")
            textures[key] = image_path(
                image,
                "color3" if name == "base_color" else "float",
                "srgb_texture" if name == "base_color" else None,
            )
            values[flag_offset] = 1.0
    normal_input = inputs_by_name.get("normal")
    if normal_input is not None and normal_input.get("nodegraph"):
        normal_map = connected_node(normal_input, "normalmap")
        graph = graphs[normal_input.get("nodegraph")]
        normal_in = next((node for node in normal_map if node.tag == "input" and node.get("name") == "in"), None)
        if normal_in is None or not normal_in.get("nodename"):
            raise ValueError("MaterialX normalmap has no image input")
        image = next((node for node in graph if node.tag == "image" and node.get("name") == normal_in.get("nodename")), None)
        if image is None:
            raise ValueError("MaterialX normalmap input does not resolve to image")
        textures["normal"] = image_path(image, "vector3")
        scale = next((node for node in normal_map if node.tag == "input" and node.get("name") == "scale"), None)
        values[17] = float(scale.get("value", "1")) if scale is not None else 1.0
        values[18] = 1.0
    displacement_path = None
    displacement_binding = next(
        (node for node in material if node.tag == "input" and node.get("name") == "displacementshader"),
        None,
    )
    if displacement_binding is not None and displacement_binding.get("nodename"):
        displacement = next(
            (node for node in root if node.tag == "displacement" and node.get("name") == displacement_binding.get("nodename")),
            None,
        )
        if displacement is None:
            raise ValueError("MaterialX displacement binding does not resolve to displacement")
        displacement_input = next(
            (node for node in displacement if node.tag == "input" and node.get("name") == "displacement"),
            None,
        )
        if displacement_input is None:
            raise ValueError("MaterialX displacement shader has no displacement input")
        displacement_path = image_path(connected_node(displacement_input, "image"), "float")
    if not float(values[14]) > 0.0 or not np.all(np.isfinite(values)):
        raise ValueError("MaterialX source material has invalid standard_surface inputs")
    return _MaterialXRuntimeState(
        source,
        values,
        textures["base_color"],
        textures["roughness"],
        textures["metalness"],
        textures["normal"],
        displacement_path,
    )


def _downsample(values: np.ndarray) -> np.ndarray:
    height, width = values.shape[:2]
    if height == 1 and width == 1:
        return values
    if height == 1:
        values = np.concatenate((values, values), axis=0)
    elif height % 2:
        values = values[:-1, ...]
    if width == 1:
        values = np.concatenate((values, values), axis=1)
    elif width % 2:
        values = values[:, :-1, ...]
    return 0.25 * (
        values[0::2, 0::2] + values[1::2, 0::2] + values[0::2, 1::2] + values[1::2, 1::2]
    )


def _box_mips(values: np.ndarray) -> list[np.ndarray]:
    result = [np.asarray(values, dtype=np.float32)]
    while result[-1].shape[0] > 1 or result[-1].shape[1] > 1:
        result.append(_downsample(result[-1]))
    return result


class MaterialXProvider(BaseProvider):
    def __init__(self, collection: CollectionConfig, config: MaterialXProviderConfig = MaterialXProviderConfig()) -> None:
        super().__init__(collection)
        self.provider_config = config
        shader = PROJECT_ROOT / "shaders/ncls/data/reference_materialx.cs.slang"
        self.descriptor = ReferenceDescriptor(
            "materialx.textured-surface@1",
            "ncls.materialx-polyhaven@1",
            "ncls.materialx-source-material@1",
            query_profile_id="ncls.materialx-local-normal-peak@1",
            incident_domain="upper-hemisphere",
            position_kind=PositionKind.UV,
            deterministic=True,
            capabilities=("evaluate", "spatial", "uv-footprint", "normal-map"),
            implementation_sha256=implementation_hash((
                Path(__file__),
                PROJECT_ROOT / "src/ncls/source_materials/materialx.py",
                shader,
                PROJECT_ROOT / "shaders/ncls/reference/materialx_standard_surface_reference.slang",
            )),
        )
        reference = MaterialXReference(config.materialx_root, config.asset_root, config.asset_manifest)
        selected = config.asset_ids or reference.catalog.asset_ids
        missing = sorted(set(selected) - set(reference.catalog.asset_ids))
        if missing:
            raise ValueError(f"unknown MaterialX asset IDs: {missing}")
        splits = assign_group_splits(selected, collection.seed)
        states = []
        for asset_id in selected:
            loaded = reference.load(asset_id, verify_files=True)
            runtime = _parse_surface(loaded.document_path, loaded.material)
            payload = loaded.material.to_json().encode("utf-8")
            source_hash = materialx_asset_sha256(
                loaded.document_path,
                (runtime.base_color, runtime.roughness, runtime.metalness, runtime.normal, runtime.displacement),
            )
            states.append(SourceState(
                make_state_id(self.descriptor.family_id, self.descriptor.native_schema_id, payload, source_hash),
                self.descriptor.family_id,
                self.descriptor.reference_id,
                asset_id,
                asset_id,
                self.descriptor.native_schema_id,
                payload,
                loaded.material.document_uri,
                source_hash,
                splits[asset_id],
                runtime,
            ))
        self._states = tuple(states)
        self._falcor = None
        self._device = None
        self._compute = None
        self._normal_compute = None

    def source_states(self) -> Sequence[SourceState]:
        return self._states

    def _runtime(self):
        if self._compute is None:
            self._falcor = import_falcor()
            self._device = self._falcor.Device(type=self._falcor.DeviceType.D3D12)
            self._compute = self._falcor.ComputePass(
                self._device,
                file=PROJECT_ROOT / "shaders/ncls/data/reference_materialx.cs.slang",
                cs_entry="evaluateReference",
            )
            self._normal_compute = self._falcor.ComputePass(
                self._device,
                file=PROJECT_ROOT / "shaders/ncls/data/reference_materialx.cs.slang",
                cs_entry="resolveReferenceNormal",
            )
        return self._falcor, self._device, self._compute

    def _resolved_shading_normals(
        self,
        state: SourceState,
        surfaces: Sequence[SurfaceSample],
    ) -> np.ndarray:
        falcor, device, _ = self._runtime()
        runtime: _MaterialXRuntimeState = state.runtime_state
        normal_texture = self._texture(runtime.normal, "normal")
        compute = self._normal_compute
        if compute is None:
            raise RuntimeError("MaterialX normal resolver was not initialized")
        uv = np.asarray([surface.uv for surface in surfaces], dtype=np.float32)
        gradients = np.asarray(
            [(*surface.uv_dx, *surface.uv_dy) for surface in surfaces], dtype=np.float32
        )
        compute.globals.gInputs = structured_buffer(device, falcor, runtime.inputs, 4)
        compute.globals.gNormalMap = normal_texture
        compute.globals.gMaterialSampler = device.create_sampler(max_anisotropy=16)
        compute.globals.gUv = structured_buffer(device, falcor, np.pad(uv, ((0, 0), (0, 2))), 16)
        compute.globals.gUvGrad = structured_buffer(device, falcor, gradients, 16)
        output = output_buffer(device, falcor, len(surfaces))
        compute.globals.gOutput = output
        compute.globals.gQueryCount = len(surfaces)
        compute.execute(threads_x=len(surfaces))
        normals = output.to_numpy().view(np.float32).reshape(len(surfaces), 4)[:, :3].copy()
        del normal_texture
        gc.collect()
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        if not np.all(np.isfinite(normals)) or np.any(lengths <= 0.0):
            raise RuntimeError("MaterialX normal resolver produced an invalid direction")
        return normals / lengths

    def query_plan(
        self,
        state: SourceState,
        surfaces: Sequence[SurfaceSample] = (),
    ) -> QueryPlan:
        base = super().query_plan(state, surfaces)
        if self.config.query_profile_id != MIXTURE_QUERY_PROFILE_ID or not surfaces:
            return base
        normals = self._resolved_shading_normals(state, surfaces)
        surface_count = len(surfaces)
        lights = np.broadcast_to(base.light_directions[None, ...], (surface_count, *base.light_directions.shape)).copy()
        weights = np.broadcast_to(base.solid_angle_weights[None, ...], (surface_count, *base.solid_angle_weights.shape)).copy()
        pdf = np.broadcast_to(base.proposal_pdf[None, ...], (surface_count, *base.proposal_pdf.shape)).copy()
        mixture_roles = np.isin(
            base.query_roles,
            (int(QueryRole.TRAIN), int(QueryRole.ADVERSARIAL_PROBE)),
        )
        mixture_views = base.view_directions[mixture_roles]
        for surface_index, normal in enumerate(normals):
            oriented_normals = np.broadcast_to(normal, mixture_views.shape).copy()
            facing = np.sum(oriented_normals * mixture_views, axis=1) < 0.0
            oriented_normals[facing] *= -1.0
            centers = (
                2.0 * np.sum(mixture_views * oriented_normals, axis=1, keepdims=True) * oriented_normals
                - mixture_views
            )
            centers /= np.linalg.norm(centers, axis=1, keepdims=True)
            surface_lights, surface_weights, surface_pdf = peak_grazing_mixture_query(
                mixture_views,
                base.direction_count,
                full_sphere=False,
                seed=base.seed ^ ((surface_index + 1) * 0x85EBCA77),
                reflection_centers=centers,
            )
            lights[surface_index, mixture_roles] = surface_lights
            weights[surface_index, mixture_roles] = surface_weights
            pdf[surface_index, mixture_roles] = surface_pdf
        proposal_ids = tuple(
            value.replace("-peak-grazing-", "-local-normal-peak-grazing-").replace("@2", "@1")
            if mixture_roles[index] else value
            for index, value in enumerate(base.proposal_id)
        )
        return QueryPlan(
            base.view_directions,
            lights,
            weights,
            pdf,
            proposal_ids,
            base.seed,
            base.query_roles,
        )

    def _texture(self, path: Path | None, semantic: str):
        falcor, device, _ = self._runtime()
        if semantic == "base-color":
            if path is None:
                encoded_mips = [np.asarray([[[255, 255, 255, 255]]], dtype=np.uint8)]
            else:
                encoded = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0
                encoded_mips = [np.rint(np.clip(mip, 0.0, 1.0) * 255.0).astype(np.uint8) for mip in _box_mips(encoded)]
            # shared shader performs the single explicit sRGB decode; the resource must stay UNorm.
            texture_format = falcor.ResourceFormat.RGBA8Unorm
            mips = encoded_mips
        elif semantic in {"roughness", "metalness"}:
            values = np.ones((1, 1, 1), dtype=np.float32) if path is None else pyexr.read(str(path)).astype(np.float32)[..., :1]
            mips = _box_mips(values)
            texture_format = falcor.ResourceFormat.R32Float
        elif semantic == "normal":
            values = np.asarray([[[0.5, 0.5, 1.0]]], dtype=np.float32) if path is None else pyexr.read(str(path)).astype(np.float32)[..., :3]
            mips = [np.concatenate((mip, np.ones((*mip.shape[:2], 1), dtype=np.float32)), axis=2).astype(np.float16) for mip in _box_mips(values)]
            texture_format = falcor.ResourceFormat.RGBA16Float
        else:
            raise ValueError(semantic)
        texture = device.create_texture(
            width=mips[0].shape[1], height=mips[0].shape[0], format=texture_format,
            mip_levels=len(mips), bind_flags=falcor.ResourceBindFlags.ShaderResource,
        )
        for level, mip in enumerate(mips):
            texture.from_numpy(np.ascontiguousarray(mip), mip_level=level)
        return texture

    def evaluate(self, state: SourceState, surfaces: Sequence[SurfaceSample], plan: QueryPlan) -> EvaluatedBlock:
        falcor, device, compute = self._runtime()
        runtime: _MaterialXRuntimeState = state.runtime_state
        textures = (
            self._texture(runtime.base_color, "base-color"),
            self._texture(runtime.roughness, "roughness"),
            self._texture(runtime.metalness, "metalness"),
            self._texture(runtime.normal, "normal"),
        )
        compute.globals.gInputs = structured_buffer(device, falcor, runtime.inputs, 4)
        compute.globals.gBaseColor, compute.globals.gRoughness, compute.globals.gMetalness, compute.globals.gNormalMap = textures
        compute.globals.gMaterialSampler = device.create_sampler(max_anisotropy=16)
        view_rows, light_rows = direction_rows(plan.view_directions, plan.light_directions, len(surfaces))
        repeat_count = len(plan.view_directions) * plan.direction_count
        uv = np.repeat(np.asarray([surface.uv for surface in surfaces], dtype=np.float32), repeat_count, axis=0)
        gradients = np.repeat(
            np.asarray([(*surface.uv_dx, *surface.uv_dy) for surface in surfaces], dtype=np.float32),
            repeat_count,
            axis=0,
        )
        compute.globals.gViews = structured_buffer(device, falcor, view_rows, 16)
        compute.globals.gLights = structured_buffer(device, falcor, light_rows, 16)
        compute.globals.gUv = structured_buffer(device, falcor, np.pad(uv, ((0, 0), (0, 2))), 16)
        compute.globals.gUvGrad = structured_buffer(device, falcor, gradients, 16)
        output = output_buffer(device, falcor, len(view_rows))
        compute.globals.gOutput = output
        compute.globals.gQueryCount = len(view_rows)
        compute.execute(threads_x=len(view_rows))
        response = output.to_numpy().view(np.float32).reshape(len(view_rows), 4)[:, :3].reshape(
            len(surfaces), len(plan.view_directions), plan.direction_count, 3
        ).copy()
        del textures
        gc.collect()
        return EvaluatedBlock.deterministic(response)

    def metadata(self):
        return {
            **super().metadata(),
            "asset_manifest": self.provider_config.asset_manifest.relative_to(PROJECT_ROOT).as_posix(),
            "material_count": len(self._states),
            "texture_filter": "native-resolution mip pyramid, trilinear, 16x anisotropic, repeat",
            "proposal_peak_center": "exact filtered MaterialX shading normal, forward-facing per wo",
        }

    def close(self) -> None:
        self._compute = None
        self._normal_compute = None
        self._device = None
        self._falcor = None
        gc.collect()
