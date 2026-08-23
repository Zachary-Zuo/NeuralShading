from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import MappingProxyType
from typing import Any, Mapping, Sequence, TypeAlias
import xml.etree.ElementTree as ET

import numpy as np


OPENPBR_SCHEMA = "ncls.openpbr-material"
OPENPBR_SCHEMA_VERSION = 1
ACESCG_TO_LINEAR_SRGB = np.asarray(
    (
        (1.70505099, -0.62179212, -0.08325887),
        (-0.13025642, 1.14080474, -0.01054832),
        (-0.02400336, -0.12896898, 1.15297233),
    ),
    dtype=np.float64,
)
LINEAR_SRGB_TO_ACESCG = np.asarray(
    (
        (0.61309740, 0.33952315, 0.04737945),
        (0.07019372, 0.91635388, 0.01345240),
        (0.02061560, 0.10956978, 0.86981464),
    ),
    dtype=np.float64,
)


@dataclass(frozen=True)
class ParameterSpec:
    value_type: str
    default: float | bool | tuple[float, ...]

    @property
    def width(self) -> int:
        return {"float": 1, "boolean": 1, "vector2": 2, "vector3": 3, "color3": 3}[self.value_type]


def _spec(value_type: str, default: float | bool | tuple[float, ...]) -> ParameterSpec:
    return ParameterSpec(value_type, default)


# 完整镜像 OpenPBR 1.1.1 参数；两个 rotation 字段是 Adobe reference 的显式扩展。
PARAMETERS: Mapping[str, ParameterSpec] = MappingProxyType({
    "base_weight": _spec("float", 1.0),
    "base_color": _spec("color3", (0.8, 0.8, 0.8)),
    "base_diffuse_roughness": _spec("float", 0.0),
    "base_metalness": _spec("float", 0.0),
    "subsurface_weight": _spec("float", 0.0),
    "subsurface_color": _spec("color3", (0.8, 0.8, 0.8)),
    "subsurface_radius": _spec("float", 1.0),
    "subsurface_radius_scale": _spec("color3", (1.0, 0.5, 0.25)),
    "subsurface_scatter_anisotropy": _spec("float", 0.0),
    "specular_weight": _spec("float", 1.0),
    "specular_color": _spec("color3", (1.0, 1.0, 1.0)),
    "specular_roughness": _spec("float", 0.3),
    "specular_roughness_anisotropy": _spec("float", 0.0),
    "specular_ior": _spec("float", 1.5),
    "specular_anisotropy_rotation_cos_sin": _spec("vector2", (1.0, 0.0)),
    "coat_weight": _spec("float", 0.0),
    "coat_color": _spec("color3", (1.0, 1.0, 1.0)),
    "coat_roughness": _spec("float", 0.0),
    "coat_roughness_anisotropy": _spec("float", 0.0),
    "coat_ior": _spec("float", 1.6),
    "coat_darkening": _spec("float", 1.0),
    "coat_anisotropy_rotation_cos_sin": _spec("vector2", (1.0, 0.0)),
    "fuzz_weight": _spec("float", 0.0),
    "fuzz_color": _spec("color3", (1.0, 1.0, 1.0)),
    "fuzz_roughness": _spec("float", 0.5),
    "transmission_weight": _spec("float", 0.0),
    "transmission_color": _spec("color3", (1.0, 1.0, 1.0)),
    "transmission_depth": _spec("float", 0.0),
    "transmission_scatter": _spec("color3", (0.0, 0.0, 0.0)),
    "transmission_scatter_anisotropy": _spec("float", 0.0),
    "transmission_dispersion_scale": _spec("float", 0.0),
    "transmission_dispersion_abbe_number": _spec("float", 20.0),
    "thin_film_weight": _spec("float", 0.0),
    "thin_film_thickness": _spec("float", 0.5),
    "thin_film_ior": _spec("float", 1.4),
    "emission_luminance": _spec("float", 0.0),
    "emission_color": _spec("color3", (1.0, 1.0, 1.0)),
    "geometry_opacity": _spec("float", 1.0),
    "geometry_thin_walled": _spec("boolean", False),
    "geometry_normal": _spec("vector3", (0.0, 0.0, 1.0)),
    "geometry_tangent": _spec("vector3", (1.0, 0.0, 0.0)),
    "geometry_coat_normal": _spec("vector3", (0.0, 0.0, 1.0)),
    "geometry_coat_tangent": _spec("vector3", (1.0, 0.0, 0.0)),
})

OFFICIAL_PARAMETER_NAMES = tuple(
    name for name in PARAMETERS if not name.endswith("_anisotropy_rotation_cos_sin")
)


def _finite_values(name: str, value: Any, width: int) -> float | bool | tuple[float, ...]:
    if isinstance(value, (bool, np.bool_)):
        if width != 1:
            raise ValueError(f"{name} requires {width} numeric components")
        return bool(value)
    if width == 1:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result
    values = tuple(float(item) for item in value)
    if len(values) != width or not all(math.isfinite(item) for item in values):
        raise ValueError(f"{name} requires {width} finite components")
    return values


@dataclass(frozen=True)
class ConstantBinding:
    value: float | bool | tuple[float, ...]
    source: str = field(default="constant", init=False)

    def to_dict(self) -> dict[str, Any]:
        value: Any = list(self.value) if isinstance(self.value, tuple) else self.value
        return {"source": self.source, "value": value}


@dataclass(frozen=True)
class GeometryBinding:
    symbol: str
    source: str = field(default="geometry", init=False)

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "symbol": self.symbol}


@dataclass(frozen=True)
class TextureBinding:
    uri: str
    color_space: str = "raw"
    channels: str = "rgb"
    wrap: str = "repeat"
    filter: str = "bilinear"
    encoding: str = "linear"
    scale: float | tuple[float, ...] = 1.0
    bias: float | tuple[float, ...] = 0.0
    source: str = field(default="texture", init=False)

    def __post_init__(self) -> None:
        if not self.uri:
            raise ValueError("texture URI must be nonempty")
        if self.wrap not in {"repeat", "clamp"} or self.filter not in {"nearest", "bilinear"}:
            raise ValueError("unsupported texture wrap or filter")
        if self.encoding not in {"linear", "srgb", "normal-map"}:
            raise ValueError("unsupported texture encoding")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "uri": self.uri,
            "color_space": self.color_space,
            "channels": self.channels,
            "wrap": self.wrap,
            "filter": self.filter,
            "encoding": self.encoding,
            "scale": list(self.scale) if isinstance(self.scale, tuple) else self.scale,
            "bias": list(self.bias) if isinstance(self.bias, tuple) else self.bias,
        }


@dataclass(frozen=True)
class GraphBinding:
    node: str
    source: str = field(default="graph", init=False)

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "node": self.node}


ParameterBinding: TypeAlias = ConstantBinding | GeometryBinding | TextureBinding | GraphBinding


def _binding_from_dict(value: Mapping[str, Any]) -> ParameterBinding:
    source = str(value.get("source", ""))
    if source == "constant":
        raw = value.get("value")
        return ConstantBinding(tuple(raw) if isinstance(raw, list) else raw)
    if source == "geometry":
        return GeometryBinding(str(value["symbol"]))
    if source == "graph":
        return GraphBinding(str(value["node"]))
    if source == "texture":
        scale = value.get("scale", 1.0)
        bias = value.get("bias", 0.0)
        return TextureBinding(
            str(value["uri"]),
            str(value.get("color_space", "raw")),
            str(value.get("channels", "rgb")),
            str(value.get("wrap", "repeat")),
            str(value.get("filter", "bilinear")),
            str(value.get("encoding", "linear")),
            tuple(scale) if isinstance(scale, list) else float(scale),
            tuple(bias) if isinstance(bias, list) else float(bias),
        )
    raise ValueError(f"unsupported OpenPBR parameter binding {source!r}")


def _default_bindings() -> dict[str, ParameterBinding]:
    result: dict[str, ParameterBinding] = {
        name: ConstantBinding(spec.default) for name, spec in PARAMETERS.items()
    }
    result["geometry_normal"] = GeometryBinding("N")
    result["geometry_tangent"] = GeometryBinding("T")
    result["geometry_coat_normal"] = GeometryBinding("coat_N")
    result["geometry_coat_tangent"] = GeometryBinding("coat_T")
    return result


def _parse_xml_value(spec: ParameterSpec, text: str) -> float | bool | tuple[float, ...]:
    if spec.value_type == "boolean":
        normalized = text.strip().lower()
        if normalized not in {"true", "false"}:
            raise ValueError(f"invalid MaterialX boolean {text!r}")
        return normalized == "true"
    parts = [item.strip() for item in text.split(",")]
    return _finite_values("MaterialX value", parts if spec.width > 1 else parts[0], spec.width)


def _xml_value(value: float | bool | tuple[float, ...]) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, tuple):
        return ", ".join(format(item, ".9g") for item in value)
    return format(value, ".9g")


@dataclass(frozen=True)
class OpenPBRMaterial:
    material_id: str
    parameters: Mapping[str, ParameterBinding]
    color_space: str = "linear-srgb"
    source_document: str | None = None
    authored_parameters: frozenset[str] = frozenset()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.material_id or not self.color_space:
            raise ValueError("OpenPBR material ID and color space must be nonempty")
        if set(self.parameters) != set(PARAMETERS):
            missing = sorted(set(PARAMETERS) - set(self.parameters))
            extra = sorted(set(self.parameters) - set(PARAMETERS))
            raise ValueError(f"OpenPBR parameter set mismatch: missing={missing}, extra={extra}")
        validated: dict[str, ParameterBinding] = {}
        for name, spec in PARAMETERS.items():
            binding = self.parameters[name]
            if isinstance(binding, ConstantBinding):
                binding = ConstantBinding(_finite_values(name, binding.value, spec.width))
            validated[name] = binding
        object.__setattr__(self, "parameters", MappingProxyType(validated))
        object.__setattr__(self, "authored_parameters", frozenset(self.authored_parameters))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def defaults(cls, material_id: str = "openpbr-default") -> OpenPBRMaterial:
        return cls(material_id, _default_bindings())

    @classmethod
    def from_materialx(cls, path: str | Path) -> OpenPBRMaterial:
        document_path = Path(path).resolve()
        root = ET.parse(document_path).getroot()
        surface = next((element for element in root if element.tag == "open_pbr_surface"), None)
        if surface is None:
            raise ValueError(f"MaterialX document has no open_pbr_surface node: {document_path}")
        material_node = next((element for element in root if element.tag == "surfacematerial"), None)
        material_id = material_node.get("name") if material_node is not None else surface.get("name")
        parameters = _default_bindings()
        authored: set[str] = set()
        nodes_by_name = {element.get("name"): element for element in root if element.get("name")}
        for input_element in surface.findall("input"):
            name = input_element.get("name", "")
            if name not in PARAMETERS:
                continue
            authored.add(name)
            if input_element.get("value") is not None:
                parameters[name] = ConstantBinding(_parse_xml_value(PARAMETERS[name], input_element.get("value", "")))
                continue
            node_name = input_element.get("nodename")
            node = nodes_by_name.get(node_name)
            if node is not None and node.tag == "image":
                file_input = next((item for item in node.findall("input") if item.get("name") == "file"), None)
                if file_input is not None and file_input.get("value"):
                    texture_color_space = node.get("colorspace", root.get("colorspace", "raw"))
                    parameters[name] = TextureBinding(
                        file_input.get("value", ""),
                        texture_color_space,
                        "r" if PARAMETERS[name].width == 1 else "rgb",
                        encoding="srgb" if texture_color_space.lower() in {"srgb", "srgb_texture"} else "linear",
                    )
                    continue
            if node_name:
                parameters[name] = GraphBinding(node_name)
        return cls(
            str(material_id or document_path.stem),
            parameters,
            root.get("colorspace", "linear-srgb"),
            str(document_path),
            frozenset(authored),
            {
                "materialx_version": root.get("version", ""),
                "source_sha256": hashlib.sha256(document_path.read_bytes()).hexdigest(),
            },
        )

    def with_parameter(self, name: str, binding: ParameterBinding | float | bool | Sequence[float]) -> OpenPBRMaterial:
        if name not in PARAMETERS:
            raise KeyError(name)
        if not isinstance(binding, (ConstantBinding, GeometryBinding, TextureBinding, GraphBinding)):
            binding = ConstantBinding(tuple(binding) if isinstance(binding, Sequence) and not isinstance(binding, str) else binding)
        parameters = dict(self.parameters)
        parameters[name] = binding
        return replace(self, parameters=parameters, authored_parameters=self.authored_parameters | {name})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": OPENPBR_SCHEMA,
            "schema_version": OPENPBR_SCHEMA_VERSION,
            "material_id": self.material_id,
            "color_space": self.color_space,
            "source_document": self.source_document,
            "authored_parameters": sorted(self.authored_parameters),
            "parameters": {name: binding.to_dict() for name, binding in self.parameters.items()},
            "metadata": dict(self.metadata),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=indent) + "\n"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OpenPBRMaterial:
        if value.get("schema_name") != OPENPBR_SCHEMA or value.get("schema_version") != OPENPBR_SCHEMA_VERSION:
            raise ValueError("unsupported OpenPBR material schema")
        raw_parameters = value.get("parameters")
        if not isinstance(raw_parameters, Mapping):
            raise ValueError("OpenPBR parameters must be an object")
        return cls(
            str(value["material_id"]),
            {str(name): _binding_from_dict(binding) for name, binding in raw_parameters.items()},
            str(value.get("color_space", "linear-srgb")),
            str(value["source_document"]) if value.get("source_document") else None,
            frozenset(str(item) for item in value.get("authored_parameters", [])),
            value.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> OpenPBRMaterial:
        value = json.loads(text)
        if not isinstance(value, Mapping):
            raise ValueError("OpenPBR material JSON root must be an object")
        return cls.from_dict(value)

    def save_materialx(self, path: str | Path) -> None:
        if self.source_document:
            tree = ET.parse(self.source_document)
            root = tree.getroot()
            surface = next((element for element in root if element.tag == "open_pbr_surface"), None)
            if surface is None:
                raise ValueError("source MaterialX document lost its open_pbr_surface node")
        else:
            root = ET.Element("materialx", {"version": "1.39", "colorspace": self.color_space})
            material = ET.SubElement(root, "surfacematerial", {"name": self.material_id, "type": "material"})
            ET.SubElement(material, "input", {"name": "surfaceshader", "type": "surfaceshader", "nodename": "open_pbr_surface"})
            surface = ET.SubElement(root, "open_pbr_surface", {"name": "open_pbr_surface", "type": "surfaceshader"})
            tree = ET.ElementTree(root)
        existing = {item.get("name"): item for item in surface.findall("input")}
        for name in sorted(self.authored_parameters):
            if name not in OFFICIAL_PARAMETER_NAMES:
                continue
            binding = self.parameters[name]
            if not isinstance(binding, ConstantBinding):
                if name in existing:
                    continue
                raise ValueError(f"cannot author new nonconstant MaterialX binding for {name!r}")
            element = existing.get(name)
            if element is None:
                element = ET.SubElement(surface, "input", {"name": name, "type": PARAMETERS[name].value_type})
            element.attrib.pop("nodename", None)
            element.set("value", _xml_value(binding.value))
        ET.indent(tree, space="  ")
        tree.write(Path(path), encoding="utf-8", xml_declaration=True)


class _TextureCache:
    def __init__(self) -> None:
        self._images: dict[Path, np.ndarray] = {}

    def read(self, path: Path) -> np.ndarray:
        path = path.resolve()
        cached = self._images.get(path)
        if cached is not None:
            return cached
        if path.suffix.lower() == ".exr":
            import pyexr

            image = np.asarray(pyexr.read(str(path)), dtype=np.float32)
        else:
            from PIL import Image

            with Image.open(path) as source:
                image = np.asarray(source.convert("RGBA"), dtype=np.float32) / 255.0
        if image.ndim == 2:
            image = image[..., None]
        self._images[path] = image
        return image


def _sample_texture(image: np.ndarray, uv: Sequence[float], binding: TextureBinding) -> np.ndarray:
    coordinate = np.asarray(uv, dtype=np.float64)
    if binding.wrap == "repeat":
        coordinate = coordinate - np.floor(coordinate)
    else:
        coordinate = np.clip(coordinate, 0.0, 1.0)
    x = coordinate[0] * max(image.shape[1] - 1, 0)
    y = (1.0 - coordinate[1]) * max(image.shape[0] - 1, 0)
    if binding.filter == "nearest":
        value = image[int(round(y)), int(round(x))]
    else:
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        x1, y1 = min(x0 + 1, image.shape[1] - 1), min(y0 + 1, image.shape[0] - 1)
        tx, ty = x - x0, y - y0
        value = (
            image[y0, x0] * (1.0 - tx) * (1.0 - ty)
            + image[y0, x1] * tx * (1.0 - ty)
            + image[y1, x0] * (1.0 - tx) * ty
            + image[y1, x1] * tx * ty
        )
    channel_indices = {"r": (0,), "g": (1,), "b": (2,), "a": (3,), "rgb": (0, 1, 2)}
    if binding.channels == "luminance":
        result = np.asarray([np.dot(value[:3], (0.2126, 0.7152, 0.0722))], dtype=np.float64)
    elif binding.channels in channel_indices:
        result = np.asarray(value[list(channel_indices[binding.channels])], dtype=np.float64)
    else:
        raise ValueError(f"unsupported texture channels {binding.channels!r}")
    if binding.encoding == "srgb":
        result = np.where(result <= 0.04045, result / 12.92, ((result + 0.055) / 1.055) ** 2.4)
    elif binding.encoding == "normal-map":
        result = result * 2.0 - 1.0
        result /= max(float(np.linalg.norm(result)), 1e-12)
    scale = np.asarray(binding.scale, dtype=np.float64)
    bias = np.asarray(binding.bias, dtype=np.float64)
    return result * scale + bias


def _normalize(name: str, value: Sequence[float]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    length = float(np.linalg.norm(result))
    if result.shape != (3,) or not np.isfinite(result).all() or length <= 1e-12:
        raise ValueError(f"{name} must be a finite nonzero Float3")
    return result / length


def _basis(normal: Sequence[float], tangent: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = _normalize("geometry normal", normal)
    raw_tangent = np.asarray(tangent, dtype=np.float64)
    t = _normalize("geometry tangent", raw_tangent - np.dot(raw_tangent, n) * n)
    b = _normalize("geometry bitangent", np.cross(n, t))
    return t, b, n


def _convert_texture_color(value: np.ndarray, source: str, target: str) -> np.ndarray:
    aliases = {
        "acescg": "acescg",
        "linear-srgb": "linear-srgb",
        "lin_rec709": "linear-srgb",
        "srgb": "linear-srgb",
        "srgb_texture": "linear-srgb",
    }
    source_key = aliases.get(source.lower())
    target_key = aliases.get(target.lower())
    if source.lower() == "raw" or source_key == target_key:
        return value
    if source_key is None or target_key is None:
        raise ValueError(f"unsupported OpenPBR texture color conversion {source!r} -> {target!r}")
    if source_key == "acescg" and target_key == "linear-srgb":
        return ACESCG_TO_LINEAR_SRGB @ value
    if source_key == "linear-srgb" and target_key == "acescg":
        return LINEAR_SRGB_TO_ACESCG @ value
    raise AssertionError(f"unhandled OpenPBR texture color conversion {source_key} -> {target_key}")


def _resolve_binding(
    name: str,
    binding: ParameterBinding,
    *,
    uv: Sequence[float],
    asset_root: Path,
    geometry: Mapping[str, Sequence[float]],
    texture_cache: _TextureCache,
    target_color_space: str,
) -> float | bool | tuple[float, ...]:
    spec = PARAMETERS[name]
    if isinstance(binding, ConstantBinding):
        return binding.value
    if isinstance(binding, GeometryBinding):
        fallback = spec.default
        return _finite_values(name, geometry.get(binding.symbol, fallback), spec.width)
    if isinstance(binding, GraphBinding):
        raise NotImplementedError(f"OpenPBR graph node {binding.node!r} for {name!r} requires a MaterialX graph evaluator")
    value = _sample_texture(texture_cache.read(asset_root / binding.uri), uv, binding)
    if binding.encoding == "normal-map":
        if name not in {"geometry_normal", "geometry_coat_normal"}:
            raise ValueError(f"normal-map encoding is only valid for OpenPBR normal inputs, not {name!r}")
        prefix = "coat_" if name == "geometry_coat_normal" else ""
        base_normal = geometry.get(f"{prefix}N", PARAMETERS[name].default)
        base_tangent = geometry.get(f"{prefix}T", PARAMETERS[f"geometry_{prefix}tangent"].default)
        tangent, bitangent, normal = _basis(base_normal, base_tangent)
        value = _normalize(name, value[0] * tangent + value[1] * bitangent + value[2] * normal)
    elif spec.value_type == "color3":
        value = _convert_texture_color(value, binding.color_space, target_color_space)
    return _finite_values(name, value if spec.width > 1 else value[0], spec.width)


_FLAT_PARAMETER_ORDER = (
    "base_weight", "base_color", "base_diffuse_roughness", "base_metalness",
    "subsurface_weight", "subsurface_color", "subsurface_radius", "subsurface_radius_scale", "subsurface_scatter_anisotropy",
    "specular_weight", "specular_color", "specular_roughness", "specular_roughness_anisotropy", "specular_ior", "specular_anisotropy_rotation_cos_sin",
    "coat_weight", "coat_color", "coat_roughness", "coat_roughness_anisotropy", "coat_ior", "coat_darkening", "coat_anisotropy_rotation_cos_sin",
    "fuzz_weight", "fuzz_color", "fuzz_roughness",
    "transmission_weight", "transmission_color", "transmission_depth", "transmission_scatter", "transmission_scatter_anisotropy", "transmission_dispersion_scale", "transmission_dispersion_abbe_number",
    "thin_film_weight", "thin_film_thickness", "thin_film_ior",
    "emission_luminance", "emission_color", "geometry_opacity", "geometry_thin_walled",
)


def _flatten_resolved(
    material: OpenPBRMaterial,
    *,
    uv: Sequence[float],
    geometry: Mapping[str, Sequence[float]],
    texture_cache: _TextureCache,
    asset_root: Path,
) -> np.ndarray:
    resolved = {
        name: _resolve_binding(
            name,
            binding,
            uv=uv,
            asset_root=asset_root,
            geometry=geometry,
            texture_cache=texture_cache,
            target_color_space=material.color_space,
        )
        for name, binding in material.parameters.items()
    }
    values: list[float] = []
    for name in _FLAT_PARAMETER_ORDER:
        value = resolved[name]
        if isinstance(value, tuple):
            values.extend(value)
        else:
            values.append(float(value))
    for normal_name, tangent_name in (
        ("geometry_normal", "geometry_tangent"),
        ("geometry_coat_normal", "geometry_coat_tangent"),
    ):
        for vector in _basis(resolved[normal_name], resolved[tangent_name]):  # type: ignore[arg-type]
            values.extend(float(item) for item in vector)
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (77,):
        raise AssertionError(f"OpenPBR probe ABI expected 77 floats, got {result.shape}")
    return result


def resolve_openpbr_inputs(
    material: OpenPBRMaterial,
    *,
    uv: Sequence[float] = (0.0, 0.0),
    geometry: Mapping[str, Sequence[float]] | None = None,
    asset_root: str | Path | None = None,
) -> np.ndarray:
    """把一个空间点的原生 binding 解析成 Adobe reference 使用的 77-float 输入。"""

    root = Path(asset_root) if asset_root is not None else (
        Path(material.source_document).parent if material.source_document else Path.cwd()
    )
    return _flatten_resolved(
        material,
        uv=uv,
        geometry=geometry or {},
        texture_cache=_TextureCache(),
        asset_root=root,
    )


@dataclass(frozen=True)
class OpenPBRReferenceResult:
    response_cos: np.ndarray
    pdf: np.ndarray


@dataclass(frozen=True)
class OpenPBRSampleResult:
    light_direction: np.ndarray
    weight: np.ndarray
    pdf: np.ndarray
    lobe_type: np.ndarray


class OpenPBRReference:
    def __init__(self, executable: str | Path):
        self.executable = Path(executable)
        self._texture_cache = _TextureCache()

    @staticmethod
    def _vectors(name: str, values: Sequence[Sequence[float]]) -> np.ndarray:
        result = np.asarray(values, dtype=np.float32)
        if result.ndim == 1:
            result = result[None, :]
        if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
            raise ValueError(f"{name} must have shape [N, 3]")
        lengths = np.linalg.norm(result, axis=1)
        if np.any(np.abs(lengths - 1.0) > 2e-4):
            raise ValueError(f"{name} must contain normalized directions")
        return result

    def _records(
        self,
        material: OpenPBRMaterial,
        views: np.ndarray,
        third_vectors: np.ndarray,
        *,
        uvs: Sequence[Sequence[float]] | None,
        geometries: Sequence[Mapping[str, Sequence[float]]] | None,
        asset_root: str | Path | None,
    ) -> list[str]:
        count = views.shape[0]
        if third_vectors.shape[0] != count:
            raise ValueError("OpenPBR query arrays must have the same length")
        uv_array = np.zeros((count, 2), dtype=np.float32) if uvs is None else np.asarray(uvs, dtype=np.float32)
        if uv_array.ndim == 1:
            uv_array = np.broadcast_to(uv_array, (count, 2))
        if uv_array.shape != (count, 2):
            raise ValueError("OpenPBR UVs must have shape [N, 2]")
        geometry_values = geometries or ({},) * count
        if len(geometry_values) != count:
            raise ValueError("OpenPBR geometry contexts must match query count")
        root = Path(asset_root) if asset_root is not None else (
            Path(material.source_document).parent if material.source_document else Path.cwd()
        )
        records: list[str] = []
        for index in range(count):
            flat = _flatten_resolved(
                material,
                uv=uv_array[index],
                geometry=geometry_values[index],
                texture_cache=self._texture_cache,
                asset_root=root,
            )
            record = np.concatenate((flat, views[index], third_vectors[index]))
            records.append(" ".join(format(float(item), ".9g") for item in record))
        return records

    def _run(self, mode: str, records: list[str]) -> np.ndarray:
        if not self.executable.is_file():
            raise FileNotFoundError(self.executable)
        completed = subprocess.run(
            [str(self.executable.resolve())],
            input=f"{mode} {len(records)}\n" + "\n".join(records) + "\n",
            text=True,
            capture_output=True,
            check=True,
        )
        rows = [[float(item) for item in line.split()] for line in completed.stdout.splitlines() if line.strip()]
        if len(rows) != len(records):
            raise RuntimeError(f"OpenPBR probe returned {len(rows)} rows for {len(records)} queries")
        return np.asarray(rows, dtype=np.float32)

    def evaluate(
        self,
        material: OpenPBRMaterial,
        view_directions: Sequence[Sequence[float]],
        light_directions: Sequence[Sequence[float]],
        *,
        uvs: Sequence[Sequence[float]] | None = None,
        geometries: Sequence[Mapping[str, Sequence[float]]] | None = None,
        asset_root: str | Path | None = None,
    ) -> OpenPBRReferenceResult:
        views = self._vectors("view_directions", view_directions)
        lights = self._vectors("light_directions", light_directions)
        rows = self._run(
            "eval",
            self._records(material, views, lights, uvs=uvs, geometries=geometries, asset_root=asset_root),
        )
        if rows.shape != (views.shape[0], 4):
            raise RuntimeError(f"unexpected OpenPBR eval output shape {rows.shape}")
        return OpenPBRReferenceResult(rows[:, :3], rows[:, 3])

    def sample(
        self,
        material: OpenPBRMaterial,
        view_directions: Sequence[Sequence[float]],
        random_samples: Sequence[Sequence[float]],
        *,
        uvs: Sequence[Sequence[float]] | None = None,
        geometries: Sequence[Mapping[str, Sequence[float]]] | None = None,
        asset_root: str | Path | None = None,
    ) -> OpenPBRSampleResult:
        views = self._vectors("view_directions", view_directions)
        random_values = np.asarray(random_samples, dtype=np.float32)
        if random_values.ndim == 1:
            random_values = random_values[None, :]
        if random_values.shape != views.shape or np.any((random_values < 0.0) | (random_values >= 1.0)):
            raise ValueError("random_samples must have shape [N, 3] and lie in [0, 1)")
        rows = self._run(
            "sample",
            self._records(material, views, random_values, uvs=uvs, geometries=geometries, asset_root=asset_root),
        )
        if rows.shape != (views.shape[0], 8):
            raise RuntimeError(f"unexpected OpenPBR sample output shape {rows.shape}")
        return OpenPBRSampleResult(rows[:, :3], rows[:, 3:6], rows[:, 6], rows[:, 7].astype(np.uint32))
