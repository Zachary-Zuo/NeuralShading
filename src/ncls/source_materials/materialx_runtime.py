from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from ncls.source_materials.materialx import MaterialXSourceMaterial


@dataclass(frozen=True)
class MaterialXRuntimeInputs:
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


def resolve_materialx_runtime(
    document_path: Path, source: MaterialXSourceMaterial
) -> MaterialXRuntimeInputs:
    """把受支持的 native standard_surface 图解析为 canonical GPU bindings。"""

    document_path = document_path.resolve()
    root = ET.parse(document_path).getroot()
    if root.tag != "materialx" or root.get("version") != "1.38":
        raise ValueError("MaterialX Falcor subset requires a 1.38 source document")
    material = next((node for node in root if node.tag == "surfacematerial"), None)
    if material is None:
        raise ValueError(f"MaterialX document has no surfacematerial: {document_path}")
    shader_binding = next(
        (
            node
            for node in material
            if node.tag == "input" and node.get("name") == "surfaceshader"
        ),
        None,
    )
    if shader_binding is None or not shader_binding.get("nodename"):
        raise ValueError("MaterialX surfacematerial has no standard_surface binding")
    surface = next(
        (
            node
            for node in root
            if node.tag == "standard_surface"
            and node.get("name") == shader_binding.get("nodename")
        ),
        None,
    )
    if surface is None:
        raise ValueError("MaterialX surface binding does not resolve to standard_surface")
    inputs_by_name = {node.get("name"): node for node in surface if node.tag == "input"}
    values = np.zeros(24, dtype=np.float32)
    values[0], values[1:4], values[8], values[9:12], values[12], values[14] = (
        1.0,
        (0.8, 0.8, 0.8),
        1.0,
        (1.0, 1.0, 1.0),
        0.2,
        1.5,
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

    def color(
        name: str, fallback: tuple[float, float, float]
    ) -> tuple[float, float, float]:
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
        "transmission",
        "transmission_scatter_anisotropy",
        "transmission_dispersion",
        "transmission_extra_roughness",
        "subsurface",
        "subsurface_anisotropy",
        "sheen",
        "coat",
        "thin_film_thickness",
    ):
        if abs(scalar(name, 0.0)) > 1e-8:
            raise ValueError(
                f"MaterialX Falcor surface-response subset does not support nonzero {name}"
            )
    thin_walled = inputs_by_name.get("thin_walled")
    if thin_walled is not None and thin_walled.get("value", "false") != "false":
        raise ValueError("MaterialX Falcor surface-response subset requires thin_walled=false")

    graphs = {node.get("name"): node for node in root if node.tag == "nodegraph"}

    def connected_node(input_node: ET.Element, expected: str) -> ET.Element:
        graph_name = input_node.get("nodegraph")
        output_name = input_node.get("output")
        if not graph_name or not output_name or graph_name not in graphs:
            raise ValueError(
                f"MaterialX input {input_node.get('name')} has an invalid graph output"
            )
        graph = graphs[graph_name]
        output = next(
            (
                node
                for node in graph
                if node.tag == "output" and node.get("name") == output_name
            ),
            None,
        )
        if output is None or not output.get("nodename"):
            raise ValueError("MaterialX graph output has no node")
        result = next(
            (
                node
                for node in graph
                if node.tag == expected and node.get("name") == output.get("nodename")
            ),
            None,
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
        file_node = next(
            (
                node
                for node in image
                if node.tag == "input" and node.get("name") == "file"
            ),
            None,
        )
        if (
            file_node is None
            or file_node.get("type") != "filename"
            or not file_node.get("value")
        ):
            raise ValueError("MaterialX image has no filename")
        color_space = file_node.get("colorspace", "")
        if expected_color_space is not None and color_space != expected_color_space:
            raise ValueError(f"MaterialX image {image.get('name')} has the wrong colorspace")
        if expected_color_space is None and color_space:
            raise ValueError(
                f"MaterialX raw image {image.get('name')} has an unexpected colorspace"
            )
        texcoord = next(
            (
                node
                for node in image
                if node.tag == "input" and node.get("name") == "texcoord"
            ),
            None,
        )
        if texcoord is None or texcoord.get("type") != "vector2" or not texcoord.get("nodename"):
            raise ValueError(
                f"MaterialX image {image.get('name')} must use an explicit texcoord node"
            )
        path = (document_path.parent / file_node.get("value", "")).resolve()
        path.relative_to(document_path.parent.resolve())
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    textures: dict[str, Path | None] = {
        "base_color": None,
        "roughness": None,
        "metalness": None,
        "normal": None,
    }
    for name, value_offset, flag_offset in (
        ("base_color", 1, 6),
        ("specular_roughness", 12, 13),
        ("metalness", 5, 7),
    ):
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
        normal_in = next(
            (
                node
                for node in normal_map
                if node.tag == "input" and node.get("name") == "in"
            ),
            None,
        )
        if normal_in is None or not normal_in.get("nodename"):
            raise ValueError("MaterialX normalmap has no image input")
        image = next(
            (
                node
                for node in graph
                if node.tag == "image" and node.get("name") == normal_in.get("nodename")
            ),
            None,
        )
        if image is None:
            raise ValueError("MaterialX normalmap input does not resolve to image")
        textures["normal"] = image_path(image, "vector3")
        scale = next(
            (
                node
                for node in normal_map
                if node.tag == "input" and node.get("name") == "scale"
            ),
            None,
        )
        values[17] = float(scale.get("value", "1")) if scale is not None else 1.0
        values[18] = 1.0
    displacement_path = None
    displacement_binding = next(
        (
            node
            for node in material
            if node.tag == "input" and node.get("name") == "displacementshader"
        ),
        None,
    )
    if displacement_binding is not None and displacement_binding.get("nodename"):
        displacement = next(
            (
                node
                for node in root
                if node.tag == "displacement"
                and node.get("name") == displacement_binding.get("nodename")
            ),
            None,
        )
        if displacement is None:
            raise ValueError(
                "MaterialX displacement binding does not resolve to displacement"
            )
        displacement_input = next(
            (
                node
                for node in displacement
                if node.tag == "input" and node.get("name") == "displacement"
            ),
            None,
        )
        if displacement_input is None:
            raise ValueError("MaterialX displacement shader has no displacement input")
        displacement_path = image_path(
            connected_node(displacement_input, "image"), "float"
        )
    if not float(values[14]) > 0.0 or not np.all(np.isfinite(values)):
        raise ValueError("MaterialX source material has invalid standard_surface inputs")
    return MaterialXRuntimeInputs(
        source,
        values,
        textures["base_color"],
        textures["roughness"],
        textures["metalness"],
        textures["normal"],
        displacement_path,
    )


__all__ = ["MaterialXRuntimeInputs", "resolve_materialx_runtime"]
