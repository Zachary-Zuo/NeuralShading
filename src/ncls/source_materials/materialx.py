from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import MaterialX as mx
import MaterialX.PyMaterialXGenGlsl as mx_gen_glsl
import MaterialX.PyMaterialXGenShader as mx_gen_shader


MATERIALX_SOURCE_SCHEMA = "ncls.materialx-source-material"
MATERIALX_SOURCE_VERSION = 1
EXPECTED_MATERIALX_VERSION = "1.39.4"


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MaterialXSourceMaterial:
    asset_id: str
    document_uri: str
    physical_size_mm: tuple[float, float]
    files_hash: str
    source_manifest_sha256: str
    license: str = "CC0-1.0"

    def __post_init__(self) -> None:
        if not self.asset_id or not self.document_uri or not self.files_hash:
            raise ValueError("MaterialX source material identity and document URI must be nonempty")
        if len(self.physical_size_mm) != 2 or any(value <= 0 for value in self.physical_size_mm):
            raise ValueError("MaterialX physical size must contain two positive millimeter values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": MATERIALX_SOURCE_SCHEMA,
            "schema_version": MATERIALX_SOURCE_VERSION,
            "asset_id": self.asset_id,
            "document_uri": self.document_uri,
            "physical_size_mm": list(self.physical_size_mm),
            "files_hash": self.files_hash,
            "source_manifest_sha256": self.source_manifest_sha256,
            "license": self.license,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=indent) + "\n"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MaterialXSourceMaterial:
        if value.get("schema_name") != MATERIALX_SOURCE_SCHEMA or value.get("schema_version") != MATERIALX_SOURCE_VERSION:
            raise ValueError("unsupported MaterialX source material schema")
        dimensions = tuple(float(item) for item in value["physical_size_mm"])
        return cls(
            str(value["asset_id"]),
            str(value["document_uri"]),
            dimensions,  # type: ignore[arg-type]
            str(value["files_hash"]),
            str(value["source_manifest_sha256"]),
            str(value.get("license", "CC0-1.0")),
        )

    @classmethod
    def from_json(cls, text: str) -> MaterialXSourceMaterial:
        value = json.loads(text)
        if not isinstance(value, Mapping):
            raise ValueError("MaterialX source material JSON root must be an object")
        return cls.from_dict(value)


class MaterialXAssetCatalog:
    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = _load_json(self.manifest_path)
        if self.manifest.get("schema_name") != "ncls.polyhaven-materialx-assets" or self.manifest.get("schema_version") != 1:
            raise ValueError("unsupported Poly Haven MaterialX asset manifest")
        self.manifest_sha256 = _hash(self.manifest_path, "sha256")
        self._assets = {str(item["asset_id"]): item for item in self.manifest["assets"]}
        if len(self._assets) != len(self.manifest["assets"]):
            raise ValueError("duplicate MaterialX asset IDs")

    @property
    def asset_ids(self) -> tuple[str, ...]:
        return tuple(self._assets)

    def source_material(self, asset_id: str) -> MaterialXSourceMaterial:
        value = self._assets[asset_id]
        dimensions = tuple(float(item) for item in value["physical_size_mm"])
        return MaterialXSourceMaterial(
            asset_id,
            str(value["materialx_file"]),
            dimensions,  # type: ignore[arg-type]
            str(value["files_hash"]),
            self.manifest_sha256,
            str(self.manifest["license"]),
        )

    def records(self, asset_id: str) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._assets[asset_id]["files"])


@dataclass(frozen=True)
class MaterialXEditableInput:
    name_path: str
    value_type: str
    value: str


@dataclass(frozen=True)
class MaterialXGeneratedShader:
    name: str
    vertex_source: str
    pixel_source: str
    source_sha256: str


class LoadedMaterialX:
    def __init__(
        self,
        material: MaterialXSourceMaterial,
        document_path: Path,
        document: mx.Document,
        standard_library: mx.Document,
        search_path: mx.FileSearchPath,
    ):
        self.material = material
        self.document_path = document_path
        self.document = document
        self.standard_library = standard_library
        self.search_path = search_path

    def editable_inputs(self) -> tuple[MaterialXEditableInput, ...]:
        result = []
        for element in self.document.traverseTree():
            if element.getCategory() == "input" and element.hasValueString():
                result.append(
                    MaterialXEditableInput(element.getNamePath(), element.getType(), element.getValueString())
                )
        return tuple(result)

    def set_input_value(self, name_path: str, value: str) -> None:
        element = self.document.getDescendant(name_path)
        if element is None or element.getCategory() != "input":
            raise KeyError(name_path)
        if element.getConnectedNode() is not None or element.getConnectedOutput() is not None:
            raise ValueError(f"cannot replace connected MaterialX input without an explicit graph edit: {name_path}")
        element.setValueString(str(value))
        valid, message = self.document.validate()
        if not valid:
            raise ValueError(f"MaterialX edit invalidated document: {message}")

    def save(self, path: str | Path) -> None:
        valid, message = self.document.validate()
        if not valid:
            raise ValueError(f"cannot save invalid MaterialX document: {message}")
        mx.writeToXmlFile(self.document, str(Path(path)))

    def referenced_files(self) -> tuple[str, ...]:
        result = []
        for element in self.document.traverseTree():
            if element.getCategory() == "input" and element.getType() == "filename" and element.hasValueString():
                result.append(element.getValueString())
        return tuple(result)

    def generate_glsl(self) -> tuple[MaterialXGeneratedShader, ...]:
        generator = mx_gen_glsl.GlslShaderGenerator.create()
        context = mx_gen_shader.GenContext(generator)
        context.registerSourceCodeSearchPath(self.search_path)
        generator.registerTypeDefs(self.document)
        options = context.getOptions()
        options.shaderInterfaceType = mx_gen_shader.ShaderInterfaceType.SHADER_INTERFACE_COMPLETE
        options.targetColorSpaceOverride = "lin_rec709"

        color_management = mx_gen_shader.DefaultColorManagementSystem.create(generator.getTarget())
        color_management.loadLibrary(self.document)
        generator.setColorManagementSystem(color_management)

        unit_system = mx_gen_shader.UnitSystem.create(generator.getTarget())
        registry = mx.UnitConverterRegistry.create()
        distance_type = self.document.getUnitTypeDef("distance")
        angle_type = self.document.getUnitTypeDef("angle")
        if distance_type is not None:
            registry.addUnitConverter(distance_type, mx.LinearUnitConverter.create(distance_type))
        if angle_type is not None:
            registry.addUnitConverter(angle_type, mx.LinearUnitConverter.create(angle_type))
        unit_system.loadLibrary(self.standard_library)
        unit_system.setUnitConverterRegistry(registry)
        generator.setUnitSystem(unit_system)
        options.targetDistanceUnit = "meter"

        generated = []
        for element in mx_gen_shader.findRenderableElements(self.document):
            name = mx.createValidName(element.getName())
            shader = generator.generate(name, element, context)
            if shader is None:
                raise RuntimeError(f"MaterialX failed to generate GLSL for {element.getNamePath()}")
            vertex = shader.getSourceCode(mx_gen_shader.VERTEX_STAGE)
            pixel = shader.getSourceCode(mx_gen_shader.PIXEL_STAGE)
            if not vertex or not pixel:
                raise RuntimeError(f"MaterialX generated empty GLSL for {element.getNamePath()}")
            digest = hashlib.sha256((vertex + "\0" + pixel).encode("utf-8")).hexdigest()
            generated.append(MaterialXGeneratedShader(name, vertex, pixel, digest))
        if not generated:
            raise RuntimeError(f"MaterialX document has no renderable element: {self.document_path}")
        return tuple(generated)


class MaterialXReference:
    def __init__(
        self,
        materialx_source_root: str | Path,
        asset_root: str | Path,
        manifest_path: str | Path,
    ):
        if mx.getVersionString() != EXPECTED_MATERIALX_VERSION:
            raise RuntimeError(
                f"MaterialX Python version mismatch: expected={EXPECTED_MATERIALX_VERSION}, actual={mx.getVersionString()}"
            )
        self.materialx_source_root = Path(materialx_source_root).resolve()
        self.asset_root = Path(asset_root).resolve()
        self.catalog = MaterialXAssetCatalog(manifest_path)
        self.search_path = mx.FileSearchPath(str(self.materialx_source_root))
        self.standard_library = mx.createDocument()
        loaded = mx.loadLibraries(mx.getDefaultDataLibraryFolders(), self.search_path, self.standard_library)
        if not loaded:
            raise RuntimeError(f"failed to load MaterialX libraries from {self.materialx_source_root}")

    def _verify_files(self, asset_id: str) -> None:
        for record in self.catalog.records(asset_id):
            path = (self.asset_root / str(record["path"])).resolve()
            if self.asset_root != path and self.asset_root not in path.parents:
                raise ValueError(f"MaterialX asset path escapes root: {record['path']}")
            if not path.is_file() or path.stat().st_size != int(record["size"]):
                raise ValueError(f"MaterialX asset file missing or has wrong size: {path}")
            if _hash(path, "md5") != str(record["md5"]):
                raise ValueError(f"MaterialX asset MD5 mismatch: {path}")

    def load(self, asset_id: str, *, verify_files: bool = True) -> LoadedMaterialX:
        material = self.catalog.source_material(asset_id)
        if verify_files:
            self._verify_files(asset_id)
        document_path = (self.asset_root / material.document_uri).resolve()
        document = mx.createDocument()
        document_search_path = mx.FileSearchPath(str(document_path.parent))
        document_search_path.append(str(self.materialx_source_root))
        mx.readFromXmlFile(document, str(document_path), document_search_path)
        document.setDataLibrary(self.standard_library)
        valid, message = document.validate()
        if not valid:
            raise ValueError(f"invalid native MaterialX document {document_path}: {message}")
        return LoadedMaterialX(material, document_path, document, self.standard_library, document_search_path)

    def render_preview(
        self,
        asset_id: str,
        viewer_executable: str | Path,
        output: str | Path,
        *,
        width: int = 512,
        height: int = 512,
        environment_samples: int = 64,
    ) -> Path:
        loaded = self.load(asset_id, verify_files=True)
        output_path = Path(output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        viewer = Path(viewer_executable).resolve()
        if not viewer.is_file():
            raise FileNotFoundError(viewer)
        command = [
            str(viewer),
            "--material", str(loaded.document_path),
            "--mesh", str(self.materialx_source_root / "resources" / "Geometry" / "shaderball.glb"),
            "--envRad", str(self.materialx_source_root / "resources" / "Lights" / "san_giuseppe_bridge.hdr"),
            "--path", str(self.materialx_source_root),
            "--screenWidth", str(width),
            "--screenHeight", str(height),
            "--envSampleCount", str(environment_samples),
            "--captureFilename", str(output_path),
        ]
        environment = os.environ.copy()
        conda_library_bin = Path(sys.prefix) / "Library" / "bin"
        environment["PATH"] = os.pathsep.join((str(viewer.parent), str(conda_library_bin), environment.get("PATH", "")))
        subprocess.run(command, check=True, env=environment, timeout=180)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError(f"MaterialXView did not produce capture: {output_path}")
        return output_path
