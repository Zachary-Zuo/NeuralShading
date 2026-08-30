from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ncls.core.identity import sha256_file

from .manifest import ScatteringPackageManifest
from .typed_texture import validate_typed_resource


@dataclass(frozen=True)
class ProgramRuntime:
    program_id: str
    capabilities: int
    module: Path
    descriptor: dict[str, Any]
    files: dict[str, Path]
    samplers: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class AssetBinding:
    asset_id: str
    source_snapshot_id: str
    descriptor: dict[str, Any]
    files: dict[str, Path]
    samplers: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class InstanceBinding:
    instance_id: str
    program_id: str
    asset_id: str
    parameters: dict[str, Any]
    descriptor: dict[str, Any]
    files: dict[str, Path]


@dataclass(frozen=True)
class ScatteringBinding:
    package_id: str
    program: ProgramRuntime
    asset: AssetBinding
    instance: InstanceBinding

    def __post_init__(self) -> None:
        if self.instance.program_id != self.program.program_id:
            raise ValueError("instance/program identity mismatch")
        if self.instance.asset_id != self.asset.asset_id:
            raise ValueError("instance/asset identity mismatch")


@dataclass(frozen=True)
class ScatteringPackage:
    root: Path
    manifest: ScatteringPackageManifest

    @classmethod
    def open(cls, root: Path | str, *, verify_hashes: bool = True) -> "ScatteringPackage":
        path = Path(root).resolve()
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("ScatteringPackage manifest.json is missing")
        manifest = ScatteringPackageManifest.from_json(
            manifest_path.read_text(encoding="utf-8")
        )
        for uri, expected in manifest.content_hashes.items():
            target = (path / uri).resolve()
            try:
                target.relative_to(path)
            except ValueError as error:
                raise ValueError(f"ScatteringPackage URI escapes package root: {uri}") from error
            if not target.is_file():
                raise ValueError(f"ScatteringPackage content is missing: {uri}")
            if verify_hashes and sha256_file(target) != expected:
                raise ValueError(f"ScatteringPackage content hash mismatch: {uri}")
        for logical_name, descriptor in manifest.asset["resources"].items():
            resource_path = path / manifest.files[logical_name]
            validate_typed_resource(resource_path.read_bytes(), descriptor)
        compiled_material_count: int | None = None
        for section in (
            manifest.program["blobs"],
            manifest.asset["blobs"],
            manifest.instance["blobs"],
        ):
            for logical_name, descriptor in section.items():
                payload = (path / manifest.files[logical_name]).read_bytes()
                stride = int(descriptor["stride"])
                if not payload or len(payload) % stride:
                    raise ValueError(
                        f"ScatteringPackage typed blob size mismatch: {logical_name}"
                    )
                if descriptor["usage"] == "gNclsCompiledMaterials":
                    compiled_material_count = len(payload) // stride
        if (
            compiled_material_count is not None
            and int(manifest.instance["parameters"]["compiled_material_index"])
            >= compiled_material_count
        ):
            raise ValueError(
                "ScatteringPackage compiled_material_index is outside the asset blob"
            )
        return cls(path, manifest)

    def file(self, logical_name: str) -> Path:
        try:
            path = (self.root / self.manifest.files[logical_name]).resolve()
        except KeyError as error:
            raise KeyError(f"ScatteringPackage has no file {logical_name!r}") from error
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError("ScatteringPackage logical file escapes package root") from error
        return path

    def create_binding(self) -> ScatteringBinding:
        module = str(self.manifest.program["module"])
        program_files = {
            name: self.file(name)
            for name in self.manifest.files
            if name.startswith("program/")
        }
        asset_files = {
            name: self.file(name)
            for name in self.manifest.files
            if name.startswith("asset/")
        }
        instance_files = {
            name: self.file(name)
            for name in self.manifest.files
            if name.startswith("instance/")
        }
        program = ProgramRuntime(
            self.manifest.program_id,
            self.manifest.capabilities,
            self.file(module),
            dict(self.manifest.program),
            program_files,
            {
                name: dict(value)
                for name, value in self.manifest.program["samplers"].items()
            },
        )
        asset = AssetBinding(
            self.manifest.asset_id,
            self.manifest.source_snapshot_id,
            dict(self.manifest.asset),
            asset_files,
            {
                name: dict(value)
                for name, value in self.manifest.asset["samplers"].items()
            },
        )
        instance = InstanceBinding(
            self.manifest.instance_id,
            str(self.manifest.instance["bindings"]["program_id"]),
            str(self.manifest.instance["bindings"]["asset_id"]),
            dict(self.manifest.instance["parameters"]),
            dict(self.manifest.instance),
            instance_files,
        )
        return ScatteringBinding(self.manifest.package_id, program, asset, instance)
