from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ncls.core.identity import sha256_file

from .manifest import ScatteringPackageManifest
from .typed_texture import validate_typed_resource


@dataclass(frozen=True)
class ScatteringBinding:
    program_runtime_id: str
    material_asset_id: str
    source_snapshot_id: str
    capabilities: int
    program_module: Path
    files: dict[str, Path]
    program: dict[str, Any]
    material: dict[str, Any]


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
        manifest = ScatteringPackageManifest.from_json(manifest_path.read_text(encoding="utf-8"))
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
        for logical_name, descriptor in manifest.material["resources"].items():
            resource_path = path / manifest.files[logical_name]
            validate_typed_resource(resource_path.read_bytes(), descriptor)
        return cls(path, manifest)

    def file(self, logical_name: str) -> Path:
        try:
            return (self.root / self.manifest.files[logical_name]).resolve()
        except KeyError as error:
            raise KeyError(f"ScatteringPackage has no file {logical_name!r}") from error

    def create_binding(self) -> ScatteringBinding:
        module = str(self.manifest.program["module"])
        return ScatteringBinding(
            self.manifest.program_runtime_id,
            self.manifest.material_asset_id,
            self.manifest.source_snapshot_id,
            self.manifest.capabilities,
            self.file(module),
            {name: self.file(name) for name in self.manifest.files},
            dict(self.manifest.program),
            dict(self.manifest.material),
        )
