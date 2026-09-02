from __future__ import annotations

from dataclasses import dataclass
import json
from collections import OrderedDict
from pathlib import Path
import re
from typing import Any, Mapping

from ncls.core.identity import sha256_file
from ncls.core.source import SourceSnapshot
from ncls.references.mdl import MDL_SDK_BUILD, MdlCompiledArtifact, canonical_mdl_payload


MDL_FAMILY_ID = "mdl.program@1"
MDL_NATIVE_SCHEMA = "ncls.mdl-source@1"
_LANGUAGE = re.compile(r"^\s*mdl\s+([0-9.]+)\s*;", re.MULTILINE)

# Source snapshots repeatedly reference the same module/texture files across
# hundreds of typed exports. Keep integrity hashing strict while avoiding a
# full read for every repeated hardlink/inode in one producer process.
_SOURCE_HASH_CACHE_CAPACITY = 4096
_SOURCE_HASH_CACHE: OrderedDict[tuple[int, int, int, int, int], str] = OrderedDict()


def _cached_source_sha256(path: Path) -> str:
    stat = path.stat()
    key = (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )
    digest = _SOURCE_HASH_CACHE.get(key)
    if digest is not None:
        _SOURCE_HASH_CACHE.move_to_end(key)
        return digest
    digest = sha256_file(path)
    _SOURCE_HASH_CACHE[key] = digest
    if len(_SOURCE_HASH_CACHE) > _SOURCE_HASH_CACHE_CAPACITY:
        _SOURCE_HASH_CACHE.popitem(last=False)
    return digest


def module_path(module_root: Path, module: str) -> Path:
    if not module.startswith("::") or module.endswith("::"):
        raise ValueError("MDL module name must be absolute")
    relative = Path(*module[2:].split("::")).with_suffix(".mdl")
    root = module_root.resolve()
    result = (root / relative).resolve()
    try:
        result.relative_to(root)
    except ValueError as error:
        raise ValueError("MDL module escapes the configured module root") from error
    if not result.is_file():
        raise FileNotFoundError(f"MDL module source is missing: {result}")
    return result


@dataclass(frozen=True)
class MdlMaterialSource:
    module_root: Path
    pack_id: str
    pack_version: str
    module: str
    export: str
    arguments: Mapping[str, Mapping[str, Any]]
    mdl_language: str
    mdl_sdk: str = MDL_SDK_BUILD

    def __post_init__(self) -> None:
        root = self.module_root.resolve()
        module_path(root, self.module)
        if not self.pack_id or not self.pack_version or not self.export.startswith(self.module + "::"):
            raise ValueError("MDL source identity is incomplete")
        if self.mdl_sdk != MDL_SDK_BUILD:
            raise ValueError("MDL source requires the locked SDK build")
        object.__setattr__(self, "module_root", root)
        object.__setattr__(self, "arguments", {name: dict(value) for name, value in self.arguments.items()})

    def to_payload(self) -> bytes:
        return canonical_mdl_payload(
            {
                "schema": MDL_NATIVE_SCHEMA,
                "pack_id": self.pack_id,
                "pack_version": self.pack_version,
                "module": self.module,
                "export": self.export,
                "arguments": dict(self.arguments),
                "compilation_mode": "class",
                "mdl_language": self.mdl_language,
                "mdl_sdk": self.mdl_sdk,
            }
        )

    @classmethod
    def from_snapshot(cls, snapshot: SourceSnapshot) -> "MdlMaterialSource":
        if snapshot.family_id != MDL_FAMILY_ID or snapshot.native_schema_id != MDL_NATIVE_SCHEMA:
            raise ValueError("snapshot is not an MDL program")
        payload = json.loads(snapshot.native_payload.decode("utf-8"))
        root = snapshot.editor_metadata.get("module_root")
        if not root:
            raise ValueError("MDL snapshot has no runtime module root")
        return cls(
            Path(str(root)),
            str(payload["pack_id"]),
            str(payload["pack_version"]),
            str(payload["module"]),
            str(payload["export"]),
            payload.get("arguments", {}),
            str(payload["mdl_language"]),
            str(payload["mdl_sdk"]),
        )


def snapshot_from_mdl_artifact(
    artifact: MdlCompiledArtifact,
    module_root: Path,
    *,
    pack_id: str,
    pack_version: str,
) -> SourceSnapshot:
    resolved_root = module_root.resolve()
    module = str(artifact.manifest["module"])
    source_path = module_path(resolved_root, module)
    source_text = source_path.read_text(encoding="utf-8")
    match = _LANGUAGE.search(source_text)
    if match is None:
        raise ValueError("MDL source has no language version declaration")
    textures_by_name = {
        str(texture.get("name")): texture
        for texture in artifact.manifest.get("textures", [])
        if texture.get("name")
    }
    arguments = {}
    for parameter in artifact.manifest.get("parameters", []):
        mdl_type = str(parameter["type"])
        value = parameter.get("value")
        if mdl_type.startswith("texture_") and value is not None:
            texture = textures_by_name.get(str(value))
            filename = None if texture is None else texture.get("path")
            if not filename:
                raise ValueError(f"MDL resource parameter has no resolved file: {parameter['name']}")
            path = Path(str(filename)).resolve()
            try:
                relative = path.relative_to(resolved_root).as_posix()
            except ValueError as error:
                raise ValueError(f"MDL resource parameter escapes the pack root: {path}") from error
            value = {
                "path": relative,
                "effective_gamma": float(texture.get("effective_gamma", 1.0)),
            }
        arguments[str(parameter["name"])] = {
            "mdl_type": mdl_type,
            "value": value,
            "editable": bool(parameter.get("editable", False)),
            **({"choices": parameter["choices"]} if parameter.get("choices") else {}),
            **({"minimum": parameter["minimum"]} if "minimum" in parameter else {}),
            **({"maximum": parameter["maximum"]} if "maximum" in parameter else {}),
            **({"soft_minimum": parameter["soft_minimum"]} if "soft_minimum" in parameter else {}),
            **({"soft_maximum": parameter["soft_maximum"]} if "soft_maximum" in parameter else {}),
        }
    source = MdlMaterialSource(
        resolved_root,
        pack_id,
        pack_version,
        module,
        str(artifact.manifest["material"]),
        arguments,
        match.group(1),
    )
    source_modules = artifact.manifest.get("source_modules") or (
        {"name": module, "path": source_path.relative_to(module_root.resolve()).as_posix()},
    )
    resource_hashes = {}
    for dependency in source_modules:
        relative = Path(str(dependency["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"MDL dependency path is not pack-relative: {relative}")
        dependency_path = (resolved_root / relative).resolve()
        try:
            dependency_path.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(f"MDL dependency escapes the pack root: {dependency_path}") from error
        if not dependency_path.is_file():
            raise FileNotFoundError(f"MDL dependency is missing: {dependency_path}")
        resource_hashes[relative.as_posix()] = _cached_source_sha256(dependency_path)
    relative_module = source_path.relative_to(resolved_root).as_posix()
    if relative_module not in resource_hashes:
        raise ValueError("MDL dependency closure does not contain the root module")
    for texture in artifact.manifest.get("textures", []):
        filename = texture.get("path")
        if not filename:
            continue
        path = Path(str(filename)).resolve()
        try:
            relative = path.relative_to(resolved_root).as_posix()
        except ValueError as error:
            raise ValueError(f"MDL texture escapes the pack root: {path}") from error
        resource_hashes[relative] = _cached_source_sha256(path)
    return SourceSnapshot(
        MDL_FAMILY_ID,
        1,
        MDL_NATIVE_SCHEMA,
        _cached_source_sha256(source_path),
        source.to_payload(),
        resource_hashes,
        editor_metadata={
            "module_root": str(resolved_root),
            "inspection_artifact": str(artifact.root),
        },
        native_object=source,
    )
