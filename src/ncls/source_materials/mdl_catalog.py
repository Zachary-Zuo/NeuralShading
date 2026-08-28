from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from ncls.core.identity import require_sha256, sha256_file, sha256_json
from ncls.references.mdl import MDL_SDK_BUILD


CATALOG_SCHEMA_NAME = "ncls.mdl-vmaterials-family-catalog"
CATALOG_SCHEMA_VERSION = 1


def _safe_resource_path(value: object) -> str:
    result = str(value)
    path = PurePosixPath(result)
    if not result or path.is_absolute() or ".." in path.parts or "\\" in result:
        raise ValueError(f"MDL catalog resource path is not pack-relative: {result!r}")
    return result


class MdlVmaterialsCatalog:
    """vMaterials authored family/preset registry; runtime execution remains generic MDL."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path).resolve()
        value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("MDL family catalog root must be an object")
        if (
            value.get("schema_name") != CATALOG_SCHEMA_NAME
            or value.get("schema_version") != CATALOG_SCHEMA_VERSION
        ):
            raise ValueError("unsupported MDL family catalog")
        if value.get("mdl_sdk") != MDL_SDK_BUILD:
            raise ValueError("MDL family catalog requires another SDK build")
        require_sha256(
            "MDL catalog bridge_executable_sha256",
            str(value.get("bridge_executable_sha256", "")),
        )
        resources_value = value.get("resources")
        if not isinstance(resources_value, Mapping):
            raise ValueError("MDL family catalog resources must be an object")
        resources: dict[str, str] = {}
        for path_value, digest_value in resources_value.items():
            path = _safe_resource_path(path_value)
            resources[path] = require_sha256(f"MDL catalog resource {path}", str(digest_value))
        if list(resources) != sorted(resources):
            raise ValueError("MDL family catalog resources must be sorted")

        families_value = value.get("families")
        if not isinstance(families_value, list) or not families_value:
            raise ValueError("MDL family catalog has no families")
        families: dict[str, Mapping[str, Any]] = {}
        presets: dict[tuple[str, str], Mapping[str, Any]] = {}
        exact_exports: set[tuple[str, str]] = set()
        supported = 0
        unsupported = 0
        for family_value in families_value:
            if not isinstance(family_value, Mapping):
                raise ValueError("MDL family catalog family must be an object")
            family_id = str(family_value.get("family_id", ""))
            if not family_id or family_id in families:
                raise ValueError("MDL family IDs must be nonempty and unique")
            module = str(family_value.get("module", ""))
            source_path = _safe_resource_path(family_value.get("source_path"))
            if source_path not in resources:
                raise ValueError(f"MDL family root module is absent from resources: {family_id}")
            preset_values = family_value.get("presets")
            if not isinstance(preset_values, list) or len(preset_values) != int(
                family_value.get("preset_count", -1)
            ):
                raise ValueError(f"MDL family preset count mismatch: {family_id}")
            for preset_value in preset_values:
                if not isinstance(preset_value, Mapping):
                    raise ValueError("MDL catalog preset must be an object")
                preset_id = str(preset_value.get("preset_id", ""))
                key = (family_id, preset_id)
                if not preset_id or key in presets:
                    raise ValueError("MDL family preset IDs must be nonempty and unique")
                exact_export = str(preset_value.get("exact_export", ""))
                exact_key = (module, exact_export)
                if not exact_export.startswith(module + "::") or exact_key in exact_exports:
                    raise ValueError("MDL catalog exact exports must match their module and be unique")
                exact_exports.add(exact_key)
                require_sha256(
                    f"MDL preset {family_id}/{preset_id} source_snapshot_id",
                    str(preset_value.get("source_snapshot_id", "")),
                )
                sub_expression_hashes = preset_value.get("sub_expression_hashes")
                if not isinstance(sub_expression_hashes, Mapping) or set(
                    sub_expression_hashes
                ) != {
                    "surface.scattering",
                    "geometry.normal",
                    "geometry.cutout_opacity",
                }:
                    raise ValueError("MDL preset sub-expression hashes are incomplete")
                for name, digest in sub_expression_hashes.items():
                    digest_text = str(digest)
                    if len(digest_text) != 32 or any(
                        character not in "0123456789abcdef" for character in digest_text
                    ):
                        raise ValueError(f"MDL preset {name} hash is invalid")
                resource_paths_value = preset_value.get("resource_paths")
                if not isinstance(resource_paths_value, list):
                    raise ValueError("MDL preset resource_paths must be an array")
                resource_paths = tuple(_safe_resource_path(item) for item in resource_paths_value)
                if resource_paths != tuple(sorted(set(resource_paths))) or source_path not in resource_paths:
                    raise ValueError("MDL preset resource paths must be sorted, unique, and include its module")
                if any(path not in resources for path in resource_paths):
                    raise ValueError("MDL preset references an unknown resource")
                runtime_resources = preset_value.get("runtime_resources")
                if not isinstance(runtime_resources, list):
                    raise ValueError("MDL preset runtime_resources must be an array")
                for item in runtime_resources:
                    if not isinstance(item, Mapping):
                        raise ValueError("MDL runtime resource descriptor must be an object")
                    if item.get("kind") not in {"ro_data", "bsdf_data"} or not isinstance(
                        item.get("index"), int
                    ):
                        raise ValueError("MDL runtime resource kind/index is invalid")
                    require_sha256("MDL runtime resource payload", str(item.get("sha256", "")))
                signature_value = {
                    "source_resources": {path: resources[path] for path in resource_paths},
                    "runtime_resources": runtime_resources,
                }
                if sha256_json(signature_value) != preset_value.get("resource_signature"):
                    raise ValueError("MDL preset resource signature mismatch")
                compiled_hash = str(preset_value.get("compiled_material_hash", ""))
                if len(compiled_hash) != 32 or any(c not in "0123456789abcdef" for c in compiled_hash):
                    raise ValueError("MDL preset compiled material hash is invalid")
                capability = preset_value.get("runtime_capability_audit")
                if not isinstance(capability, Mapping) or not isinstance(
                    capability.get("cutout_opacity"), bool
                ):
                    raise ValueError("MDL preset has no cutout capability audit")
                runtime_supported = preset_value.get("runtime_supported")
                reasons = preset_value.get("unsupported_reasons")
                if not isinstance(runtime_supported, bool) or not isinstance(reasons, list):
                    raise ValueError("MDL preset runtime support fields are invalid")
                expected_reasons = (
                    ["geometry.cutout_opacity"] if capability["cutout_opacity"] else []
                )
                if reasons != expected_reasons or runtime_supported == bool(
                    capability["cutout_opacity"]
                ):
                    raise ValueError("MDL preset runtime support disagrees with capability evidence")
                supported += int(runtime_supported)
                unsupported += int(not runtime_supported)
                presets[key] = preset_value
            if sum(
                str(item.get("export_name", "")) == family_value.get("primary_export")
                for item in preset_values
            ) != 1:
                raise ValueError(f"MDL family primary export is missing or ambiguous: {family_id}")
            families[family_id] = family_value

        if len(families) != int(value.get("family_count", -1)):
            raise ValueError("MDL catalog family_count mismatch")
        if len(presets) != int(value.get("preset_count", -1)):
            raise ValueError("MDL catalog preset_count mismatch")
        if supported != int(value.get("runtime_supported_count", -1)):
            raise ValueError("MDL catalog runtime_supported_count mismatch")
        if unsupported != int(value.get("runtime_unsupported_count", -1)):
            raise ValueError("MDL catalog runtime_unsupported_count mismatch")
        self.manifest: Mapping[str, Any] = value
        self.resources = resources
        self._families = families
        self._presets = presets

    @property
    def family_ids(self) -> tuple[str, ...]:
        return tuple(self._families)

    @property
    def preset_count(self) -> int:
        return len(self._presets)

    def family(self, family_id: str) -> Mapping[str, Any]:
        return self._families[family_id]

    def preset(self, family_id: str, preset_id: str) -> Mapping[str, Any]:
        return self._presets[(family_id, preset_id)]

    def verify_resources(self, module_root: str | Path) -> None:
        root = Path(module_root).resolve()
        for relative, digest in self.resources.items():
            path = (root / Path(*PurePosixPath(relative).parts)).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError(f"MDL catalog resource escapes module root: {relative}") from error
            if not path.is_file():
                raise FileNotFoundError(f"MDL catalog resource is missing: {relative}")
            if sha256_file(path) != digest:
                raise ValueError(f"MDL catalog resource hash mismatch: {relative}")

    def locator(
        self,
        family_id: str,
        preset_id: str,
        *,
        module_root: str | Path,
        allow_unsupported: bool = False,
    ) -> dict[str, Any]:
        family = self.family(family_id)
        preset = self.preset(family_id, preset_id)
        if not bool(preset["runtime_supported"]) and not allow_unsupported:
            reasons = ", ".join(str(item) for item in preset["unsupported_reasons"])
            raise ValueError(f"MDL preset is not runtime-supported: {reasons}")
        return {
            "kind": "mdl-export",
            "module_root": str(Path(module_root).resolve()),
            "module": str(family["module"]),
            "export": str(preset["exact_export"]),
            "pack_id": str(self.manifest["source_pack_id"]),
            "pack_version": str(self.manifest["source_pack_version"]),
        }
