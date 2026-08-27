from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ncls.core.identity import safe_relative_uri, sha256_bytes, sha256_json, write_json_atomic
from ncls.core.scattering import MaterialPayload, RuntimePayload
from ncls.core.source import SourceSnapshot

from .manifest import FORMAT_NAME, FORMAT_VERSION, ScatteringPackageManifest
from .typed_texture import validate_typed_resource


def _typed_descriptor(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {"dtype", "shape", "stride", "alignment", "usage"}
    if set(value) != required:
        raise ValueError(f"typed blob descriptor fields must be exactly {sorted(required)}")
    return {name: value[name] for name in sorted(value)}


def write_scattering_package(
    root: Path | str,
    *,
    program_kind: str,
    program_key: str,
    program_version: int,
    program_descriptor_sha256: str,
    runtime_abi: str,
    source: SourceSnapshot,
    runtime: RuntimePayload,
    material: MaterialPayload,
    validation: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> ScatteringPackageManifest:
    """由 reference/method 公用的一次性 package writer。"""

    target = Path(root)
    if target.exists() and any(target.iterdir()):
        raise ValueError("ScatteringPackage output directory must be new or empty")
    target.mkdir(parents=True, exist_ok=True)
    if material.source_snapshot_id != source.snapshot_id:
        raise ValueError("material payload source snapshot does not match package source")
    files: dict[str, str] = {}
    contents: dict[str, bytes] = {}
    module_names: dict[str, str] = {}
    for index, (name, payload) in enumerate(sorted(runtime.module_closure.items())):
        safe_relative_uri(name)
        logical = f"runtime/module/{index:04d}"
        uri = f"runtime/modules/{name}"
        files[logical], contents[uri], module_names[name] = uri, payload, logical
    files["runtime/program"] = "runtime/program.slang"
    contents["runtime/program.slang"] = (
        f'#include "modules/{runtime.program_module}"\n'.encode("utf-8")
    )
    runtime_blobs: dict[str, Any] = {}
    for name, payload in sorted(runtime.blobs.items()):
        if safe_relative_uri(name) != Path(name).name:
            raise ValueError("runtime blob names must be single safe path components")
        logical, uri = f"runtime/blob/{name}", f"runtime/blobs/{name}.bin"
        files[logical], contents[uri] = uri, payload
        runtime_blobs[logical] = _typed_descriptor(runtime.blob_descriptors[name])
    material_blobs: dict[str, Any] = {}
    for name, payload in sorted(material.blobs.items()):
        if safe_relative_uri(name) != Path(name).name:
            raise ValueError("material blob names must be single safe path components")
        logical, uri = f"material/blob/{name}", f"materials/asset/blobs/{name}.bin"
        files[logical], contents[uri] = uri, payload
        material_blobs[logical] = _typed_descriptor(material.blob_descriptors[name])
    material_resources: dict[str, Any] = {}
    for name, payload in sorted(material.resources.items()):
        safe_relative_uri(name)
        logical, uri = f"material/resource/{name}", f"materials/asset/resources/{name}"
        files[logical], contents[uri] = uri, payload
        descriptor = material.resource_descriptors.get(name, {
            "dtype": "uint8", "shape": [len(payload)], "stride": 1,
            "alignment": 1, "usage": "source-resource",
        })
        validate_typed_resource(payload, descriptor)
        material_resources[logical] = _typed_descriptor(descriptor)
    documents = (
        ("provenance/source", "provenance/source.json", source.to_identity_dict()),
        ("provenance/program", "provenance/program.json", dict(provenance)),
        ("validation/parity", "validation/parity.json", dict(validation)),
    )
    for logical, uri, value in documents:
        files[logical] = uri
        contents[uri] = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    for uri, payload in contents.items():
        safe_relative_uri(uri)
        path = target / uri
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    content_hashes = {uri: sha256_bytes(payload) for uri, payload in sorted(contents.items())}
    program = {"module": "runtime/program", "defines": dict(runtime.defines), "blobs": runtime_blobs}
    material_value = {"blobs": material_blobs, "resources": material_resources}
    program_identity = {
        "program_kind": program_kind, "program_key": program_key, "program_version": program_version,
        "program_descriptor_sha256": program_descriptor_sha256, "scattering_contract_version": 1,
        "runtime_abi": runtime_abi, "capabilities": runtime.capabilities, "program": program,
        "content_hashes": {files[name]: content_hashes[files[name]] for name in files if name == program["module"] or name.startswith("runtime/")},
    }
    program_runtime_id = sha256_json(program_identity)
    material_identity = {
        "source_family_id": source.family_id, "source_contract_version": source.source_contract_version,
        "source_snapshot_id": source.snapshot_id, "program_descriptor_sha256": program_descriptor_sha256,
        "material": material_value,
        "content_hashes": {files[name]: content_hashes[files[name]] for name in files if name.startswith("material/")},
    }
    material_asset_id = sha256_json(material_identity)
    package_identity = {
        "format_name": FORMAT_NAME, "format_version": FORMAT_VERSION,
        "program_runtime_id": program_runtime_id, "material_asset_id": material_asset_id,
        "validation": dict(validation), "provenance": dict(provenance),
        "content_hashes": {uri: digest for uri, digest in content_hashes.items() if uri.startswith("validation/") or uri.startswith("provenance/")},
    }
    manifest = ScatteringPackageManifest(
        sha256_json(package_identity), program_runtime_id, material_asset_id, program_kind,
        program_key, program_version, program_descriptor_sha256, source.family_id,
        source.source_contract_version, source.snapshot_id, 1, runtime_abi, runtime.capabilities,
        program, material_value, dict(validation), dict(provenance), files, content_hashes,
    )
    write_json_atomic(target / "manifest.json", manifest.to_dict())
    return manifest
