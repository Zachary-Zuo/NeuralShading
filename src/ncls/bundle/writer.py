from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ncls.core.identity import safe_relative_uri, sha256_bytes, sha256_json, write_json_atomic
from ncls.core.scattering import MaterialPayload, RuntimePayload, read_resource_payload
from ncls.core.source import SourceSnapshot

from .manifest import FORMAT_NAME, FORMAT_VERSION, ScatteringPackageManifest
from .typed_texture import validate_typed_resource


def _typed_descriptor(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {"dtype", "shape", "stride", "alignment", "usage"}
    optional = {"kind", "module_name", "format", "color_space"}
    fields = set(value)
    if not required.issubset(fields) or not fields.issubset(required | optional):
        raise ValueError(
            f"typed payload descriptors require {sorted(required)} and only {sorted(optional)}"
        )
    return {name: value[name] for name in sorted(value)}


def _sampler_descriptor(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {"kind", "usage", "filter", "address_mode"}
    if set(value) != required:
        raise ValueError(f"sampler descriptors require exactly {sorted(required)}")
    result = {name: value[name] for name in sorted(value)}
    if (
        result["kind"] != "sampler"
        or not str(result["usage"])
        or result["filter"] not in {"point", "linear", "anisotropic"}
        or result["address_mode"] not in {"clamp", "wrap"}
    ):
        raise ValueError("sampler descriptor is invalid")
    return result


def write_scattering_package(
    root: Path | str,
    *,
    program_kind: str,
    program_key: str,
    program_version: int,
    program_descriptor_sha256: str,
    runtime_abi: str,
    source: SourceSnapshot,
    program_payload: RuntimePayload,
    asset_payload: MaterialPayload,
    validation: Mapping[str, Any],
    provenance: Mapping[str, Any],
    instance_parameters: Mapping[str, Any] | None = None,
) -> ScatteringPackageManifest:
    """写出唯一的ScatteringPackage@2 program/asset/instance布局。"""

    target = Path(root)
    if target.exists() and any(target.iterdir()):
        raise ValueError("ScatteringPackage output directory must be new or empty")
    target.mkdir(parents=True, exist_ok=True)
    if asset_payload.source_snapshot_id != source.snapshot_id:
        raise ValueError("asset payload source snapshot does not match package source")
    parameters = {
        "compiled_material_index": 0,
        **({} if instance_parameters is None else dict(instance_parameters)),
    }
    if set(parameters) != {"compiled_material_index"}:
        raise ValueError("instance parameters contain unknown fields")

    files: dict[str, str] = {}
    contents: dict[str, bytes] = {}
    for index, (name, payload) in enumerate(sorted(program_payload.module_closure.items())):
        safe_relative_uri(name)
        logical = f"program/module/{index:04d}"
        uri = f"program/modules/{name}"
        files[logical], contents[uri] = uri, payload
    files["program/entry"] = "program/entry.slang"
    contents["program/entry.slang"] = (
        f'#include "modules/{program_payload.program_module}"\n'.encode("utf-8")
    )
    program_blobs: dict[str, Any] = {}
    for name, payload in sorted(program_payload.blobs.items()):
        if safe_relative_uri(name) != Path(name).name:
            raise ValueError("program blob names must be single safe path components")
        descriptor = program_payload.blob_descriptors[name]
        if descriptor.get("kind") == "slang-module-source":
            raise ValueError(
                "ScatteringPackage program source must occur in module_closure"
            )
        logical, uri = f"program/blob/{name}", f"program/blobs/{name}.bin"
        files[logical], contents[uri] = uri, payload
        program_blobs[logical] = _typed_descriptor(descriptor)
    asset_blobs: dict[str, Any] = {}
    for name, payload in sorted(asset_payload.blobs.items()):
        if safe_relative_uri(name) != Path(name).name:
            raise ValueError("asset blob names must be single safe path components")
        logical, uri = f"asset/blob/{name}", f"assets/blobs/{name}.bin"
        files[logical], contents[uri] = uri, payload
        asset_blobs[logical] = _typed_descriptor(asset_payload.blob_descriptors[name])
    asset_resources: dict[str, Any] = {}
    for name, payload in sorted(asset_payload.resources.items()):
        safe_relative_uri(name)
        logical, uri = f"asset/resource/{name}", f"assets/resources/{name}"
        materialized = read_resource_payload(payload)
        files[logical], contents[uri] = uri, materialized
        descriptor = asset_payload.resource_descriptors[name]
        validate_typed_resource(materialized, descriptor)
        asset_resources[logical] = _typed_descriptor(descriptor)
    documents = (
        ("provenance/source", "provenance/source.json", source.to_identity_dict()),
        ("provenance/program", "provenance/program.json", dict(provenance)),
        ("validation/parity", "validation/parity.json", dict(validation)),
    )
    for logical, uri, value in documents:
        files[logical] = uri
        contents[uri] = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    for uri, payload in contents.items():
        safe_relative_uri(uri)
        path = target / uri
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    content_hashes = {
        uri: sha256_bytes(payload) for uri, payload in sorted(contents.items())
    }
    program = {
        "module": "program/entry",
        "defines": dict(program_payload.defines),
        "blobs": program_blobs,
        "samplers": {
            name: _sampler_descriptor(descriptor)
            for name, descriptor in sorted(
                program_payload.sampler_descriptors.items()
            )
        },
    }
    asset = {
        "blobs": asset_blobs,
        "resources": asset_resources,
        "samplers": {
            name: _sampler_descriptor(descriptor)
            for name, descriptor in sorted(
                asset_payload.sampler_descriptors.items()
            )
        },
    }

    def section_hashes(prefix: str) -> dict[str, str]:
        return {
            files[name]: content_hashes[files[name]]
            for name in files
            if name.startswith(prefix)
        }

    program_identity = {
        "program_kind": program_kind,
        "program_key": program_key,
        "program_version": program_version,
        "program_descriptor_sha256": program_descriptor_sha256,
        "scattering_contract_version": 1,
        "runtime_abi": runtime_abi,
        "capabilities": program_payload.capabilities,
        "program": program,
        "content_hashes": section_hashes("program/"),
    }
    program_id = sha256_json(program_identity)
    asset_identity = {
        "source_family_id": source.family_id,
        "source_contract_version": source.source_contract_version,
        "source_snapshot_id": source.snapshot_id,
        "program_descriptor_sha256": program_descriptor_sha256,
        "asset": asset,
        "content_hashes": section_hashes("asset/"),
    }
    asset_id = sha256_json(asset_identity)
    instance = {
        "bindings": {"program_id": program_id, "asset_id": asset_id},
        "parameters": {"compiled_material_index": int(parameters["compiled_material_index"])},
    }
    instance_identity = {
        "program_id": program_id,
        "asset_id": asset_id,
        "source_snapshot_id": source.snapshot_id,
        "instance": instance,
        "content_hashes": section_hashes("instance/"),
    }
    instance_id = sha256_json(instance_identity)
    package_identity = {
        "format_name": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "program_id": program_id,
        "asset_id": asset_id,
        "instance_id": instance_id,
        "validation": dict(validation),
        "provenance": dict(provenance),
        "content_hashes": {
            uri: digest
            for uri, digest in content_hashes.items()
            if uri.startswith("validation/") or uri.startswith("provenance/")
        },
    }
    manifest = ScatteringPackageManifest(
        sha256_json(package_identity),
        program_id,
        asset_id,
        instance_id,
        program_kind,
        program_key,
        program_version,
        program_descriptor_sha256,
        source.family_id,
        source.source_contract_version,
        source.snapshot_id,
        1,
        runtime_abi,
        program_payload.capabilities,
        program,
        asset,
        instance,
        dict(validation),
        dict(provenance),
        files,
        content_hashes,
    )
    write_json_atomic(target / "manifest.json", manifest.to_dict())
    return manifest
