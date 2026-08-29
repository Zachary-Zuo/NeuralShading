from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ncls.core.identity import sha256_file, sha256_json
from ncls.references.mdl import (
    MdlCompiledArtifact,
    MdlModuleDiscovery,
    create_mdl_program_provider,
)


INSPECTION_SCHEMA = "ncls.mdl-metal-export-inspection@1"
EXPECTED_MODULES = 127
EXPECTED_EXPORTS = 837
_SLUG = re.compile(r"[^a-z0-9]+")


def _slug(value: str) -> str:
    return _SLUG.sub("-", value.lower()).strip("-")


def _export_name(exact_export: str) -> str:
    return exact_export.split("(", 1)[0].rsplit("::", 1)[-1]


def _module_name(module_root: Path, path: Path) -> str:
    relative = path.resolve().relative_to(module_root).with_suffix("")
    return "::" + "::".join(relative.parts)


def _artifact_directory(root: Path, module: str, exact_export: str) -> Path:
    digest = hashlib.sha256(exact_export.encode("utf-8")).hexdigest()[:12]
    return root / _slug(module) / f"{_slug(_export_name(exact_export))}-{digest}"


def _relative_resource(module_root: Path, value: Any) -> str | None:
    if not value:
        return None
    return Path(str(value)).resolve().relative_to(module_root).as_posix()


def _discover(
    provider: Any,
    module_root: Path,
    cache_root: Path,
    bridge_digest: str,
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    metal_root = module_root / "vMaterials_2/Metal"
    paths = tuple(sorted(metal_root.rglob("*.mdl")))
    if len(paths) != EXPECTED_MODULES:
        raise ValueError(f"Metal module closure changed: expected 127, got {len(paths)}")
    result = []
    for path in tqdm(paths, desc="discover Metal modules", unit="module"):
        module = _module_name(module_root, path)
        destination = cache_root / _slug(module)
        discovery = (
            MdlModuleDiscovery.load(
                destination,
                expected_module=module,
                expected_bridge_sha256=bridge_digest,
            )
            if (destination / "discovery.json").is_file()
            else provider.discover_module(module, output=destination)
        )
        result.append(
            (
                module,
                path.relative_to(module_root).as_posix(),
                tuple(discovery.materials),
            )
        )
    if sum(len(exports) for _, _, exports in result) != EXPECTED_EXPORTS:
        raise ValueError("Metal exact export set changed from the frozen 837 entries")
    return tuple(result)


def _load_or_inspect(
    provider: Any,
    destination: Path,
    module: str,
    exact_export: str,
    bridge_digest: str,
) -> MdlCompiledArtifact:
    if (destination / "manifest.json").is_file():
        artifact = MdlCompiledArtifact.load(destination)
        manifest = artifact.manifest
        if manifest.get("module") != module or manifest.get("material") != exact_export:
            raise ValueError(f"inspection identity mismatch: {destination}")
        if manifest.get("texture_payloads") != "metadata-only":
            raise ValueError(f"inspection is not metadata-only: {destination}")
        if manifest["compiler_identity"].get("bridge_executable_sha256") != bridge_digest:
            raise ValueError(f"inspection bridge identity mismatch: {destination}")
        return artifact
    destination.parent.mkdir(parents=True, exist_ok=True)
    return provider.inspect(module, exact_export, output=destination)


def _parameter_schema(parameters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "name",
        "type",
        "editable",
        "minimum",
        "maximum",
        "soft_minimum",
        "soft_maximum",
    )
    return [{key: item[key] for key in keys if key in item} for item in parameters]


def _record(
    module_root: Path,
    module: str,
    source_path: str,
    exact_export: str,
    artifact: MdlCompiledArtifact,
) -> dict[str, Any]:
    manifest = artifact.manifest
    parameters = [dict(item) for item in manifest.get("parameters", [])]
    schema = _parameter_schema(parameters)
    textures = [
        {
            "index": int(item["index"]),
            "shape": str(item["shape"]),
            "pixel_type": str(item["pixel_type"]),
            "gamma": str(item["gamma"]),
            "effective_gamma": float(item["effective_gamma"]),
            "dimensions": [
                int(item["width"]),
                int(item["height"]),
                int(item["depth"]),
            ],
            "source_path": _relative_resource(module_root, item.get("path")),
        }
        for item in manifest.get("textures", [])
    ]
    source_modules = [
        {"name": str(item["name"]), "path": str(item["path"])}
        for item in manifest.get("source_modules", [])
    ]
    capability = dict(manifest.get("capability_audit", {}))
    graph = {
        "compiled_material_hash": str(manifest["compiled_material_hash"]),
        "sub_expression_hashes": dict(manifest.get("sub_expression_hashes", {})),
        "df_handle_count": int(manifest.get("df_handle_count", 0)),
    }
    return {
        "module": module,
        "source_path": source_path,
        "export_name": _export_name(exact_export),
        "exact_export": exact_export,
        "parameters": parameters,
        "parameter_schema": schema,
        "parameter_schema_id": sha256_json(schema),
        "editable_parameter_count": sum(bool(item.get("editable")) for item in parameters),
        "texture_count": len(textures),
        "textures": textures,
        "texture_set_id": sha256_json(textures),
        "source_modules": source_modules,
        "source_module_set_id": sha256_json(source_modules),
        "capability_audit": capability,
        "capability_id": sha256_json(capability),
        "graph_identity": graph,
        "graph_identity_id": sha256_json(graph),
        "argument_block_bytes": 0
        if manifest.get("argument_block") is None
        else int(manifest["argument_block"]["size"]),
        "ro_data_bytes": sum(int(item["size"]) for item in manifest.get("ro_data", [])),
        "generated_code_bytes": (artifact.root / str(manifest["code"])).stat().st_size,
        "artifact_manifest_sha256": sha256_file(artifact.root / "manifest.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="使用锁定MDL SDK discovery/class inspection审计全量vMaterials 2 Metal。"
    )
    parser.add_argument(
        "--module-root",
        type=Path,
        default=PROJECT_ROOT / "assets/source-materials/mdl-vmaterials2/2.4.0/Materials",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    module_root = arguments.module_root.resolve()
    output = arguments.output.resolve()
    provider = create_mdl_program_provider(module_root)
    bridge_digest = sha256_file(provider.executable)
    modules = _discover(
        provider, module_root, output / "discovery", bridge_digest
    )
    records = []
    failures = []
    progress = tqdm(total=EXPECTED_EXPORTS, desc="inspect Metal exports", unit="export")
    for module, source_path, exports in modules:
        for exact_export in exports:
            destination = _artifact_directory(output / "artifacts", module, exact_export)
            try:
                artifact = _load_or_inspect(
                    provider, destination, module, exact_export, bridge_digest
                )
                records.append(
                    _record(module_root, module, source_path, exact_export, artifact)
                )
            except Exception as error:
                failures.append(
                    {
                        "module": module,
                        "source_path": source_path,
                        "exact_export": exact_export,
                        "artifact_path": str(destination),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            progress.update(1)
    progress.close()
    counters = {
        "parameter": Counter(record["parameter_schema_id"] for record in records),
        "graph": Counter(record["graph_identity_id"] for record in records),
        "texture": Counter(record["texture_set_id"] for record in records),
        "capability": Counter(record["capability_id"] for record in records),
    }
    summary = {
        "schema": INSPECTION_SCHEMA,
        "module_root": str(module_root),
        "bridge_executable": str(provider.executable),
        "bridge_executable_sha256": bridge_digest,
        "module_count": len(modules),
        "expected_export_count": EXPECTED_EXPORTS,
        "inspected_export_count": len(records),
        "failed_export_count": len(failures),
        "unique_parameter_schema_count": len(counters["parameter"]),
        "unique_graph_identity_count": len(counters["graph"]),
        "unique_texture_set_count": len(counters["texture"]),
        "unique_capability_count": len(counters["capability"]),
        "total_unique_texture_paths": len(
            {
                texture["source_path"]
                for record in records
                for texture in record["textures"]
                if texture["source_path"] is not None
            }
        ),
        "records": records,
        "failures": failures,
    }
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "summary.json.tmp"
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output / "summary.json")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key.endswith("_count")},
            ensure_ascii=False,
        )
    )
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
