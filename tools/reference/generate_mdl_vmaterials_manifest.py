from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping
import uuid

from tqdm import tqdm

from ncls.core.identity import sha256_file, sha256_json
from ncls.paths import PROJECT_ROOT
from ncls.references.mdl import (
    MDL_SDK_BUILD,
    MdlCompiledArtifact,
    MdlModuleDiscovery,
    MdlProgramProvider,
    create_mdl_program_provider,
)
from ncls.source_materials.mdl import module_path, snapshot_from_mdl_artifact
from ncls.source_materials.mdl_catalog import MdlVmaterialsCatalog


ARCHIVE = {
    "url": "https://d4i3qtqj3r0z5.cloudfront.net/vMaterials_2_4_0_NVD%40020240.zip",
    "file": "vMaterials_2_4_0_NVD@020240.zip",
    "content_length": 2_220_534_625,
    "sha256": "ab8116e1944c03ae622b2637939510eca9c522a07fd701cf91948fa54e194204",
    "etag": '"719019a683a90c3081489351984b0735-265"',
    "last_modified": "Fri, 12 Jul 2024 13:31:48 GMT",
}


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    primary_asset_id: str
    coverage_role: str
    asset_coverage_role: str
    evaluation_role: str
    module: str
    primary_export: str
    preset_count: int


FAMILIES = (
    FamilySpec("ceramic-tiles-glazed-versailles", "ceramic-tiles-glazed-versailles", "glazed-ceramic", "glazed-ceramic", "parameterized", "::vMaterials_2::Ceramic::Ceramic_Tiles_Glazed_Versailles", "Ceramic_Tiles_Glazed_Versailles", 27),
    FamilySpec("carpaint-metallic", "carpaint-metallic", "metallic-car-paint", "metallic-car-paint", "parameterized", "::vMaterials_2::Paint::Carpaint::Carpaint_Metallic", "Carpaint_Metallic", 31),
    FamilySpec("carpaint-shifting-flakes", "carpaint-shifting-flakes", "color-shifting-car-paint", "color-shifting-car-paint", "parameterized", "::vMaterials_2::Paint::Carpaint::Carpaint_Shifting_Flakes", "Carpaint_Shifting_Flakes", 31),
    FamilySpec("effect-pigment-metallic", "effect-pigment-metallic", "metallic-effect-pigment", "metallic-effect-pigment", "parameterized", "::vMaterials_2::Paint::Carpaint::Effect_Pigment_Metallic", "Effect_Pigment_Metallic", 11),
    FamilySpec("velvet", "velvet", "velvet-sheen", "velvet-sheen", "parameterized", "::vMaterials_2::Fabric::Velvet", "Velvet", 15),
    FamilySpec("copper-antique-brushed-patinated", "copper-antique-brushed-patinated", "brushed-patinated-metal", "brushed-patinated-metal", "parameterized", "::vMaterials_2::Metal::Copper_Antique_Brushed_Patinated", "Copper_Antique_Brushed_Patinated", 9),
    FamilySpec("aluminum-scratched", "aluminum-scratched", "micro-scratched-metal", "micro-scratched-metal", "parameterized", "::vMaterials_2::Metal::Aluminum_Scratched", "Aluminum_Scratched", 6),
    FamilySpec("retroreflective-material", "retroreflective-material", "retroreflective-layer", "retroreflective-layer", "parameterized", "::vMaterials_2::Other::Retroreflective::Retroreflective_Material", "Retroreflective_Material", 7),
    FamilySpec("carbon-fiber", "carbon-fiber", "anisotropic-carbon-fiber", "anisotropic-carbon-fiber", "discrete-subfamilies", "::vMaterials_2::Composite::Carbon_Fiber", "Carbon_Fiber", 8),
    FamilySpec("suede-leather", "suede-leather", "suede-leather-cutout", "suede-leather-cutout", "discrete-subfamilies", "::vMaterials_2::Leather::Suede_Leather", "Suede_Leather", 16),
    FamilySpec("wood-tiles-pine", "wood-tiles-pine-mosaic", "spatial-wood-tiles", "nvidia-pipeline-correspondence", "spatial-resource-control", "::vMaterials_2::Wood::Wood_Tiles_Pine", "Wood_Tiles_Pine_Mosaic", 11),
)

LEGACY_ASSET_IDS = (
    "carpaint-shifting-flakes",
    "copper-antique-brushed-patinated",
    "aluminum-scratched",
    "ceramic-tiles-glazed-versailles",
    "velvet",
    "wood-tiles-pine-mosaic",
)
ASSET_ORDER = LEGACY_ASSET_IDS + (
    "carpaint-metallic",
    "effect-pigment-metallic",
    "retroreflective-material",
    "carbon-fiber",
    "suede-leather",
)
EXPECTED_FAMILY_COUNTS = tuple(spec.preset_count for spec in FAMILIES)
EXPECTED_PRESET_COUNT = 172
EXPECTED_SUPPORTED_COUNT = 164
EXPECTED_UNSUPPORTED_COUNT = 8
_PRESET_ID = re.compile(r"[^a-z0-9]+")


def _export_name(exact_export: str) -> str:
    return exact_export.split("(", 1)[0].rsplit("::", 1)[-1]


def _preset_id(exact_export: str) -> str:
    result = _PRESET_ID.sub("-", _export_name(exact_export).lower()).strip("-")
    if not result:
        raise ValueError(f"cannot derive preset ID from MDL export: {exact_export}")
    return result


def _relative_resource(module_root: Path, filename: str) -> str:
    path = Path(filename).resolve()
    try:
        return path.relative_to(module_root).as_posix()
    except ValueError as error:
        raise ValueError(f"MDL resource escapes module root: {path}") from error


def _load_artifact(bridge: MdlProgramProvider, output: Path, *, module: str, exact_export: str) -> MdlCompiledArtifact:
    artifact = MdlCompiledArtifact.load(output)
    if artifact.manifest.get("module") != module or artifact.manifest.get("material") != exact_export:
        raise ValueError(f"MDL preset artifact identity mismatch: {output}")
    if artifact.manifest.get("texture_payloads") != "metadata-only":
        raise ValueError(f"MDL catalog artifact is not metadata-only: {output}")
    if artifact.manifest["compiler_identity"]["bridge_executable_sha256"] != sha256_file(bridge.executable):
        raise ValueError(f"MDL preset artifact was produced by another bridge: {output}")
    return artifact


def _ensure_discovery(bridge: MdlProgramProvider, artifact_root: Path, spec: FamilySpec, *, refresh_artifacts: bool) -> MdlModuleDiscovery:
    output = artifact_root / "discoveries" / spec.family_id
    bridge_digest = sha256_file(bridge.executable)
    if output.exists():
        discovery = MdlModuleDiscovery.load(output, expected_module=spec.module, expected_bridge_sha256=bridge_digest)
    elif refresh_artifacts:
        output.parent.mkdir(parents=True, exist_ok=True)
        discovery = bridge.discover_module(spec.module, output=output)
    else:
        raise FileNotFoundError(f"MDL family discovery is missing: {output}")
    if len(discovery.materials) != spec.preset_count:
        raise ValueError(f"MDL family {spec.family_id} has {len(discovery.materials)} exports, expected {spec.preset_count}")
    if sum(_export_name(item) == spec.primary_export for item in discovery.materials) != 1:
        raise ValueError(f"MDL family primary export is missing or ambiguous: {spec.family_id}")
    return discovery


def _ensure_artifact(bridge: MdlProgramProvider, artifact_root: Path, spec: FamilySpec, exact_export: str, *, refresh_artifacts: bool) -> MdlCompiledArtifact:
    output = artifact_root / "presets" / spec.family_id / _preset_id(exact_export)
    if output.exists():
        return _load_artifact(bridge, output, module=spec.module, exact_export=exact_export)
    if not refresh_artifacts:
        raise FileNotFoundError(f"MDL preset artifact is missing: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    bridge.inspect(spec.module, exact_export, output=output)
    return _load_artifact(bridge, output, module=spec.module, exact_export=exact_export)


def _runtime_resources(artifact: MdlCompiledArtifact) -> list[dict[str, Any]]:
    result = [
        {
            "kind": "ro_data",
            "index": index,
            "sha256": sha256_file(artifact.root / str(segment["path"])),
        }
        for index, segment in enumerate(artifact.manifest.get("ro_data", []))
    ]
    for texture in artifact.manifest.get("textures", []):
        if texture.get("shape") != "bsdf_data":
            continue
        data = texture.get("data")
        if not data:
            raise ValueError("MDL SDK BSDF-data texture has no payload")
        result.append({"kind": "bsdf_data", "index": int(texture["index"]), "sha256": sha256_file(artifact.root / str(data))})
    return result


def _texture_audit(module_root: Path, artifact: MdlCompiledArtifact, resource_hashes: Mapping[str, str]) -> list[dict[str, Any]]:
    result = []
    for texture in artifact.manifest.get("textures", []):
        source_value = texture.get("path")
        source_path = None if not source_value else _relative_resource(module_root, str(source_value))
        data = texture.get("data")
        result.append(
            {
                "mdl_index": int(texture["index"]),
                "shape": str(texture["shape"]),
                "pixel_type": str(texture["pixel_type"]),
                "gamma": str(texture["gamma"]),
                "effective_gamma": float(texture["effective_gamma"]),
                "dimensions": [int(texture["width"]), int(texture["height"]), int(texture["depth"])],
                "source_path": source_path,
                "source_sha256": None if source_path is None else resource_hashes[source_path],
                "payload_sha256": None if data is None else sha256_file(artifact.root / str(data)),
            }
        )
    return result


def _preset_record(module_root: Path, spec: FamilySpec, artifact: MdlCompiledArtifact) -> tuple[dict[str, Any], Mapping[str, str]]:
    snapshot = snapshot_from_mdl_artifact(artifact, module_root, pack_id="nvidia.vmaterials", pack_version="2.4.0")
    payload = json.loads(snapshot.native_payload.decode("utf-8"))
    exact_export = str(payload["export"])
    source_module_paths = sorted(str(item["path"]) for item in artifact.manifest.get("source_modules", []))
    resource_paths = sorted(snapshot.resource_hashes)
    runtime_resources = _runtime_resources(artifact)
    signature_value = {
        "source_resources": {path: snapshot.resource_hashes[path] for path in resource_paths},
        "runtime_resources": runtime_resources,
    }
    capability = dict(artifact.manifest["capability_audit"])
    cutout = bool(capability["cutout_opacity"])
    argument_block = artifact.manifest.get("argument_block")
    audit = {
        **capability,
        "argument_block_bytes": 0 if argument_block is None else int(argument_block["size"]),
        "ro_data": [
            {"size": int(item["size"]), "sha256": sha256_file(artifact.root / str(item["path"]))}
            for item in artifact.manifest.get("ro_data", [])
        ],
        "df_handle_count": int(artifact.manifest["df_handle_count"]),
        "textures": _texture_audit(module_root, artifact, snapshot.resource_hashes),
    }
    evaluation_subfamily = (
        "punched-cutout" if cutout else
        "authored-opaque" if spec.evaluation_role == "discrete-subfamilies" else
        "spatial-resource" if spec.evaluation_role == "spatial-resource-control" else
        "parameterized"
    )
    return (
        {
            "preset_id": _preset_id(exact_export),
            "export_name": _export_name(exact_export),
            "exact_export": exact_export,
            "source_snapshot_id": snapshot.snapshot_id,
            "mdl_language": str(payload["mdl_language"]),
            "parameters": payload["arguments"],
            "source_module_paths": source_module_paths,
            "resource_paths": resource_paths,
            "runtime_resources": runtime_resources,
            "resource_signature": sha256_json(signature_value),
            "compiled_material_hash": str(artifact.manifest["compiled_material_hash"]),
            "sub_expression_hashes": dict(artifact.manifest["sub_expression_hashes"]),
            "runtime_capability_audit": audit,
            "runtime_supported": not cutout,
            "unsupported_reasons": ["geometry.cutout_opacity"] if cutout else [],
            "evaluation_subfamily": evaluation_subfamily,
        },
        snapshot.resource_hashes,
    )


def _assign_resource_subfamilies(presets: list[dict[str, Any]]) -> None:
    signatures = sorted({str(item["resource_signature"]) for item in presets})
    if len(signatures) <= 1:
        return
    signature_ids = {signature: index + 1 for index, signature in enumerate(signatures)}
    for item in presets:
        if item["evaluation_subfamily"] != "punched-cutout":
            item["evaluation_subfamily"] = f"resource-set-{signature_ids[str(item['resource_signature'])]:02d}"


def _asset_record(module_root: Path, spec: FamilySpec, artifact: MdlCompiledArtifact) -> dict[str, Any]:
    snapshot = snapshot_from_mdl_artifact(artifact, module_root, pack_id="nvidia.vmaterials", pack_version="2.4.0")
    payload = json.loads(snapshot.native_payload.decode("utf-8"))
    exact_export = str(payload["export"])
    if _export_name(exact_export) != spec.primary_export:
        raise ValueError(f"artifact primary export mismatch for {spec.primary_asset_id}")
    textures = []
    for texture in artifact.manifest.get("textures", []):
        source_path = texture.get("path")
        relative = None if not source_path else _relative_resource(module_root, str(source_path))
        textures.append(
            {
                "mdl_index": int(texture["index"]),
                "shape": str(texture["shape"]),
                "source_path": relative,
                "source_sha256": None if relative is None else snapshot.resource_hashes[relative],
                "gamma": str(texture["gamma"]),
                "effective_gamma": float(texture["effective_gamma"]),
                "dimensions": [int(texture["width"]), int(texture["height"]), int(texture["depth"])],
            }
        )
    argument_block = artifact.manifest.get("argument_block")
    return {
        "asset_id": spec.primary_asset_id,
        "coverage_role": spec.asset_coverage_role,
        "module": spec.module,
        "export": exact_export,
        "mdl_language": str(payload["mdl_language"]),
        "source_snapshot_id": snapshot.snapshot_id,
        "source_modules": [
            {"module": str(item["name"]), "path": str(item["path"]), "sha256": snapshot.resource_hashes[str(item["path"])]}
            for item in artifact.manifest.get("source_modules", [])
        ],
        "parameters": payload["arguments"],
        "resource_hashes": dict(sorted(snapshot.resource_hashes.items())),
        "runtime_capability_audit": {
            "argument_block_bytes": 0 if argument_block is None else int(argument_block["size"]),
            "ro_data_segment_bytes": [int(item["size"]) for item in artifact.manifest.get("ro_data", [])],
            "df_handle_count": int(artifact.manifest["df_handle_count"]),
            "textures": textures,
            "measured_bsdf": False,
            "light_profile": False,
            "surface_bsdf_evaluate": True,
        },
    }


def _base_manifest() -> dict[str, Any]:
    return {
        "schema_name": "ncls.mdl-vmaterials-assets",
        "schema_version": 1,
        "provider": "NVIDIA",
        "pack": "vMaterials 2.4.0",
        "pack_id": "nvidia.vmaterials@2.4.0",
        "provider_page": "https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html",
        "archive": ARCHIVE,
        "license": {
            "acquisition": "NVIDIA Omniverse terms; explicit acceptance required by fetch script",
            "material_sources": "embedded NVIDIA MDL Materials license retained in each source module",
            "package_notices": "PACKAGE-LICENSES/",
        },
        "module_root": "Materials",
        "mdl_sdk": MDL_SDK_BUILD,
    }


def _verify_legacy_assets(output: Path, assets: list[dict[str, Any]]) -> None:
    if not output.is_file():
        return
    previous = json.loads(output.read_text(encoding="utf-8"))
    old_records = {str(item["asset_id"]): item for item in previous.get("assets", [])}
    new_records = {str(item["asset_id"]): item for item in assets}
    for asset_id in LEGACY_ASSET_IDS:
        if old_records.get(asset_id) != new_records.get(asset_id):
            raise ValueError(f"legacy MDL asset record drifted: {asset_id}")


def _validate_counts(families: list[dict[str, Any]]) -> None:
    counts = tuple(int(item["preset_count"]) for item in families)
    if counts != EXPECTED_FAMILY_COUNTS or sum(counts) != EXPECTED_PRESET_COUNT:
        raise ValueError(f"MDL family counts differ from the frozen cohort: {counts}")
    presets = [preset for family in families for preset in family["presets"]]
    supported = sum(bool(item["runtime_supported"]) for item in presets)
    unsupported = len(presets) - supported
    if supported != EXPECTED_SUPPORTED_COUNT or unsupported != EXPECTED_UNSUPPORTED_COUNT:
        raise ValueError(f"MDL runtime capability counts differ from 164/8: {supported}/{unsupported}")
    punched = [item for item in presets if item["unsupported_reasons"] == ["geometry.cutout_opacity"] and item["evaluation_subfamily"] == "punched-cutout"]
    if len(punched) != EXPECTED_UNSUPPORTED_COUNT:
        raise ValueError("MDL punched suede classification differs from the frozen cohort")


def _stage_and_replace(assets_output: Path, assets: Mapping[str, Any], families_output: Path, families: Mapping[str, Any]) -> None:
    token = uuid.uuid4().hex
    assets_output.parent.mkdir(parents=True, exist_ok=True)
    families_output.parent.mkdir(parents=True, exist_ok=True)
    assets_temp = assets_output.with_name(f".{assets_output.name}.{token}.tmp")
    families_temp = families_output.with_name(f".{families_output.name}.{token}.tmp")
    try:
        assets_temp.write_text(json.dumps(assets, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        families_temp.write_text(json.dumps(families, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        staged_assets = json.loads(assets_temp.read_text(encoding="utf-8"))
        if len(staged_assets.get("assets", [])) != len(FAMILIES):
            raise ValueError("staged MDL primary asset manifest is incomplete")
        MdlVmaterialsCatalog(families_temp)
        os.replace(assets_temp, assets_output)
        os.replace(families_temp, families_output)
    finally:
        for path in (assets_temp, families_temp):
            if path.exists():
                path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="发现并审计 vMaterials 首批 11 个 families 的 172 个 authored presets")
    parser.add_argument("--module-root", type=Path, default=PROJECT_ROOT / "assets/source-materials/mdl-vmaterials2/2.4.0/Materials")
    parser.add_argument("--artifact-root", type=Path, default=PROJECT_ROOT / "build/mdl-reference/vmaterials-preset-audit-v1")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "references/mdl-vmaterials2-v1/assets.json")
    parser.add_argument("--families-output", type=Path, default=PROJECT_ROOT / "references/mdl-vmaterials2-v1/families.json")
    parser.add_argument("--refresh-artifacts", action="store_true", help="只生成缺失的 discovery/preset artifacts；已存在 artifact 必须完整验证且不会被覆盖")
    parser.add_argument("--check", action="store_true", help="从已验证 artifacts 重建并比较 tracked manifests，不写 tracked 文件")
    args = parser.parse_args()
    module_root = args.module_root.resolve()
    artifact_root = args.artifact_root.resolve()
    assets_output = args.output.resolve()
    families_output = args.families_output.resolve()
    if not module_root.is_dir():
        raise FileNotFoundError("vMaterials module root is required")
    for spec in FAMILIES:
        module_path(module_root, spec.module)
    artifact_root.mkdir(parents=True, exist_ok=True)
    bridge = create_mdl_program_provider(module_root)
    discoveries = {
        spec.family_id: _ensure_discovery(bridge, artifact_root, spec, refresh_artifacts=args.refresh_artifacts)
        for spec in FAMILIES
    }
    if sum(len(item.materials) for item in discoveries.values()) != EXPECTED_PRESET_COUNT:
        raise ValueError("MDL SDK discovery did not produce exactly 172 authored exports")

    work = [(spec, exact_export) for spec in FAMILIES for exact_export in discoveries[spec.family_id].materials]
    family_presets: dict[str, list[dict[str, Any]]] = {spec.family_id: [] for spec in FAMILIES}
    resources: dict[str, str] = {}
    primary_artifacts: dict[str, MdlCompiledArtifact] = {}
    with tqdm(work, total=EXPECTED_PRESET_COUNT, unit="preset", desc="MDL preset audit") as progress:
        for spec, exact_export in progress:
            progress.set_postfix_str(spec.family_id, refresh=False)
            artifact = _ensure_artifact(bridge, artifact_root, spec, exact_export, refresh_artifacts=args.refresh_artifacts)
            preset, preset_resources = _preset_record(module_root, spec, artifact)
            family_presets[spec.family_id].append(preset)
            for path, digest in preset_resources.items():
                previous = resources.setdefault(path, digest)
                if previous != digest:
                    raise ValueError(f"MDL resource has inconsistent hashes: {path}")
            if preset["export_name"] == spec.primary_export:
                primary_artifacts[spec.primary_asset_id] = artifact

    family_records = []
    for spec in FAMILIES:
        presets = sorted(family_presets[spec.family_id], key=lambda item: item["exact_export"])
        _assign_resource_subfamilies(presets)
        source_path = module_path(module_root, spec.module).relative_to(module_root).as_posix()
        family_records.append(
            {
                "family_id": spec.family_id,
                "primary_asset_id": spec.primary_asset_id,
                "coverage_role": spec.coverage_role,
                "evaluation_role": spec.evaluation_role,
                "module": spec.module,
                "source_path": source_path,
                "primary_export": spec.primary_export,
                "preset_count": len(presets),
                "presets": presets,
            }
        )
    _validate_counts(family_records)

    specs_by_asset = {spec.primary_asset_id: spec for spec in FAMILIES}
    if set(primary_artifacts) != set(specs_by_asset):
        raise ValueError("MDL primary artifacts are incomplete")
    asset_records = [_asset_record(module_root, specs_by_asset[asset_id], primary_artifacts[asset_id]) for asset_id in ASSET_ORDER]
    _verify_legacy_assets(assets_output, asset_records)

    archive = PROJECT_ROOT / "assets" / str(ARCHIVE["file"])
    if archive.is_file() and (archive.stat().st_size != ARCHIVE["content_length"] or sha256_file(archive) != ARCHIVE["sha256"]):
        raise ValueError("local vMaterials archive differs from the registered official identity")
    assets_manifest = {**_base_manifest(), "assets": asset_records}
    families_manifest = {
        "schema_name": "ncls.mdl-vmaterials-family-catalog",
        "schema_version": 1,
        "provider": "NVIDIA",
        "pack": "vMaterials 2.4.0",
        "pack_id": "nvidia.vmaterials@2.4.0",
        "source_pack_id": "nvidia.vmaterials",
        "source_pack_version": "2.4.0",
        "module_root": "Materials",
        "mdl_sdk": MDL_SDK_BUILD,
        "bridge_executable_sha256": sha256_file(bridge.executable),
        "family_count": len(family_records),
        "preset_count": EXPECTED_PRESET_COUNT,
        "runtime_supported_count": EXPECTED_SUPPORTED_COUNT,
        "runtime_unsupported_count": EXPECTED_UNSUPPORTED_COUNT,
        "resources": dict(sorted(resources.items())),
        "families": family_records,
    }
    validation_path = artifact_root / ".families-validation.json"
    validation_path.write_text(json.dumps(families_manifest, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    try:
        MdlVmaterialsCatalog(validation_path).verify_resources(module_root)
    finally:
        validation_path.unlink(missing_ok=True)

    if args.check:
        if not assets_output.is_file() or not families_output.is_file():
            raise FileNotFoundError("tracked MDL manifests are missing")
        if json.loads(assets_output.read_text(encoding="utf-8")) != assets_manifest:
            raise ValueError("tracked MDL primary asset manifest is not deterministic")
        if json.loads(families_output.read_text(encoding="utf-8")) != families_manifest:
            raise ValueError("tracked MDL family catalog is not deterministic")
        print(f"checked {len(FAMILIES)} families / {EXPECTED_PRESET_COUNT} presets")
        return 0
    _stage_and_replace(assets_output, assets_manifest, families_output, families_manifest)
    print(assets_output)
    print(families_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
