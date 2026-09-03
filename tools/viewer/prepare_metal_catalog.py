from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Mapping, cast

import torch
from tqdm import tqdm

from ncls.bundle import write_scattering_package
from ncls.core.identity import sha256_file, sha256_json, write_json_atomic
from ncls.core.scattering import BackendCapability, InstancePayload
from ncls.core.source import SourceSnapshot, create_source_family
from ncls.learning.conformance import MethodArtifactInventory, validate_artifact_coverage
from ncls.learning.metal_asset_cook import MetalAssetCooker
from ncls.learning.metal_runtime import pack_metal_asset, quantize_runtime_model
from ncls.learning.methods import get_method
from ncls.learning.source_adapters import MetalFusedMdlSourceAdapter
from ncls.learning.training import (
    TrainingConfig,
    assess_checkpoint_readiness,
    load_checkpoint,
)
from ncls.paths import PROJECT_ROOT
from ncls.references.mdl import (
    MdlCompiledArtifact,
    MdlProgramProvider,
    create_mdl_program_provider,
)
from ncls.source_materials.mdl import snapshot_from_mdl_artifact
from ncls.source_materials.mdl_metal import MdlMetalRegistry
from ncls.viewer import (
    ViewerMaterialCatalog,
    finalize_catalog_document,
    link_parameter_view,
)


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts/metal-linux-training/long/checkpoint.step00120000.pt"
)
DEFAULT_REGISTRY = (
    PROJECT_ROOT / "references/mdl-vmaterials2-v1/metal-opaque-v1.json"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/viewer/metal-step00120000"
MAX_MDL_COMPILE_WORKERS = 4


def _validate_slang_path_budget(output_root: Path) -> None:
    if os.name != "nt":
        return
    probe = (
        output_root
        / "packages"
        / ("0" * 64)
        / "program/modules/ncls/contracts/scattering_backend.slang"
    )
    if len(str(probe)) >= 220:
        raise ValueError(
            "ViewerMaterialCatalog output path is too long for the pinned Windows "
            "Slang include loader; choose a shorter root such as artifacts/viewer/metal"
        )


def validate_preview_checkpoint(
    checkpoint: object, descriptor: object, *, diagnostic: bool = False
) -> str:
    """Require exact method identity plus explicit readiness for every preview."""

    readiness = assess_checkpoint_readiness(
        checkpoint,
        descriptor,
        mode="diagnostic-evaluator" if diagnostic else "formal",
    )
    readiness.require_ready()
    return "exact-diagnostic-evaluator-preview" if diagnostic else "exact"


def _portable(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _copy_or_link(
    source: Path,
    target: Path,
    objects: dict[str, Path],
    *,
    expected_sha256: str | None = None,
) -> None:
    digest = expected_sha256 or sha256_file(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = objects.get(digest)
    if expected_sha256 is not None and (
        existing is None or not os.path.samefile(source, existing)
    ):
        if sha256_file(source) != expected_sha256:
            raise ValueError(f"viewer catalog source hash mismatch: {source}")
    link_source = source if existing is None else existing
    try:
        os.link(link_source, target)
    except OSError:
        shutil.copy2(link_source, target)
    objects.setdefault(digest, target)


def _materialize_tree(
    source: Path,
    target: Path,
    objects: dict[str, Path],
    declared_hashes: Mapping[str, object],
) -> None:
    if target.exists():
        raise ValueError(f"viewer catalog target already exists: {target}")
    target.mkdir(parents=True)
    files = sorted(item for item in source.rglob("*") if item.is_file())
    relative_files = {path.relative_to(source).as_posix(): path for path in files}
    if set(relative_files) != set(declared_hashes) | {"manifest.json"}:
        raise ValueError("MDL runtime artifact file set differs from its manifest")
    for relative, path in relative_files.items():
        _copy_or_link(
            path,
            target / relative,
            objects,
            expected_sha256=(
                None if relative == "manifest.json" else str(declared_hashes[relative])
            ),
        )


def _artifact_sha256(artifact: MdlCompiledArtifact) -> str:
    declared = artifact.manifest.get("files_sha256")
    if not isinstance(declared, Mapping):
        raise ValueError("MDL runtime artifact has no finalized file hash table")
    return sha256_json(
        {
            "manifest.json": sha256_file(artifact.root / "manifest.json"),
            **{str(name): str(digest) for name, digest in declared.items()},
        }
    )


def _select_registry_records(
    registry: MdlMetalRegistry,
    locators: Mapping[tuple[str, str], Mapping[str, object]],
    *,
    diagnostic_preview: bool,
    limit: int | None,
) -> tuple[object, ...]:
    registry_records = tuple(registry.exports)
    registry_keys = {
        (
            str(record.exact_locator["module"]),
            str(record.exact_locator["export"]),
        )
        for record in registry_records
    }
    locator_keys = set(locators)
    if not diagnostic_preview:
        if locator_keys != registry_keys:
            raise ValueError(
                "formal checkpoint source list does not cover the Metal registry exactly"
            )
        selected = registry_records
    else:
        unknown = locator_keys - registry_keys
        if unknown:
            raise ValueError(
                "diagnostic checkpoint source list contains exports outside the Metal registry"
            )
        selected = tuple(
            record
            for record in registry_records
            if (
                str(record.exact_locator["module"]),
                str(record.exact_locator["export"]),
            )
            in locator_keys
        )
        if not selected:
            raise ValueError(
                "diagnostic checkpoint source list has no Metal registry intersection"
            )
    if limit is not None:
        if not diagnostic_preview:
            raise ValueError(
                "ViewerMaterialCatalog limit is only valid for diagnostic preview"
            )
        if not 1 <= limit <= len(selected):
            raise ValueError(
                "ViewerMaterialCatalog diagnostic limit is outside checkpoint coverage"
            )
        selected = selected[:limit]
    return selected


def _compile_reference_program(
    provider: MdlProgramProvider,
    snapshot: SourceSnapshot,
) -> MdlCompiledArtifact:
    # Publication retry and winner validation belong to the shared provider so
    # training and viewer materialization use one cross-platform policy.
    return provider.compile_snapshot(snapshot)


def _load_snapshot_with_provider(
    locator: Mapping[str, object],
    provider: MdlProgramProvider,
) -> SourceSnapshot:
    value = dict(locator)
    if value.pop("kind", None) != "mdl-export":
        raise ValueError("Metal viewer locator requires kind=mdl-export")
    module_root = Path(str(value.pop("module_root"))).resolve()
    module = str(value.pop("module"))
    export = str(value.pop("export"))
    arguments = value.pop("arguments", {})
    pack_id = str(value.pop("pack_id", "project.fixtures"))
    pack_version = str(value.pop("pack_version", "1"))
    if value or not isinstance(arguments, Mapping):
        raise ValueError(f"unexpected Metal viewer locator fields: {sorted(value)}")
    cache_key = sha256_json(
        {
            "module_root": str(module_root),
            "module": module,
            "export": export,
            "arguments": dict(arguments),
            "pack_id": pack_id,
            "pack_version": pack_version,
            "semantic_identity": provider.descriptor.semantic_identity,
            "build_identity": provider.descriptor.build_identity,
        }
    )
    output = provider.cache_root / "source-locators" / cache_key
    if output.is_dir() and not (output / "manifest.json").is_file():
        # An interrupted provider process may leave only its final cache directory.
        # The SHA-addressed locator cache is reproducible; never apply this recovery
        # to a directory that has a manifest, because manifest/hash drift must fail.
        resolved_cache = (provider.cache_root / "source-locators").resolve()
        output.resolve().relative_to(resolved_cache)
        shutil.rmtree(output)
    if output.is_dir():
        artifact = MdlCompiledArtifact.load(
            output, verify_texture_payloads=False
        )
    else:
        artifact = provider.inspect(module, export, arguments, output=output)
    return snapshot_from_mdl_artifact(
        artifact,
        module_root,
        pack_id=pack_id,
        pack_version=pack_version,
    )


def prepare_metal_catalog(
    output_root: Path,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    registry_path: Path = DEFAULT_REGISTRY,
    *,
    limit: int | None = None,
    diagnostic_preview: bool = False,
) -> ViewerMaterialCatalog:
    output_root = output_root.resolve()
    _validate_slang_path_budget(output_root)
    checkpoint_path = checkpoint_path.resolve()
    registry_path = registry_path.resolve()
    catalog_path = output_root / "catalog.json"
    if catalog_path.is_file():
        # Payload integrity is checked when the viewer selects an entry. Avoid
        # rereading every hard-linked grid and decoded texture on every launch.
        catalog = ViewerMaterialCatalog.open(catalog_path, verify_payloads=False)
        expected_compatibility = (
            "exact-diagnostic-evaluator-preview" if diagnostic_preview else "exact"
        )
        if (
            catalog.checkpoint_sha256 != sha256_file(checkpoint_path)
            or catalog.registry_sha256 != sha256_file(registry_path)
            or catalog.checkpoint_compatibility != expected_compatibility
        ):
            raise ValueError(
                "existing ViewerMaterialCatalog was built from another registry/checkpoint"
            )
        return catalog
    if output_root.exists():
        raise ValueError(
            "ViewerMaterialCatalog output root must be absent or contain a valid catalog"
        )

    checkpoint = load_checkpoint(checkpoint_path)
    definition = get_method(checkpoint.method_key)
    compatibility = validate_preview_checkpoint(
        checkpoint, definition.descriptor, diagnostic=diagnostic_preview
    )
    readiness = assess_checkpoint_readiness(
        checkpoint,
        definition.descriptor,
        mode="diagnostic-evaluator" if diagnostic_preview else "formal",
    )
    config = TrainingConfig.from_dict(checkpoint.training_config)
    if str(config.source.get("family_id")) != "mdl.program@1":
        raise ValueError("Metal viewer catalog checkpoint has another source family")
    registry = MdlMetalRegistry.load(registry_path)
    raw_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    raw_by_export = {
        str(item["export_id"]): item for item in raw_registry["opaque_exports"]
    }
    locators = {
        (str(item["locator"]["module"]), str(item["locator"]["export"])): dict(
            item["locator"]
        )
        for item in config.source["materials"]
    }
    if len(locators) != len(config.source["materials"]):
        raise ValueError("checkpoint source list contains duplicate Metal locators")
    selected_records = _select_registry_records(
        registry,
        locators,
        diagnostic_preview=diagnostic_preview,
        limit=limit,
    )
    module_roots = {
        str(locator["module_root"]) for locator in locators.values()
    }
    if len(module_roots) != 1:
        raise ValueError("Metal viewer catalog requires one canonical module root")
    module_root = Path(next(iter(module_roots)))
    if not module_root.is_absolute():
        module_root = PROJECT_ROOT / module_root
    module_root = module_root.resolve()

    family = create_source_family("mdl.program@1")
    bridge = create_mdl_program_provider(module_root)
    snapshots = []
    for record in tqdm(
        selected_records,
        desc="resolve Metal authored snapshots",
        unit="entry",
    ):
        key = (
            str(record.exact_locator["module"]),
            str(record.exact_locator["export"]),
        )
        if key not in locators:
            raise ValueError("checkpoint source list is missing a registry export")
        locator = dict(locators[key])
        locator["module_root"] = str(module_root)
        snapshot = _load_snapshot_with_provider(locator, bridge)
        family.validate_snapshot(snapshot)
        if snapshot.snapshot_id not in checkpoint.source_snapshot_ids:
            raise ValueError("authored Metal snapshot is absent from the checkpoint")
        inspection_root = Path(str(snapshot.editor_metadata["inspection_artifact"]))
        expected_manifest = str(raw_by_export[record.export_id]["artifact_manifest_sha256"])
        if sha256_file(inspection_root / "manifest.json") != expected_manifest:
            raise ValueError("Metal registry inspection artifact identity drifted")
        snapshots.append(snapshot)

    payload = checkpoint.to_payload()
    program_payload = definition.compile_program(payload)
    if diagnostic_preview:
        evaluator_capabilities = int(
            BackendCapability.PREPARE
            | BackendCapability.EVALUATE
            | BackendCapability.ANISOTROPIC_FRAME
        )
        program_payload = replace(
            program_payload, capabilities=evaluator_capabilities
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context = payload["training_config"].get("model_context")
    if not isinstance(context, Mapping):
        raise ValueError("Metal viewer deployment checkpoint has no model_context")
    model = definition.create_trainable(context).to(device)
    definition.restore_training_state(model, payload["model_state"])
    quantize_runtime_model(model)
    adapter = MetalFusedMdlSourceAdapter(tuple(snapshots), device)
    native_assets = adapter.native_assets()
    cooker = MetalAssetCooker(
        model,
        native_assets,
        max_core_texels=262_144,
        encoder_halo=32,
        encoder_batch_tiles=1,
    )
    editor_views = []
    for index, (record, snapshot) in enumerate(
        tqdm(
            zip(selected_records, snapshots),
            total=len(snapshots),
            desc="validate Metal editor contracts",
            unit="entry",
        )
    ):
        try:
            editor_views.append(
                link_parameter_view(
                    definition._editor_view(snapshot, adapter),
                    record.parameters,
                )
            )
        except Exception as error:
            raw = raw_by_export[record.export_id]
            raise ValueError(
                "Metal editor contract failed for "
                f"entry {index + 1}/{len(snapshots)} "
                f"{raw['export_name']} ({record.export_id})"
            ) from error
    with ThreadPoolExecutor(
        max_workers=min(MAX_MDL_COMPILE_WORKERS, len(snapshots))
    ) as executor:
        runtime_artifacts = list(
            tqdm(
                executor.map(
                    lambda snapshot: _compile_reference_program(bridge, snapshot),
                    snapshots,
                ),
                total=len(snapshots),
                desc="compile Metal reference programs",
                unit="entry",
            )
        )
    for artifact in runtime_artifacts:
        artifact.require_runtime_supported()
    phase = checkpoint.phase_name
    checkpoint_sha256 = sha256_file(checkpoint_path)
    target_types = (
        PROJECT_ROOT
        / "external/MDL-SDK-2025.0.0-387700.1252-nt-x86-64"
        / "examples/mdl_sdk/dxr/content/mdl_target_code_types.hlsl"
    )
    renderer_runtime = PROJECT_ROOT / "shaders/ncls/reference_backends/mdl_runtime.slangh"
    if not target_types.is_file() or not renderer_runtime.is_file():
        raise FileNotFoundError("MDL viewer runtime includes are missing")
    staging = output_root.with_name(
        f".{output_root.name}.{uuid.uuid4().hex}.partial"
    )
    objects: dict[str, Path] = {}
    entries: list[dict[str, object] | None] = [None] * len(snapshots)
    try:
        staging.mkdir(parents=True)
        runtime_root = staging / "runtime"
        _copy_or_link(target_types, runtime_root / "mdl_target_code_types.hlsl", objects)
        _copy_or_link(renderer_runtime, runtime_root / "mdl_runtime.slangh", objects)
        asset_indices = [
            adapter.asset_index_for_source(index) for index in range(len(snapshots))
        ]
        work = sorted(
            zip(
                range(len(snapshots)),
                selected_records,
                snapshots,
                editor_views,
                asset_indices,
                runtime_artifacts,
            ),
            key=lambda item: (item[4], item[0]),
        )
        progress = tqdm(
            work,
            total=len(snapshots),
            desc="materialize linked Metal catalog",
            unit="entry",
        )
        current_asset_index: int | None = None
        current_asset: tuple[object, object, dict[str, str]] | None = None
        for index, record, snapshot, editor_view, asset_index, artifact in progress:
            reference_root = staging / "reference" / record.export_id
            declared_hashes = artifact.manifest.get("files_sha256")
            if not isinstance(declared_hashes, Mapping):
                raise ValueError("MDL runtime artifact has no finalized file hash table")
            _materialize_tree(
                artifact.root,
                reference_root,
                objects,
                declared_hashes,
            )

            if current_asset_index != asset_index:
                cooked = cooker.cook_asset(asset_index, mode="encoder-only")
                descriptor = native_assets.descriptors[asset_index]
                address_modes = {
                    domain.domain_id: domain.address_mode
                    for domain in descriptor.domains
                }
                packed_asset = pack_metal_asset(
                    cooked, address_modes=address_modes
                )
                current_asset = (cooked, packed_asset, address_modes)
                current_asset_index = asset_index
            if current_asset is None:
                raise RuntimeError("Metal viewer asset cache was not initialized")
            cooked, packed_asset, address_modes = current_asset
            tensors = adapter.compiler_tensors_for_source(index, device=device)
            with torch.no_grad():
                program_state = model.typed_compiler(tensors)
            # Reuse the method's existing deployment serializers and parity oracle,
            # while keeping the expensive model and native asset cook shared across
            # all 692 viewer entries. This cache is deployment-only and never enters
            # training or changes the evaluator representation.
            definition._deployment_cache = (
                (snapshot.snapshot_id, id(payload)),
                {
                    "model": model,
                    "adapter": adapter,
                    "tensors": tensors,
                    "cooked": cooked,
                    "packed_asset": packed_asset,
                    "program_state": program_state,
                    "address_modes": address_modes,
                },
            )
            asset_payload = definition.compile_asset(snapshot, payload)
            instance = definition.compile_instance(snapshot, payload)
            editor = dict(instance.editor)
            editor["parameter_view"] = deepcopy(editor_view)
            instance_payload = InstancePayload(
                instance.parameters,
                instance.blobs,
                instance.blob_descriptors,
                editor,
                instance.compiler,
            )
            validate_artifact_coverage(
                definition.descriptor,
                MethodArtifactInventory.from_payloads(
                    program_payload,
                    asset_payload,
                    checkpoint_model_state=bool(checkpoint.model_state),
                ),
            )
            validation = dict(definition.package_validation(snapshot, payload))
            validation["checkpoint_step"] = checkpoint.global_step
            validation["checkpoint_readiness"] = readiness.to_dict()
            package_root = staging / "packages" / record.export_id
            manifest = write_scattering_package(
                package_root,
                program_kind="method",
                program_key=definition.descriptor.method_key,
                program_version=definition.descriptor.version,
                program_descriptor_sha256=definition.descriptor.descriptor_sha256,
                runtime_abi=definition.descriptor.runtime_abi,
                source=snapshot,
                program_payload=program_payload,
                asset_payload=asset_payload,
                instance_payload=instance_payload,
                validation=validation,
                provenance={
                    "checkpoint_sha256": checkpoint_sha256,
                    "checkpoint_descriptor_sha256": checkpoint.method_descriptor_sha256,
                    "runtime_descriptor_sha256": definition.descriptor.descriptor_sha256,
                    "checkpoint_compatibility": compatibility,
                    "checkpoint_readiness_mode": readiness.mode,
                    "viewer_catalog_export_id": record.export_id,
                },
                linked_content_store=objects,
            )
            raw = raw_by_export[record.export_id]
            entries[index] = (
                {
                    "export_id": record.export_id,
                    "display_name": str(raw["export_name"]).replace("_", " "),
                    "metal": record.metal,
                    "finish": record.finish,
                    "graph_id": record.graph_id,
                    "texture_set_id": record.texture_set_id,
                    "parameter_schema_id": record.parameter_schema_id,
                    "source_snapshot_id": snapshot.snapshot_id,
                    "artifact_sha256": _artifact_sha256(artifact),
                    "artifact_root": _portable(reference_root, staging),
                    "package_id": manifest.package_id,
                    "package_root": _portable(package_root, staging),
                    "program_id": manifest.program_id,
                    "asset_id": manifest.asset_id,
                    "instance_id": manifest.instance_id,
                    "parameter_view": deepcopy(editor_view),
                }
            )
            progress.set_postfix_str(
                f"{record.metal}/{record.finish} {raw['export_name']}",
                refresh=False,
            )

        if any(entry is None for entry in entries):
            raise RuntimeError("Metal viewer catalog did not materialize every entry")
        finalized_entries = [cast(dict[str, object], entry) for entry in entries]
        document = finalize_catalog_document(
            {
                "schema_name": "ncls.viewer-material-catalog",
                "schema_version": 1,
                "registry": {
                    "identity": registry.identity,
                    "sha256": sha256_file(registry_path),
                    "opaque_entry_count": len(finalized_entries),
                    "rejected_cutout_count": len(registry.rejected_cutout_exports),
                },
                "checkpoint": {
                    "sha256": checkpoint_sha256,
                    "checkpoint_descriptor_sha256": checkpoint.method_descriptor_sha256,
                    "runtime_descriptor_sha256": definition.descriptor.descriptor_sha256,
                    "compatibility": compatibility,
                    "method_key": checkpoint.method_key,
                    "step": checkpoint.global_step,
                    "phase": phase,
                },
                "reference_runtime": {
                    "mdl_sdk": bridge.descriptor.sdk_build,
                    "target_code_types": {
                        "path": "runtime/mdl_target_code_types.hlsl",
                        "sha256": sha256_file(target_types),
                    },
                    "renderer_runtime": {
                        "path": "runtime/mdl_runtime.slangh",
                        "sha256": sha256_file(renderer_runtime),
                    },
                },
                "default_export_id": finalized_entries[0]["export_id"],
                "entries": finalized_entries,
            }
        )
        write_json_atomic(staging / "catalog.json", document)
        ViewerMaterialCatalog.open(
            staging / "catalog.json", verify_payloads=False
        )
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output_root)
    except BaseException:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise
    return ViewerMaterialCatalog.open(catalog_path, verify_payloads=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the 692-entry linked Metal ViewerMaterialCatalog from an "
            "existing checkpoint; this command performs deployment compilation only"
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--diagnostic-limit",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--diagnostic-preview",
        action="store_true",
        help="显式允许未完成但已有联合梯度证据的evaluate-only诊断预览",
    )
    args = parser.parse_args()
    catalog = prepare_metal_catalog(
        args.output_root,
        args.checkpoint,
        args.registry,
        limit=args.diagnostic_limit,
        diagnostic_preview=args.diagnostic_preview,
    )
    print(catalog.source_path)
    print(
        f"entries={len(catalog.entries)} references={len(catalog.entries)} "
        f"programs={len({item.program_id for item in catalog.entries})} "
        f"asset_identities={len({item.asset_id for item in catalog.entries})} "
        f"texture_sets={len({item.texture_set_id for item in catalog.entries})} "
        f"instances={len({item.instance_id for item in catalog.entries})} "
        f"storage=hard-linked step={catalog.checkpoint_step} "
        f"phase={catalog.checkpoint_phase}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
