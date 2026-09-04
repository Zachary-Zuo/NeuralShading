from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ncls.bundle import ScatteringPackageManifest, write_scattering_package
from ncls.core.source import SourceSnapshot, create_source_family
from ncls.learning.conformance import MethodArtifactInventory, validate_artifact_coverage
from ncls.learning.methods import get_method_plugin
from ncls.learning.training import CheckpointReadinessMode, EvaluationSnapshot
from ncls.paths import PROJECT_ROOT


@dataclass(frozen=True)
class CompiledEvaluationPackage:
    root: Path
    manifest: ScatteringPackageManifest
    source_snapshot: SourceSnapshot
    source_material_path: Path | None


def _source_material_path(
    locator: Mapping[str, Any], snapshot: SourceSnapshot
) -> Path | None:
    raw = locator.get("path")
    if raw is not None:
        path = Path(str(raw))
        return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    native = snapshot.native_object
    document = getattr(native, "document_path", None)
    if document is not None:
        return Path(document).resolve()
    return None


def compile_evaluation_package(
    evaluation: EvaluationSnapshot,
    output: Path | str,
    *,
    material_index: int,
    readiness_mode: CheckpointReadinessMode,
) -> CompiledEvaluationPackage:
    readiness = dict(evaluation.require_ready(readiness_mode))
    plugin = get_method_plugin(evaluation.public_method_key)
    if plugin.descriptor.method_key != evaluation.implementation_key:
        raise ValueError("evaluation snapshot method implementation drifted")
    source = evaluation.source
    materials = source.get("materials")
    if not isinstance(materials, (list, tuple)) or not 0 <= material_index < len(materials):
        raise ValueError("evaluation material index is outside the checkpoint source list")
    material = materials[material_index]
    if not isinstance(material, Mapping) or not isinstance(material.get("locator"), Mapping):
        raise ValueError("evaluation checkpoint source locator is invalid")
    locator = dict(material["locator"])
    family = create_source_family(str(source["family_id"]))
    snapshot = family.load_snapshot(locator)
    family.validate_snapshot(snapshot)
    if snapshot.snapshot_id not in evaluation.source_snapshot_ids:
        raise ValueError("evaluation source snapshot does not occur in the checkpoint")
    payload = evaluation.deployment_payload
    runtime = plugin.deployment.compile_program(payload)
    asset = plugin.deployment.compile_asset(snapshot, payload)
    instance = plugin.deployment.compile_instance(snapshot, payload)
    validate_artifact_coverage(
        plugin.descriptor,
        MethodArtifactInventory.from_payloads(
            runtime,
            asset,
            checkpoint_model_state=bool(payload["model_state"]),
        ),
    )
    validation = dict(plugin.deployment.package_validation(snapshot, payload))
    validation["checkpoint_step"] = evaluation.global_step
    validation["checkpoint_readiness"] = readiness
    root = Path(output).resolve()
    manifest = write_scattering_package(
        root,
        program_kind="method",
        program_key=plugin.descriptor.method_key,
        program_version=plugin.descriptor.version,
        program_descriptor_sha256=plugin.descriptor.descriptor_sha256,
        runtime_abi=plugin.descriptor.runtime_abi,
        source=snapshot,
        program_payload=runtime,
        asset_payload=asset,
        instance_payload=instance,
        validation=validation,
        provenance={
            "checkpoint_sha256": evaluation.checkpoint_sha256,
            "checkpoint_readiness_mode": readiness_mode,
            "checkpoint_legacy_v4": evaluation.legacy_v4,
        },
    )
    return CompiledEvaluationPackage(
        root,
        manifest,
        snapshot,
        _source_material_path(locator, snapshot),
    )


__all__ = ["CompiledEvaluationPackage", "compile_evaluation_package"]
