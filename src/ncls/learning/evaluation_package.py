from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ncls.bundle import ScatteringPackageManifest, write_scattering_package
from ncls.core.source import SourceSnapshot, create_source_family
from ncls.learning.conformance import MethodArtifactInventory, validate_artifact_coverage
from ncls.learning.methods import get_method
from ncls.learning.training.checkpoint import TrainingCheckpoint
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
    evaluation: TrainingCheckpoint,
    output: Path | str,
    *,
    material_index: int,
    checkpoint_sha256: str | None = None,
) -> CompiledEvaluationPackage:
    plugin = get_method(evaluation.method_key)
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
    payload = plugin.prepare_export(snapshot, evaluation.model_payload)
    training_config = payload.get("training_config")
    model_context = (
        training_config.get("model_context")
        if isinstance(training_config, Mapping)
        else None
    )
    profile_id = (
        str(model_context.get("profile_id", ""))
        if isinstance(model_context, Mapping)
        else ""
    )
    runtime = plugin.compile_program(payload)
    asset = plugin.compile_asset(snapshot, payload)
    instance = plugin.compile_instance(snapshot, payload)
    validate_artifact_coverage(
        plugin.descriptor,
        MethodArtifactInventory.from_payloads(
            runtime,
            asset,
            checkpoint_model_state=bool(payload["model_state"]),
        ),
    )
    validation = dict(plugin.package_validation(snapshot, payload))
    validation["checkpoint_step"] = evaluation.global_step
    validation["training_diagnostics"] = {"phase": evaluation.phase_name, "gradient_coverage": dict(evaluation.gradient_coverage)}
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
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_profile_id": profile_id,
            "training_method": dict(evaluation.model_payload.get("training_method", {})),
        },
    )
    return CompiledEvaluationPackage(
        root,
        manifest,
        snapshot,
        _source_material_path(locator, snapshot),
    )


__all__ = ["CompiledEvaluationPackage", "compile_evaluation_package"]
