from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Any, Mapping

from ncls.core.identity import sha256_file, sha256_json, write_json_atomic
from ncls.core.source import SourceSnapshot, create_source_family
from ncls.learning.evaluation_package import (
    CompiledEvaluationPackage,
    compile_evaluation_package,
)
from ncls.learning.models.metal_budgeted_profile import (
    METAL_BUDGETED_DIRECT_PROFILE_ID,
    METAL_BUDGETED_HYBRID_PROFILE_ID,
)
from ncls.learning.training import EvaluationSnapshot, load_evaluation_snapshot
from ncls.paths import PROJECT_ROOT
from ncls.references.mdl import (
    MdlCompiledArtifact,
    MdlProgramProvider,
    create_mdl_program_provider,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/viewer/metal-budgeted-pair"
HANDOFF_FORMAT = "ncls.metal-budgeted-viewer-handoff"
HANDOFF_VERSION = 1
CHECKPOINT_COMPATIBILITY = "exact-diagnostic-evaluator-preview"
_EXPECTED_PROFILES = {
    "hybrid": METAL_BUDGETED_HYBRID_PROFILE_ID,
    "direct": METAL_BUDGETED_DIRECT_PROFILE_ID,
}


def validate_preview_checkpoint(snapshot: EvaluationSnapshot) -> str:
    """只接受具有当前精确实现身份的 evaluator-only checkpoint。"""

    snapshot.require_ready("diagnostic-evaluator")
    return CHECKPOINT_COMPATIBILITY


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
    if expected_sha256 is not None and sha256_file(source) != expected_sha256:
        raise ValueError(f"viewer handoff source hash mismatch: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    link_source = objects.get(digest, source)
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
        raise ValueError(f"viewer handoff target already exists: {target}")
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


def _compile_reference_program(
    provider: MdlProgramProvider,
    snapshot: SourceSnapshot,
) -> MdlCompiledArtifact:
    return provider.compile_snapshot(snapshot)


def _checkpoint_profile(snapshot: EvaluationSnapshot) -> str:
    training_config = snapshot.deployment_payload.get("training_config")
    if not isinstance(training_config, Mapping):
        raise ValueError("Metal budgeted checkpoint has no training_config")
    context = training_config.get("model_context")
    if not isinstance(context, Mapping):
        raise ValueError("Metal budgeted checkpoint has no model_context")
    return str(context.get("profile_id", ""))


def _single_locator(snapshot: EvaluationSnapshot) -> Mapping[str, Any]:
    source = snapshot.source
    if source.get("family_id") != "mdl.program@1":
        raise ValueError("Metal budgeted viewer handoff requires an MDL source")
    materials = source.get("materials")
    if not isinstance(materials, (list, tuple)) or len(materials) != 1:
        raise ValueError("Metal budgeted viewer handoff requires one fixed source material")
    material = materials[0]
    if not isinstance(material, Mapping) or not isinstance(material.get("locator"), Mapping):
        raise ValueError("Metal budgeted checkpoint source locator is invalid")
    return dict(material["locator"])


def _validate_pair(
    hybrid: EvaluationSnapshot,
    direct: EvaluationSnapshot,
) -> tuple[Mapping[str, Any], str]:
    snapshots = {"hybrid": hybrid, "direct": direct}
    for role, snapshot in snapshots.items():
        validate_preview_checkpoint(snapshot)
        if snapshot.public_method_key != "metal":
            raise ValueError(f"{role} checkpoint is not the Metal method")
        profile = _checkpoint_profile(snapshot)
        if profile != _EXPECTED_PROFILES[role]:
            raise ValueError(
                f"{role} checkpoint profile is {profile!r}, expected {_EXPECTED_PROFILES[role]!r}"
            )
    hybrid_locator = _single_locator(hybrid)
    direct_locator = _single_locator(direct)
    if hybrid_locator != direct_locator:
        raise ValueError("hybrid/direct checkpoints do not use the same source locator")
    if hybrid.source_snapshot_ids != direct.source_snapshot_ids:
        raise ValueError("hybrid/direct checkpoints do not use the same source snapshot")
    if len(hybrid.source_snapshot_ids) != 1:
        raise ValueError("Metal budgeted viewer pair must contain one source snapshot")
    if hybrid.global_step != direct.global_step:
        raise ValueError("hybrid/direct checkpoints are not at a common training milestone")
    return hybrid_locator, hybrid.source_snapshot_ids[0]


def _runtime_paths() -> tuple[Path, Path]:
    relative = Path("examples/mdl_sdk/dxr/content/mdl_target_code_types.hlsl")
    candidates = tuple(
        PROJECT_ROOT / "external" / package / relative
        for package in (
            "MDL-SDK-2025.0.0-387700.1252-nt-x86-64",
            "MDL-SDK-2025.0.0-387700.1252-linux-x86-64",
        )
    )
    existing = tuple(path for path in candidates if path.is_file())
    renderer_runtime = PROJECT_ROOT / "shaders/ncls/reference_backends/mdl_runtime.slangh"
    if not existing or not renderer_runtime.is_file():
        raise FileNotFoundError("MDL viewer runtime includes are missing")
    if len({sha256_file(path) for path in existing}) != 1:
        raise ValueError("Windows/Linux MDL target-code headers differ")
    return existing[0], renderer_runtime


def _reference_catalog(
    *,
    staging: Path,
    snapshot: SourceSnapshot,
    artifact: MdlCompiledArtifact,
    display_name: str,
    provider: MdlProgramProvider,
    target_types: Path,
    renderer_runtime: Path,
) -> dict[str, object]:
    asset_id = f"metal-budgeted-{snapshot.snapshot_id[:16]}"
    return {
        "schema_name": "ncls.mdl-viewer-catalog",
        "schema_version": 1,
        "reference_id": "ncls.metal-budgeted-fixed-reference@1",
        "source_material_family_id": "mdl.program@1",
        "formal_executor": "project-mdl-sdk-bridge-to-current-falcor-8",
        "validation_oracle": "falcor2-isolated-not-a-runtime-dependency",
        "mdl_sdk": provider.descriptor.sdk_build,
        "texture_filtering": "explicit-lod0",
        "uv_derivatives_consumed": False,
        "default_asset_id": asset_id,
        "target_code_types": {
            "path": "runtime/mdl_target_code_types.hlsl",
            "sha256": sha256_file(target_types),
        },
        "renderer_runtime": {
            "path": "runtime/mdl_runtime.slangh",
            "sha256": sha256_file(renderer_runtime),
        },
        "assets": [
            {
                "asset_id": asset_id,
                "display_name": display_name,
                "source_snapshot_id": snapshot.snapshot_id,
                "artifact_root": _portable(
                    staging / "reference" / snapshot.snapshot_id, staging
                ),
                "compiled_artifact_sha256": _artifact_sha256(artifact),
            }
        ],
    }


def _package_record(
    role: str,
    evaluation: EvaluationSnapshot,
    compiled: CompiledEvaluationPackage,
    staging: Path,
) -> dict[str, object]:
    return {
        "role": role,
        "profile_id": _checkpoint_profile(evaluation),
        "checkpoint_path": f"checkpoints/{role}.pt",
        "checkpoint_sha256": evaluation.checkpoint_sha256,
        "checkpoint_step": evaluation.global_step,
        "checkpoint_phase": evaluation.phase_name,
        "checkpoint_compatibility": CHECKPOINT_COMPATIBILITY,
        "package_root": _portable(compiled.root, staging),
        "package_id": compiled.manifest.package_id,
        "program_id": compiled.manifest.program_id,
        "asset_id": compiled.manifest.asset_id,
        "instance_id": compiled.manifest.instance_id,
        "capabilities": ["prepare", "evaluate", "anisotropic-frame"],
        "unsupported_capabilities": ["sample", "pdf", "typed-edit"],
    }


def _readme(document: Mapping[str, Any]) -> str:
    packages = {str(item["role"]): item for item in document["packages"]}
    return f"""# Metal budgeted 双模型 Windows 诊断交接

本目录包含同一 source snapshot、共同 step {packages['hybrid']['checkpoint_step']} 的 hybrid 与 direct checkpoint，以及各自精确编译的 `ScatteringPackage@2`。两者仅验证 `prepare/evaluate/anisotropic-frame`，不声明 `sample/pdf` 或 typed edit。

在 NeuralShading 根目录的 Windows PowerShell 中运行：

```powershell
.\\scripts\\launch_metal_viewer.ps1 -Handoff \"{document['handoff_hint']}\" -Comparison ReferenceVsHybrid
.\\scripts\\launch_metal_viewer.ps1 -Handoff \"{document['handoff_hint']}\" -Comparison ReferenceVsDirect -SkipBuild
.\\scripts\\launch_metal_viewer.ps1 -Handoff \"{document['handoff_hint']}\" -Comparison HybridVsDirect -SkipBuild
```

`ReferenceVs*` 左侧使用精确 MDL source path tracing，右侧使用 neural deferred evaluator；`HybridVsDirect` 两侧都使用 deferred。Viewer 首次加载会重新执行 D3D12 GPU parity，失败时不会静默回退。

- hybrid profile：`{packages['hybrid']['profile_id']}`，package `{packages['hybrid']['package_id']}`
- direct profile：`{packages['direct']['profile_id']}`，package `{packages['direct']['package_id']}`
- compatibility：`{CHECKPOINT_COMPATIBILITY}`
"""


def prepare_metal_catalog(
    output_root: Path,
    hybrid_checkpoint: Path,
    direct_checkpoint: Path,
) -> Mapping[str, Any]:
    output_root = output_root.resolve()
    hybrid_checkpoint = hybrid_checkpoint.resolve()
    direct_checkpoint = direct_checkpoint.resolve()
    handoff_path = output_root / "handoff.json"
    if handoff_path.is_file():
        existing = json.loads(handoff_path.read_text(encoding="utf-8"))
        expected = {
            "hybrid": sha256_file(hybrid_checkpoint),
            "direct": sha256_file(direct_checkpoint),
        }
        actual = {
            str(item["role"]): str(item["checkpoint_sha256"])
            for item in existing.get("packages", [])
        }
        if actual != expected:
            raise ValueError("existing viewer handoff was built from other checkpoints")
        return existing
    if output_root.exists():
        raise ValueError("viewer handoff output root must be absent or complete")

    evaluations = {
        "hybrid": load_evaluation_snapshot(hybrid_checkpoint, map_location="cpu"),
        "direct": load_evaluation_snapshot(direct_checkpoint, map_location="cpu"),
    }
    locator, expected_snapshot_id = _validate_pair(
        evaluations["hybrid"], evaluations["direct"]
    )
    family = create_source_family("mdl.program@1")
    snapshot = family.load_snapshot(locator)
    family.validate_snapshot(snapshot)
    if snapshot.snapshot_id != expected_snapshot_id:
        raise ValueError("resolved MDL source no longer matches the checkpoints")
    module_root = Path(str(locator["module_root"]))
    if not module_root.is_absolute():
        module_root = PROJECT_ROOT / module_root
    provider = create_mdl_program_provider(module_root.resolve())
    artifact = _compile_reference_program(provider, snapshot)
    artifact.require_runtime_supported()
    declared_hashes = artifact.manifest.get("files_sha256")
    if not isinstance(declared_hashes, Mapping):
        raise ValueError("MDL runtime artifact has no finalized file hash table")
    target_types, renderer_runtime = _runtime_paths()

    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.partial")
    objects: dict[str, Path] = {}
    try:
        staging.mkdir(parents=True)
        _copy_or_link(
            target_types, staging / "runtime/mdl_target_code_types.hlsl", objects
        )
        _copy_or_link(
            renderer_runtime, staging / "runtime/mdl_runtime.slangh", objects
        )
        _materialize_tree(
            artifact.root,
            staging / "reference" / snapshot.snapshot_id,
            objects,
            declared_hashes,
        )
        compiled = {
            role: compile_evaluation_package(
                evaluation,
                staging / "packages" / role,
                material_index=0,
                readiness_mode="diagnostic-evaluator",
            )
            for role, evaluation in evaluations.items()
        }
        checkpoints_root = staging / "checkpoints"
        _copy_or_link(
            hybrid_checkpoint,
            checkpoints_root / "hybrid.pt",
            objects,
            expected_sha256=evaluations["hybrid"].checkpoint_sha256,
        )
        _copy_or_link(
            direct_checkpoint,
            checkpoints_root / "direct.pt",
            objects,
            expected_sha256=evaluations["direct"].checkpoint_sha256,
        )
        reference_catalog = _reference_catalog(
            staging=staging,
            snapshot=snapshot,
            artifact=artifact,
            display_name="Tungsten budgeted diagnostic source",
            provider=provider,
            target_types=target_types,
            renderer_runtime=renderer_runtime,
        )
        write_json_atomic(staging / "catalog.json", reference_catalog)
        document: dict[str, Any] = {
            "format_name": HANDOFF_FORMAT,
            "format_version": HANDOFF_VERSION,
            "checkpoint_compatibility": CHECKPOINT_COMPATIBILITY,
            "source_family_id": snapshot.family_id,
            "source_snapshot_id": snapshot.snapshot_id,
            "reference_catalog": "catalog.json",
            "bundle_root": "packages",
            "packages": [
                _package_record(role, evaluations[role], compiled[role], staging)
                for role in ("hybrid", "direct")
            ],
            "handoff_hint": _portable(output_root / "handoff.json", PROJECT_ROOT),
        }
        write_json_atomic(staging / "handoff.json", document)
        (staging / "README.md").write_text(_readme(document), encoding="utf-8")
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output_root)
    except BaseException:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise
    return json.loads(handoff_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "从共同里程碑的 hybrid/direct checkpoint 生成固定 source、双 package 的 "
            "Windows evaluator-only 诊断交接"
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--hybrid-checkpoint", type=Path, required=True)
    parser.add_argument("--direct-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    document = prepare_metal_catalog(
        args.output_root,
        args.hybrid_checkpoint,
        args.direct_checkpoint,
    )
    print((args.output_root.resolve() / "handoff.json"))
    for package in document["packages"]:
        print(
            f"{package['role']}: profile={package['profile_id']} "
            f"package={package['package_id']} step={package['checkpoint_step']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
