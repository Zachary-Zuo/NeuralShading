"""为当前模型导出和图像评估准备原生 source reference。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
import platform

from ncls.core.identity import sha256_file, sha256_json
from ncls.paths import PROJECT_ROOT


def prepare_source_reference(compiled, output: Path, locator) -> Path:
    if compiled.source_material_path is not None:
        return compiled.source_material_path
    # MDL 原生 source 通过 viewer 的公共 catalog 呈现。
    from ncls.references.mdl import create_mdl_program_provider
    from ncls.viewer.material_catalog import source_catalog_document, source_catalog_entry

    output.mkdir(parents=True, exist_ok=True)
    snapshot = compiled.source_snapshot
    module_root = Path(str(locator["module_root"]))
    if not module_root.is_absolute():
        module_root = PROJECT_ROOT / module_root
    provider = create_mdl_program_provider(module_root)
    artifact = provider.compile_snapshot(snapshot)
    artifact.require_runtime_supported()
    reference = output / "reference"
    shutil.copytree(artifact.root, reference, dirs_exist_ok=True)
    runtime = output / "runtime"
    runtime.mkdir(exist_ok=True)
    target_types = PROJECT_ROOT / "external" / ("MDL-SDK-2025.0.0-387700.1252-nt-x86-64" if platform.system() == "Windows" else "MDL-SDK-2025.0.0-387700.1252-linux-x86-64") / "examples/mdl_sdk/dxr/content/mdl_target_code_types.hlsl"
    renderer = PROJECT_ROOT / "shaders/ncls/reference_backends/mdl_runtime.slangh"
    shutil.copyfile(target_types, runtime / target_types.name)
    shutil.copyfile(renderer, runtime / renderer.name)
    document = source_catalog_document(
        mdl_sdk=provider.descriptor.sdk_build,
        target_code_types={"path": f"runtime/{target_types.name}", "sha256": sha256_file(target_types)},
        renderer_runtime={"path": f"runtime/{renderer.name}", "sha256": sha256_file(renderer)},
        default_export_id=snapshot.snapshot_id,
        entries=[source_catalog_entry(
            export_id=snapshot.snapshot_id, display_name="训练材质",
            source_snapshot_id=snapshot.snapshot_id,
            artifact_sha256=sha256_json({"manifest.json": sha256_file(artifact.root / "manifest.json"), **artifact.manifest["files_sha256"]}),
            artifact_root="reference",
        )],
    )
    path = output / "catalog.json"
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
