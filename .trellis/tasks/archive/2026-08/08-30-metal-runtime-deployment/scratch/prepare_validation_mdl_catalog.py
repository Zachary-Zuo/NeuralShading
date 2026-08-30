from __future__ import annotations

import argparse
from pathlib import Path

from ncls.core.identity import sha256_file, write_json_atomic
from ncls.core.source import create_source_family
from ncls.paths import PROJECT_ROOT
from ncls.references.mdl import create_mdl_program_provider


MODULE = "::vMaterials_2::Metal::Steel_Painted_Cracked"
EXPORT = (
    "::vMaterials_2::Metal::Steel_Painted_Cracked::"
    "Steel_Painted_Russet_Cracked_Dirty(bool,color,float,float,float,float,float,float,"
    "bool,float,float,float,float2,float,float2,int,bool,float,bool)"
)


def _portable(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="为 Metal runtime deployment 生成一次性、同 locator 的 MDL viewer reference catalog"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    module_root = (
        PROJECT_ROOT / "assets/source-materials/mdl-vmaterials2/2.4.0/Materials"
    ).resolve()
    snapshot = create_source_family("mdl.program@1").load_snapshot(
        {
            "kind": "mdl-export",
            "module_root": str(module_root),
            "module": MODULE,
            "export": EXPORT,
            "pack_id": "nvidia.vmaterials2",
            "pack_version": "2.4.0",
        }
    )
    expected_snapshot = "3930c77f97b4514f29e19b671702ebfff415e7fe6af98ce40349c19e9b3cdbb4"
    if snapshot.snapshot_id != expected_snapshot:
        raise RuntimeError(
            f"validation source snapshot drift: {snapshot.snapshot_id} != {expected_snapshot}"
        )
    artifact = create_mdl_program_provider(module_root).compile_snapshot(snapshot)

    target_types = (
        PROJECT_ROOT
        / "external/MDL-SDK-2025.0.0-387700.1252-nt-x86-64"
        / "examples/mdl_sdk/dxr/content/mdl_target_code_types.hlsl"
    )
    renderer_runtime = PROJECT_ROOT / "shaders/ncls/reference_backends/mdl_runtime.slangh"
    base = output.parent
    document = {
        "schema_name": "ncls.mdl-viewer-catalog",
        "schema_version": 1,
        "reference_id": "ncls.mdl-vmaterials2@1",
        "source_material_family_id": "mdl.program@1",
        "formal_executor": "project-mdl-sdk-bridge-to-current-falcor-8",
        "validation_oracle": "falcor2-isolated-not-a-runtime-dependency",
        "mdl_sdk": "2025.0.0-387700.1252",
        "texture_filtering": "explicit-lod0",
        "uv_derivatives_consumed": False,
        "default_asset_id": "steel-painted-russet-cracked-dirty",
        "target_code_types": {
            "path": _portable(target_types, base),
            "sha256": sha256_file(target_types),
        },
        "renderer_runtime": {
            "path": _portable(renderer_runtime, base),
            "sha256": sha256_file(renderer_runtime),
        },
        "assets": [
            {
                "asset_id": "steel-painted-russet-cracked-dirty",
                "display_name": "Steel — painted russet, cracked and dirty",
                "source_snapshot_id": snapshot.snapshot_id,
                "artifact_root": _portable(artifact.root, base),
                "compiled_artifact_sha256": artifact.artifact_sha256,
            }
        ],
    }
    write_json_atomic(output, document)
    print(output)
    print(f"snapshot={snapshot.snapshot_id}")
    print(f"artifact={artifact.artifact_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
