from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ncls.core.identity import sha256_file, write_json_atomic
from ncls.core.source import create_source_family
from ncls.paths import PROJECT_ROOT
from ncls.references.mdl import create_mdl_program_provider
from ncls.viewer.material_catalog import source_catalog_document, source_catalog_entry, ViewerMaterialCatalog


ASSET_IDS = (
    "carpaint-shifting-flakes",
    "copper-antique-brushed-patinated",
    "aluminum-scratched",
    "ceramic-tiles-glazed-versailles",
    "velvet",
    "wood-tiles-pine-mosaic",
)

DISPLAY_NAMES = {
    "carpaint-shifting-flakes": "Car paint — shifting flakes",
    "copper-antique-brushed-patinated": "Copper — antique brushed patina",
    "aluminum-scratched": "Aluminum — scratched",
    "ceramic-tiles-glazed-versailles": "Ceramic — glazed Versailles",
    "velvet": "Velvet",
    "wood-tiles-pine-mosaic": "Pine — wood tile mosaic",
}


def _portable(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def prepare_catalog(output: Path, default_asset_id: str = ASSET_IDS[0]) -> dict[str, object]:
    if default_asset_id not in ASSET_IDS:
        raise ValueError(f"unknown default MDL viewer asset: {default_asset_id}")
    manifest_path = PROJECT_ROOT / "references/mdl-vmaterials2-v1/assets.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {str(item["asset_id"]): item for item in manifest["assets"]}
    if any(asset_id not in records for asset_id in ASSET_IDS):
        raise RuntimeError("MDL source manifest is missing a viewer asset")
    module_root = (
        PROJECT_ROOT
        / "assets/source-materials/mdl-vmaterials2/2.4.0"
        / str(manifest["module_root"])
    ).resolve()
    family = create_source_family("mdl.program@1")
    bridge = create_mdl_program_provider(module_root)
    snapshots = {}
    artifacts = {}
    for asset_id in ASSET_IDS:
        record = records[asset_id]
        snapshot = family.load_snapshot(
            {
                "kind": "mdl-export",
                "module_root": str(module_root),
                "module": str(record["module"]),
                "export": str(record["export"]),
                "pack_id": "nvidia.vmaterials",
                "pack_version": "2.4.0",
            }
        )
        snapshots[asset_id] = snapshot
        artifacts[asset_id] = bridge.compile_snapshot(snapshot)
    target_types = (
        PROJECT_ROOT
        / "external/MDL-SDK-2025.0.0-387700.1252-nt-x86-64"
        / "examples/mdl_sdk/dxr/content/mdl_target_code_types.hlsl"
    )
    renderer_runtime = PROJECT_ROOT / "shaders/ncls/reference_backends/mdl_runtime.slangh"
    base = output.resolve().parent
    base.mkdir(parents=True, exist_ok=True)
    runtime = base / "runtime"
    runtime.mkdir(exist_ok=True)
    shutil.copy2(target_types, runtime / "mdl_target_code_types.hlsl")
    shutil.copy2(renderer_runtime, runtime / "mdl_runtime.slangh")
    entries = []
    for asset_id in ASSET_IDS:
        snapshot, artifact = snapshots[asset_id], artifacts[asset_id]
        artifact_root = base / "reference" / snapshot.snapshot_id
        shutil.copytree(artifact.root, artifact_root, dirs_exist_ok=True)
        entries.append(source_catalog_entry(
            export_id=snapshot.snapshot_id, display_name=DISPLAY_NAMES[asset_id],
            source_snapshot_id=snapshot.snapshot_id, artifact_sha256=artifact.artifact_sha256,
            artifact_root=_portable(artifact_root, base),
        ))
    document = source_catalog_document(
        mdl_sdk="2025.0.0-387700.1252",
        target_code_types={"path": "runtime/mdl_target_code_types.hlsl", "sha256": sha256_file(target_types)},
        renderer_runtime={"path": "runtime/mdl_runtime.slangh", "sha256": sha256_file(renderer_runtime)},
        default_export_id=snapshots[default_asset_id].snapshot_id, entries=entries,
    )
    write_json_atomic(output.resolve(), document)
    ViewerMaterialCatalog.open(output)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the formal MDL catalog consumed by NclsViewer")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "build/mdl-reference/viewer/catalog.json",
    )
    parser.add_argument("--default-asset", choices=ASSET_IDS, default=ASSET_IDS[0])
    args = parser.parse_args()
    document = prepare_catalog(args.output, args.default_asset)
    print(args.output.resolve())
    print(f"assets={len(document['entries'])} default={document['default_export_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
