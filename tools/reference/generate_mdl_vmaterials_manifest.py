from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ncls.core.identity import sha256_file
from ncls.paths import PROJECT_ROOT
from ncls.references.mdl import MDL_SDK_BUILD, MdlCompiledArtifact, MdlSdkCompilerBridge
from ncls.source_materials.mdl import snapshot_from_mdl_artifact


ARCHIVE = {
    "url": "https://d4i3qtqj3r0z5.cloudfront.net/vMaterials_2_4_0_NVD%40020240.zip",
    "file": "vMaterials_2_4_0_NVD@020240.zip",
    "content_length": 2_220_534_625,
    "sha256": "ab8116e1944c03ae622b2637939510eca9c522a07fd701cf91948fa54e194204",
    "etag": '"719019a683a90c3081489351984b0735-265"',
    "last_modified": "Fri, 12 Jul 2024 13:31:48 GMT",
}

ASSETS = (
    (
        "carpaint-shifting-flakes",
        "color-shifting-car-paint",
        "::vMaterials_2::Paint::Carpaint::Carpaint_Shifting_Flakes",
        "Carpaint_Shifting_Flakes",
    ),
    (
        "copper-antique-brushed-patinated",
        "brushed-patinated-metal",
        "::vMaterials_2::Metal::Copper_Antique_Brushed_Patinated",
        "Copper_Antique_Brushed_Patinated",
    ),
    (
        "aluminum-scratched",
        "micro-scratched-metal",
        "::vMaterials_2::Metal::Aluminum_Scratched",
        "Aluminum_Scratched",
    ),
    (
        "ceramic-tiles-glazed-versailles",
        "glazed-ceramic",
        "::vMaterials_2::Ceramic::Ceramic_Tiles_Glazed_Versailles",
        "Ceramic_Tiles_Glazed_Versailles",
    ),
    (
        "velvet",
        "velvet-sheen",
        "::vMaterials_2::Fabric::Velvet",
        "Velvet",
    ),
    (
        "wood-tiles-pine-mosaic",
        "nvidia-pipeline-correspondence",
        "::vMaterials_2::Wood::Wood_Tiles_Pine",
        "Wood_Tiles_Pine_Mosaic",
    ),
)


def _relative_resource(module_root: Path, filename: str) -> str:
    path = Path(filename).resolve()
    try:
        return path.relative_to(module_root).as_posix()
    except ValueError as error:
        raise ValueError(f"MDL resource escapes module root: {path}") from error


def _asset_record(
    module_root: Path,
    artifact_root: Path,
    asset_id: str,
    coverage_role: str,
    module: str,
    export_name: str,
) -> dict[str, Any]:
    artifact = MdlCompiledArtifact.load(artifact_root / asset_id)
    if artifact.manifest["module"] != module:
        raise ValueError(f"artifact module mismatch for {asset_id}")
    snapshot = snapshot_from_mdl_artifact(
        artifact,
        module_root,
        pack_id="nvidia.vmaterials",
        pack_version="2.4.0",
    )
    payload = json.loads(snapshot.native_payload.decode("utf-8"))
    exact_export = str(payload["export"])
    if f"::{export_name}(" not in exact_export:
        raise ValueError(f"artifact export mismatch for {asset_id}: {exact_export}")

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
                "dimensions": [
                    int(texture["width"]),
                    int(texture["height"]),
                    int(texture["depth"]),
                ],
            }
        )

    return {
        "asset_id": asset_id,
        "coverage_role": coverage_role,
        "module": module,
        "export": exact_export,
        "mdl_language": str(payload["mdl_language"]),
        "source_snapshot_id": snapshot.snapshot_id,
        "source_modules": [
            {
                "module": str(item["name"]),
                "path": str(item["path"]),
                "sha256": snapshot.resource_hashes[str(item["path"])],
            }
            for item in artifact.manifest.get("source_modules", [])
        ],
        "parameters": payload["arguments"],
        "resource_hashes": dict(sorted(snapshot.resource_hashes.items())),
        "runtime_capability_audit": {
            "argument_block_bytes": int(artifact.manifest["argument_block"]["size"]),
            "ro_data_segment_bytes": [
                int(segment["size"]) for segment in artifact.manifest.get("ro_data", [])
            ],
            "df_handle_count": int(artifact.manifest["df_handle_count"]),
            "textures": textures,
            "measured_bsdf": False,
            "light_profile": False,
            "surface_bsdf_evaluate": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module-root",
        type=Path,
        default=PROJECT_ROOT / "assets/source-materials/mdl-vmaterials2/2.4.0/Materials",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT / "build/mdl-reference/shortlist-audit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "references/mdl-vmaterials2-v1/assets.json",
    )
    parser.add_argument(
        "--refresh-artifacts",
        action="store_true",
        help="用锁定 MDL SDK bridge 在全新的 artifact root 中重新 discovery/compile 六个材质",
    )
    args = parser.parse_args()
    module_root = args.module_root.resolve()
    artifact_root = args.artifact_root.resolve()
    if not module_root.is_dir():
        raise FileNotFoundError("vMaterials module root is required")
    if args.refresh_artifacts:
        if artifact_root.exists() and any(artifact_root.iterdir()):
            raise ValueError("refresh artifact root must be absent or empty")
        artifact_root.mkdir(parents=True, exist_ok=True)
        bridge = MdlSdkCompilerBridge(module_root)
        for asset_id, _, module, export_name in ASSETS:
            bridge.inspect(module, export_name, output=artifact_root / asset_id)
    if not artifact_root.is_dir():
        raise FileNotFoundError("vMaterials shortlist artifacts are required")
    archive = PROJECT_ROOT / "assets" / str(ARCHIVE["file"])
    if archive.is_file():
        if archive.stat().st_size != ARCHIVE["content_length"] or sha256_file(archive) != ARCHIVE["sha256"]:
            raise ValueError("local vMaterials archive differs from the frozen official identity")

    manifest = {
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
        "assets": [
            _asset_record(module_root, artifact_root, *item)
            for item in ASSETS
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
