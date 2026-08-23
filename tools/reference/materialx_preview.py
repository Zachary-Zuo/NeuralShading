from __future__ import annotations

import argparse
from pathlib import Path

from ncls.source_materials import MaterialXReference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用锁定的 MaterialXView 渲染 Poly Haven 原始材质")
    parser.add_argument("asset_ids", nargs="*", help="材质 ID；省略时渲染 manifest 中的全部材质")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--environment-samples", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reference-previews/materialx-polyhaven"))
    parser.add_argument(
        "--viewer",
        type=Path,
        default=Path("build/materialx-reference/bin/Release/MaterialXView.exe"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    reference = MaterialXReference(
        root / "external/MaterialX",
        root / "assets/source-materials/materialx-polyhaven/v1",
        root / "references/materialx-polyhaven-v1/assets.json",
    )
    asset_ids = tuple(args.asset_ids) or reference.catalog.asset_ids
    unknown = sorted(set(asset_ids) - set(reference.catalog.asset_ids))
    if unknown:
        raise SystemExit(f"未知材质 ID：{', '.join(unknown)}")
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    viewer = (root / args.viewer).resolve() if not args.viewer.is_absolute() else args.viewer
    for asset_id in asset_ids:
        output = output_dir / f"{asset_id}.png"
        reference.render_preview(
            asset_id,
            viewer,
            output,
            width=args.width,
            height=args.height,
            environment_samples=args.environment_samples,
        )
        print(f"{asset_id}: {output}")


if __name__ == "__main__":
    main()
