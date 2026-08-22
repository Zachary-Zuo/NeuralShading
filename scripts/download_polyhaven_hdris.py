from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "NeuralShadingResearch/0.1"


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_manifest(manifest_path: Path, output_dir: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for asset in manifest["assets"]:
        target = output_dir / f"{asset['id']}_1k.exr"
        if target.exists() and target.stat().st_size == asset["size"] and _md5(target) == asset["md5"]:
            print(f"verified {target.name}")
            continue
        request = Request(asset["url"], headers={"User-Agent": USER_AGENT})
        with urlopen(request) as response, target.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        if target.stat().st_size != asset["size"] or _md5(target) != asset["md5"]:
            raise RuntimeError(f"download checksum mismatch: {target}")
        print(f"downloaded {target.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the frozen Poly Haven HDRI oracle manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "docs" / "manifests" / "polyhaven_hdri_v0.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "hdris" / "polyhaven_1k",
    )
    args = parser.parse_args()
    download_manifest(args.manifest, args.output)


if __name__ == "__main__":
    main()
