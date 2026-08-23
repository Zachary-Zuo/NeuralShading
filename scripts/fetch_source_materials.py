from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping
import urllib.request
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MATERIAL_ROOT = PROJECT_ROOT / "data" / "source-materials"
USER_AGENT = "NeuralShading source-material importer/1.0"


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, *, expected_size: int, md5: str) -> None:
    if destination.is_file():
        if destination.stat().st_size == expected_size and _hash(destination, "md5") == md5:
            print(f"已存在并通过校验：{destination}", flush=True)
            return
        raise RuntimeError(f"目标文件已存在但大小或 MD5 不匹配：{destination}")

    partial = destination.with_suffix(destination.suffix + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise RuntimeError(f"断点文件大于发布文件：{partial}")
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", response.getcode())
        if offset and status != 206:
            print("服务器未接受 Range，从头重新下载 partial 文件。", flush=True)
            offset = 0
        mode = "ab" if offset and status == 206 else "wb"
        destination.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        last_report = started
        with partial.open(mode) as stream:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                offset += len(chunk)
                now = time.monotonic()
                if now - last_report >= 5.0:
                    elapsed = max(now - started, 1e-6)
                    print(
                        f"下载 {offset / 2**30:.2f}/{expected_size / 2**30:.2f} GiB "
                        f"({offset / expected_size:.1%}, {(offset / 2**20) / elapsed:.1f} MiB/s)",
                        flush=True,
                    )
                    last_report = now
    if partial.stat().st_size != expected_size:
        raise RuntimeError(f"下载大小不匹配：expected={expected_size}, actual={partial.stat().st_size}")
    actual_md5 = _hash(partial, "md5")
    if actual_md5 != md5:
        raise RuntimeError(f"下载 MD5 不匹配：expected={md5}, actual={actual_md5}")
    os.replace(partial, destination)
    print(f"下载完成并通过 MD5：{destination}", flush=True)


def _extract_zip(archive: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    extracted = 0
    with zipfile.ZipFile(archive) as source:
        for index, member in enumerate(source.infolist(), start=1):
            target = (destination / member.filename).resolve()
            if destination_root != target and destination_root not in target.parents:
                raise RuntimeError(f"ZIP member 逃出目标目录：{member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.stat().st_size == member.file_size:
                extracted += 1
                continue
            with source.open(member) as input_stream, target.open("wb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream, length=8 * 1024 * 1024)
            extracted += 1
            if index % 20 == 0:
                print(f"解压进度 {index}/{len(source.infolist())}", flush=True)
    return extracted


def fetch_merl() -> None:
    package = _load_json(PROJECT_ROOT / "references" / "merl-brdf-v1" / "reference.json")
    asset = package["source_assets"][0]
    target = SOURCE_MATERIAL_ROOT / "merl-brdf" / "v1"
    archive = target / str(asset["archive_file"])
    _download(
        str(asset["download_url"]),
        archive,
        expected_size=int(asset["size"]),
        md5=str(asset["md5"]),
    )
    expanded = target / "expanded"
    member_count = _extract_zip(archive, expanded)
    database_root = expanded / "BRDFDatabase"
    tables = []
    for table in sorted((database_root / "brdfs").glob("*.binary")):
        tables.append(
            {
                "material_id": table.stem,
                "table_uri": table.relative_to(target).as_posix(),
                "size": table.stat().st_size,
                "sha256": _hash(table, "sha256"),
            }
        )
    if len(tables) != 100:
        raise RuntimeError(f"MERL archive should contain 100 BRDF tables, got {len(tables)}")
    material_index = {
        "schema_name": "ncls.merl-material-index",
        "schema_version": 1,
        "source_record": "zenodo:8101681",
        "license": "CC-BY-SA-4.0",
        "parameterization": "rusinkiewicz-half-difference-official-nearest-index",
        "channel_scale": [1.0 / 1500.0, 1.15 / 1500.0, 1.66 / 1500.0],
        "materials": tables,
    }
    material_index_path = PROJECT_ROOT / "references" / "merl-brdf-v1" / "materials.json"
    material_index_path.write_text(
        json.dumps(material_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    marker = {
        "schema_name": "ncls.source-material-import",
        "schema_version": 1,
        "asset_set_id": asset["asset_set_id"],
        "source_url": asset["download_url"],
        "archive": {
            "path": archive.name,
            "size": archive.stat().st_size,
            "md5": _hash(archive, "md5"),
        },
        "expanded_path": "expanded",
        "expanded_file_count": member_count,
        "material_count": len(tables),
        "material_index": "references/merl-brdf-v1/materials.json",
        "material_index_sha256": _hash(material_index_path, "sha256"),
    }
    (target / "complete.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"MERL source material ready: {target}", flush=True)


def fetch_polyhaven() -> None:
    manifest_path = PROJECT_ROOT / "references" / "materialx-polyhaven-v1" / "assets.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema_name") != "ncls.polyhaven-materialx-assets":
        raise ValueError("unsupported Poly Haven asset manifest")
    target = SOURCE_MATERIAL_ROOT / "materialx-polyhaven" / "v1"
    target_root = target.resolve()
    file_count = 0
    for asset in manifest["assets"]:
        for record in asset["files"]:
            destination = (target / str(record["path"])).resolve()
            if target_root != destination and target_root not in destination.parents:
                raise RuntimeError(f"asset path escapes target directory: {record['path']}")
            _download(
                str(record["source_url"]),
                destination,
                expected_size=int(record["size"]),
                md5=str(record["md5"]),
            )
            file_count += 1
    marker = {
        "schema_name": "ncls.source-material-import",
        "schema_version": 1,
        "asset_set_id": "polyhaven.materialx-curated@1",
        "manifest": "references/materialx-polyhaven-v1/assets.json",
        "manifest_sha256": _hash(manifest_path, "sha256"),
        "file_count": file_count,
        "total_size": int(manifest["total_size"]),
        "license": manifest["license"],
        "api_credit": manifest["api_credit"],
    }
    target.mkdir(parents=True, exist_ok=True)
    (target / "complete.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Poly Haven MaterialX source materials ready: {target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="下载并校验固定的原始源材质资产。")
    parser.add_argument("family", choices=("merl", "polyhaven"))
    args = parser.parse_args()
    if args.family == "merl":
        fetch_merl()
    elif args.family == "polyhaven":
        fetch_polyhaven()
    else:
        raise AssertionError("unreachable source material family")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
