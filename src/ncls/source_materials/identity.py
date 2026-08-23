from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def materialx_asset_sha256(
    document_path: str | Path,
    textures: Iterable[str | Path | None],
) -> str:
    """返回 H5 provider 与 viewer 共用的 MaterialX 原始资产身份。

    纹理按标准语义槽顺序传入：base color、roughness、metalness、normal、
    displacement。空槽不参与；URI 本身属于 native state，资产身份只绑定实际字节。
    """

    document = Path(document_path)
    identity = sha256_file(document)
    for texture_value in textures:
        if texture_value is None:
            continue
        texture = Path(texture_value)
        identity += f"\n{texture.name}:{sha256_file(texture)}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()
