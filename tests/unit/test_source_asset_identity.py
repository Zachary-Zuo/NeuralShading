from __future__ import annotations

import hashlib
from pathlib import Path

from ncls.source_materials.identity import materialx_asset_sha256, sha256_file


def test_materialx_asset_identity_matches_viewer_contract(tmp_path: Path) -> None:
    document = tmp_path / "material.mtlx"
    base_color = tmp_path / "base.jpg"
    roughness = tmp_path / "rough.exr"
    document.write_bytes(b"document")
    base_color.write_bytes(b"base")
    roughness.write_bytes(b"rough")

    identity = sha256_file(document)
    identity += f"\n{base_color.name}:{sha256_file(base_color)}"
    identity += f"\n{roughness.name}:{sha256_file(roughness)}"
    expected = hashlib.sha256(identity.encode("utf-8")).hexdigest()

    assert materialx_asset_sha256(
        document,
        (base_color, roughness, None, None, None),
    ) == expected
