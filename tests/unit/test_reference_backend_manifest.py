from __future__ import annotations

import json
from pathlib import Path

import pytest

from ncls.references.backend_manifest import (
    CANONICAL_PROGRAMS,
    REFERENCE_BACKEND_MANIFEST,
    ReferenceBackendManifest,
    current_reference_platform_id,
)
from ncls.references.programs import discover_reference_programs


def _manifest() -> dict[str, object]:
    return json.loads(REFERENCE_BACKEND_MANIFEST.read_text(encoding="utf-8"))


def _write(tmp_path: Path, value: dict[str, object]) -> Path:
    path = tmp_path / "reference-backend-toolchains.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_reference_backend_manifest_covers_the_canonical_program_registry() -> None:
    manifest = ReferenceBackendManifest.load()
    discovered = {
        (program.descriptor.program_key, program.descriptor.version)
        for program in discover_reference_programs()
    }
    assert discovered == CANONICAL_PROGRAMS
    assert {
        (program.program_key, program.version) for program in manifest.programs
    } == discovered
    assert manifest.asset_policy == "external-only-no-source-assets"
    assert {platform.platform_id for platform in manifest.platforms} == {
        "windows-x86_64@1",
        "linux-x86_64@1",
    }
    assert len(manifest.semantic_identity) == 64


def test_reference_backend_build_inputs_never_address_source_assets() -> None:
    manifest = ReferenceBackendManifest.load()
    assert all(source.path.startswith("external/") for source in manifest.source_providers)
    serialized = REFERENCE_BACKEND_MANIFEST.read_text(encoding="utf-8").lower()
    for forbidden in (
        "assets/source-materials",
        "fetch_source_materials",
        "fetch_mdl_assets",
        "polyhaven.org/file",
        "omniverse",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    ("platform_name", "machine", "expected"),
    (
        ("win32", "AMD64", "windows-x86_64@1"),
        ("linux", "x86_64", "linux-x86_64@1"),
    ),
)
def test_reference_platform_identity_is_explicit(
    platform_name: str, machine: str, expected: str
) -> None:
    assert (
        current_reference_platform_id(
            platform_name=platform_name, machine=machine
        )
        == expected
    )


def test_reference_platform_identity_rejects_unknown_os_and_architecture() -> None:
    with pytest.raises(RuntimeError, match="platform"):
        current_reference_platform_id(platform_name="darwin", machine="x86_64")
    with pytest.raises(RuntimeError, match="architecture"):
        current_reference_platform_id(platform_name="linux", machine="aarch64")


def test_reference_backend_manifest_rejects_duplicate_provider(tmp_path: Path) -> None:
    value = _manifest()
    providers = list(value["source_providers"])
    providers[1] = dict(providers[0])
    value["source_providers"] = providers
    with pytest.raises(ValueError, match="unique"):
        ReferenceBackendManifest.load(_write(tmp_path, value))


def test_reference_backend_manifest_rejects_unsafe_or_asset_paths(
    tmp_path: Path,
) -> None:
    for invalid in ("../outside", "assets/source-materials/mdl"):
        value = _manifest()
        providers = list(value["source_providers"])
        first = dict(providers[0])
        first["path"] = invalid
        providers[0] = first
        value["source_providers"] = providers
        with pytest.raises(ValueError, match="unsafe|source assets"):
            ReferenceBackendManifest.load(_write(tmp_path, value))


def test_reference_backend_manifest_rejects_unknown_program_provider(
    tmp_path: Path,
) -> None:
    value = _manifest()
    programs = list(value["programs"])
    first = dict(programs[0])
    first["providers"] = ["not-a-provider"]
    programs[0] = first
    value["programs"] = programs
    with pytest.raises(ValueError, match="unknown providers"):
        ReferenceBackendManifest.load(_write(tmp_path, value))


def test_reference_backend_manifest_rejects_archive_hash_drift(tmp_path: Path) -> None:
    value = _manifest()
    platforms = list(value["platforms"])
    first = dict(platforms[0])
    sdk = dict(first["mdl_sdk"])
    archive = dict(sdk["archive"])
    archive["sha256"] = "bad"
    sdk["archive"] = archive
    first["mdl_sdk"] = sdk
    platforms[0] = first
    value["platforms"] = platforms
    with pytest.raises(ValueError, match="SHA-256"):
        ReferenceBackendManifest.load(_write(tmp_path, value))
