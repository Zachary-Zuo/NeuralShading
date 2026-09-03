from __future__ import annotations

from pathlib import Path

import pytest

from ncls.core.identity import sha256_file
from ncls.references.mdl import MdlCompiledArtifact
from tools.viewer import prepare_metal_catalog as exporter


def test_catalog_materializer_hardlinks_verified_content(tmp_path: Path) -> None:
    first_source = tmp_path / "first.bin"
    second_source = tmp_path / "second.bin"
    first_source.write_bytes(b"shared payload")
    second_source.write_bytes(b"shared payload")
    digest = sha256_file(first_source)
    objects: dict[str, Path] = {}

    first_target = tmp_path / "catalog/first.bin"
    second_target = tmp_path / "catalog/second.bin"
    exporter._copy_or_link(
        first_source,
        first_target,
        objects,
        expected_sha256=digest,
    )
    exporter._copy_or_link(
        second_source,
        second_target,
        objects,
        expected_sha256=digest,
    )

    assert first_source.samefile(first_target)
    assert first_target.samefile(second_target)
    assert objects[digest].samefile(first_target)


def test_catalog_materializer_rejects_a_false_declared_hash(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    with pytest.raises(ValueError, match="source hash mismatch"):
        exporter._copy_or_link(
            source,
            tmp_path / "catalog/target.bin",
            {},
            expected_sha256="0" * 64,
        )


def test_catalog_artifact_identity_uses_the_validated_manifest_hashes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    payload = root / "payload.bin"
    payload.write_bytes(b"payload")
    manifest_path = root / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    artifact = MdlCompiledArtifact(
        root,
        {"files_sha256": {"payload.bin": sha256_file(payload)}},
    )

    assert exporter._artifact_sha256(artifact) == artifact.artifact_sha256


def test_parallel_reference_compile_retries_transient_windows_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = object()

    class FlakyProvider:
        calls = 0

        def compile_snapshot(self, _snapshot: object) -> object:
            self.calls += 1
            if self.calls < 3:
                raise PermissionError("transient directory publication")
            return expected

    provider = FlakyProvider()
    delays: list[float] = []
    monkeypatch.setattr(exporter.time, "sleep", delays.append)

    assert exporter._compile_reference_program(provider, object()) is expected
    assert provider.calls == 3
    assert delays == [0.05, 0.1]


def test_parallel_reference_compile_retry_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingProvider:
        calls = 0

        def compile_snapshot(self, _snapshot: object) -> object:
            self.calls += 1
            raise PermissionError("persistent directory publication failure")

    provider = FailingProvider()
    monkeypatch.setattr(exporter.time, "sleep", lambda _delay: None)

    with pytest.raises(PermissionError, match="persistent"):
        exporter._compile_reference_program(provider, object())
    assert provider.calls == 4
