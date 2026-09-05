from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ncls.core.identity import sha256_file
from ncls.learning.models.metal_budgeted_profile import (
    METAL_BUDGETED_DIRECT_PROFILE_ID,
    METAL_BUDGETED_HYBRID_PROFILE_ID,
)
from ncls.references.mdl import MdlCompiledArtifact
from tools.viewer import prepare_metal_catalog as exporter


def _checkpoint(role: str, *, step: int = 2048, snapshot_id: str = "a" * 64):
    profile = {
        "hybrid": METAL_BUDGETED_HYBRID_PROFILE_ID,
        "direct": METAL_BUDGETED_DIRECT_PROFILE_ID,
    }[role]
    calls: list[str] = []
    return SimpleNamespace(
        public_method_key="metal",
        deployment_payload={
            "training_config": {"model_context": {"profile_id": profile}}
        },
        source={
            "family_id": "mdl.program@1",
            "materials": [
                {
                    "locator": {
                        "kind": "mdl-export",
                        "module_root": "assets/mdl",
                        "module": "::test",
                        "export": "main",
                    }
                }
            ],
        },
        source_snapshot_ids=(snapshot_id,),
        global_step=step,
        require_ready=lambda mode: calls.append(mode) or {"ready": True},
        readiness_calls=calls,
    )


def test_pair_requires_exact_profiles_source_and_common_step() -> None:
    hybrid = _checkpoint("hybrid")
    direct = _checkpoint("direct")

    locator, snapshot_id = exporter._validate_pair(hybrid, direct)

    assert locator["export"] == "main"
    assert snapshot_id == "a" * 64
    assert hybrid.readiness_calls == ["diagnostic-evaluator"]
    assert direct.readiness_calls == ["diagnostic-evaluator"]
    assert exporter.validate_deployment_checkpoint(hybrid) == (
        "exact"
    )


def test_pair_rejects_profile_role_swaps_and_asymmetric_milestones() -> None:
    with pytest.raises(ValueError, match="hybrid checkpoint profile"):
        exporter._validate_pair(_checkpoint("direct"), _checkpoint("direct"))
    with pytest.raises(ValueError, match="common training milestone"):
        exporter._validate_pair(_checkpoint("hybrid"), _checkpoint("direct", step=1024))


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


def test_reference_compile_delegates_publication_policy_to_provider() -> None:
    expected = object()

    class Provider:
        calls = 0

        def compile_snapshot(self, _snapshot: object) -> object:
            self.calls += 1
            return expected

    provider = Provider()

    assert exporter._compile_reference_program(provider, object()) is expected
    assert provider.calls == 1


def test_runtime_paths_accept_locked_linux_sdk_and_reject_cross_platform_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = Path("examples/mdl_sdk/dxr/content/mdl_target_code_types.hlsl")
    linux = (
        tmp_path
        / "external/MDL-SDK-2025.0.0-387700.1252-linux-x86-64"
        / relative
    )
    runtime = tmp_path / "shaders/ncls/reference_backends/mdl_runtime.slangh"
    linux.parent.mkdir(parents=True)
    runtime.parent.mkdir(parents=True)
    linux.write_bytes(b"shared target types")
    runtime.write_bytes(b"renderer runtime")
    monkeypatch.setattr(exporter, "PROJECT_ROOT", tmp_path)

    assert exporter._runtime_paths() == (linux, runtime)

    windows = (
        tmp_path
        / "external/MDL-SDK-2025.0.0-387700.1252-nt-x86-64"
        / relative
    )
    windows.parent.mkdir(parents=True)
    windows.write_bytes(b"different target types")
    with pytest.raises(ValueError, match="headers differ"):
        exporter._runtime_paths()
