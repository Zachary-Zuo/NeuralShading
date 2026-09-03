from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ncls.core.identity import sha256_file
from ncls.learning.methods.metal_fused import METHOD_DEFINITION
from ncls.learning.training import assess_checkpoint_readiness
from ncls.references.mdl import MdlCompiledArtifact
from tools.viewer import prepare_metal_catalog as exporter


def _coverage(*, include_proposal: bool = True) -> dict[str, dict[str, object]]:
    return {
        group: {
            "finite_observed": True,
            "nonzero_gradient_observed": True,
            "parameter_update_observed": True,
            "last_audit_step": 0,
        }
        for group in METHOD_DEFINITION.descriptor.parameter_groups
        if include_proposal or group != "proposal_sampler"
    }


def _training_config(run_class: str) -> dict[str, object]:
    return {
        "format_name": "ncls.training-config",
        "format_version": 4,
        "method_key": METHOD_DEFINITION.descriptor.method_key,
        "run_class": run_class,
        "correspondence_id": "test",
        "recipe_id": "test",
        "source_adaptation_id": "test",
        "source": {
            "family_id": "mdl.program@1",
            "materials": [{"locator": {"kind": "test"}}],
        },
        "online_query": {"recipe": "test"},
        "model_context": {"test": True},
        "phases": [
            {
                "name": "joint-coarse-to-fine",
                "steps": 1,
                "routes": [
                    {
                        "name": "test",
                        "kind": "asset-tile",
                        "batch_size": 1,
                        "direction_count": 1,
                        "seed_offset": 0,
                        "options": {},
                    }
                ],
                "parameter_groups": ["codec_encoder"],
                "loss_terms": ["test"],
                "recipes": {"test": True},
                "optimizer": {
                    "kind": "adam",
                    "betas": [0.9, 0.999],
                    "epsilon": 1e-8,
                    "weight_decay": 0.0,
                },
                "optimizer_state_policy": "reset",
                "schedule": {
                    "kind": "cosine",
                    "start": 0.001,
                    "end": 0.0001,
                    "total_steps": 1,
                    "offset": 0,
                },
                "precision": {"autocast": "fp32", "gradient_scaler": False},
                "checkpoint_boundary": True,
                "transition": None,
                "log_interval": 1,
                "gradient_audit_interval": 1,
                "prefetch_depth": 1,
            }
        ],
        "seed": 0,
        "device": "cuda",
        "validation": {"interval": 1, "batches": 1},
        "checkpoint_selection": "tail_guard",
    }


def test_checkpoint_readiness_separates_formal_and_explicit_diagnostic_preview() -> None:
    checkpoint = SimpleNamespace(
        phase_name="joint-coarse-to-fine",
        global_step=1,
        training_config=_training_config("smoke"),
        gradient_coverage=_coverage(),
        validate_method=lambda descriptor: None,
    )
    formal = assess_checkpoint_readiness(
        checkpoint, METHOD_DEFINITION.descriptor, mode="formal"
    )
    diagnostic = assess_checkpoint_readiness(
        checkpoint, METHOD_DEFINITION.descriptor, mode="diagnostic-evaluator"
    )
    assert not formal.ready
    assert formal.training_run_class == "smoke"
    assert diagnostic.ready
    assert (
        exporter.validate_preview_checkpoint(
            checkpoint, METHOD_DEFINITION.descriptor, diagnostic=True
        )
        == "exact-diagnostic-evaluator-preview"
    )
    with pytest.raises(ValueError, match="complete training"):
        exporter.validate_preview_checkpoint(checkpoint, METHOD_DEFINITION.descriptor)


def test_checkpoint_readiness_rejects_shape_only_or_incomplete_evaluator_coverage() -> None:
    coverage = _coverage(include_proposal=False)
    coverage.pop("hybrid_evaluator")
    checkpoint = SimpleNamespace(
        phase_name="joint-coarse-to-fine",
        global_step=1,
        training_config=_training_config("smoke"),
        gradient_coverage=coverage,
        validate_method=lambda descriptor: (_ for _ in ()).throw(
            ValueError("TrainingCheckpoint method descriptor identity mismatch")
        ),
    )
    readiness = assess_checkpoint_readiness(
        checkpoint, METHOD_DEFINITION.descriptor, mode="diagnostic-evaluator"
    )
    assert not readiness.ready
    assert not readiness.exact_method_identity
    assert readiness.failed_groups == ("hybrid_evaluator",)


def test_checkpoint_readiness_requires_formal_run_class_even_when_complete() -> None:
    checkpoint = SimpleNamespace(
        phase_name="complete",
        global_step=1,
        training_config=_training_config("smoke"),
        gradient_coverage=_coverage(),
        validate_method=lambda descriptor: None,
    )
    readiness = assess_checkpoint_readiness(
        checkpoint, METHOD_DEFINITION.descriptor, mode="formal"
    )
    assert not readiness.ready
    assert readiness.complete_training
    assert readiness.training_run_class == "smoke"
    assert any("run_class=formal" in reason for reason in readiness.reasons)


def test_diagnostic_catalog_selects_only_checkpoint_registry_intersection() -> None:
    records = tuple(
        SimpleNamespace(exact_locator={"module": "::m", "export": f"e{index}"})
        for index in range(3)
    )
    registry = SimpleNamespace(exports=records)
    locators = {
        ("::m", "e1"): {"module": "::m", "export": "e1"},
        ("::m", "e2"): {"module": "::m", "export": "e2"},
    }

    selected = exporter._select_registry_records(
        registry,
        locators,
        diagnostic_preview=True,
        limit=1,
    )

    assert selected == (records[1],)
    with pytest.raises(ValueError, match="does not cover.*exactly"):
        exporter._select_registry_records(
            registry,
            locators,
            diagnostic_preview=False,
            limit=None,
        )


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


def test_reference_compile_does_not_duplicate_provider_retry() -> None:
    class FailingProvider:
        calls = 0

        def compile_snapshot(self, _snapshot: object) -> object:
            self.calls += 1
            raise PermissionError("persistent directory publication failure")

    provider = FailingProvider()

    with pytest.raises(PermissionError, match="persistent"):
        exporter._compile_reference_program(provider, object())
    assert provider.calls == 1
