from dataclasses import replace

import pytest
import torch

from ncls.core.identity import sha256_json
from ncls.learning.training import TrainingConfig, TrainingPhase, TrainingRoute
from ncls.learning.training.checkpoint import TrainingCheckpoint, load_checkpoint, save_checkpoint
from tests.fixtures.method_definition import METHOD_DEFINITION


def _config() -> TrainingConfig:
    return TrainingConfig(
        "contract-fixture",
        "smoke",
        "fixture-correspondence@1",
        "fixture-recipe@1",
        "fixture-adaptation@1",
        {"family_id": "ncls.layer-stack@1", "materials": [{"locator": {"kind": "fixture"}}]},
        {"recipe_id": "online@1"},
        {"fixture": True},
        (
            TrainingPhase(
                "fit", 2,
                (TrainingRoute("evaluator", "reference-evaluator", 1, 1, 0, {}),),
                ("fixture",), ("l1",), {"fixture": True},
                {"kind": "adam", "betas": [0.9, 0.999], "epsilon": 1e-7,
                 "weight_decay": 0.0},
                "reset",
                {"kind": "cosine", "start": 1e-3, "end": 1e-4,
                 "total_steps": 2, "offset": 0},
                {"autocast": "fp32", "gradient_scaler": False},
                True, None, 1, 1, 1,
            ),
        ),
        7, "cpu", {"interval": 1, "batches": 1}, "tail_guard",
    )


def test_checkpoint_v4_roundtrip_and_tensor_schema(tmp_path):
    descriptor = METHOD_DEFINITION.descriptor
    config = _config()
    config_value = config.to_dict()
    component_manifest = {
        "schema": "ncls.method-components@1",
        "parameter_groups": {name: list(values) for name, values in descriptor.parameter_groups.items()},
        "components": [component.to_dict() for component in descriptor.components],
    }
    checkpoint = TrainingCheckpoint(
        descriptor.method_key,
        descriptor.descriptor_sha256,
        descriptor.implementation_sha256,
        component_manifest,
        config_value,
        sha256_json(config_value),
        sha256_json([phase.to_dict() for phase in config.phases]),
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
        tuple(x.to_dict() for x in descriptor.supported_sources),
        ("a" * 64,),
        1,
        0,
        "fit",
        1,
        {"metric": 1.0},
        {"fixture.scale": torch.ones(3), "fixture.bias": torch.zeros(3)},
        {
            "phase_name": "fit",
            "optimizer": {"parameter_names": ["weight", "bias"], "state_by_name": {}},
            "scheduler": {"kind": "cosine"},
            "precision": {"config": {"autocast": "fp32", "gradient_scaler": False}, "scaler": {}},
        },
        {"torch": torch.get_rng_state()},
        {"query_stream_identity": "4" * 64},
        {
            "fixture": {
                "finite_observed": True,
                "nonzero_gradient_observed": True,
                "parameter_update_observed": True,
                "last_audit_step": 0,
            }
        },
        {"rows": []},
    )
    path = tmp_path / "checkpoint.pt"
    digest = save_checkpoint(path, checkpoint)
    assert len(digest) == 64
    restored = load_checkpoint(path, descriptor=descriptor)
    assert restored.format_version == 4 and restored.global_step == 1

    with pytest.raises(ValueError, match="method descriptor identity mismatch"):
        replace(restored, implementation_identity="0" * 64).validate_method(
            descriptor
        )


def test_complete_checkpoint_rejects_incomplete_required_gradient_coverage() -> None:
    descriptor = METHOD_DEFINITION.descriptor
    config = _config()
    config_value = config.to_dict()
    checkpoint = TrainingCheckpoint(
        descriptor.method_key,
        descriptor.descriptor_sha256,
        descriptor.implementation_sha256,
        {
            "schema": "ncls.method-components@1",
            "parameter_groups": {
                name: list(values) for name, values in descriptor.parameter_groups.items()
            },
            "components": [component.to_dict() for component in descriptor.components],
        },
        config_value,
        sha256_json(config_value),
        sha256_json([phase.to_dict() for phase in config.phases]),
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
        tuple(value.to_dict() for value in descriptor.supported_sources),
        ("a" * 64,),
        1,
        0,
        "fit",
        1,
        {},
        {"fixture.scale": torch.ones(3), "fixture.bias": torch.zeros(3)},
        {
            "phase_name": "fit",
            "optimizer": {"parameter_names": ["weight", "bias"], "state_by_name": {}},
            "scheduler": {"kind": "cosine"},
            "precision": {
                "config": {"autocast": "fp32", "gradient_scaler": False},
                "scaler": {},
            },
        },
        {"torch": torch.get_rng_state()},
        {"query_stream_identity": "4" * 64},
        {
            "fixture": {
                "finite_observed": True,
                "nonzero_gradient_observed": False,
                "parameter_update_observed": True,
                "last_audit_step": 0,
            }
        },
        {"rows": []},
    )

    with pytest.raises(ValueError, match="incomplete required gradient coverage"):
        replace(
            checkpoint,
            global_step=2,
            phase_index=1,
            phase_name="complete",
            phase_step=0,
            phase_optimization_state={},
        )
