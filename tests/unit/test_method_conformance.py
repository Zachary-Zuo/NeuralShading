import pytest
import torch

from ncls.core.source import SourceSnapshot
from ncls.learning.conformance import (
    MethodArtifactInventory,
    validate_artifact_coverage,
    validate_gradient_coverage,
    validate_objective_outputs,
    validate_phase_execution,
)
from tests.fixtures.method import METHOD


def test_required_component_conformance_accepts_complete_contract() -> None:
    descriptor = METHOD.descriptor
    validate_phase_execution(
        descriptor,
        (
            {
                "name": "fit",
                "routes": [{"kind": "reference-evaluator"}],
                "parameter_groups": ["fixture"],
            },
        ),
    )
    validate_objective_outputs(descriptor, "fit", {"l1": torch.tensor(1.0)})
    validate_gradient_coverage(
        descriptor,
        {
            "fixture": {
                "finite_observed": True,
                "nonzero_gradient_observed": True,
                "parameter_update_observed": True,
            }
        },
    )
    validate_artifact_coverage(
        descriptor,
        MethodArtifactInventory(
            frozenset({"program:fixture", "asset:fixture"}),
            frozenset({"NclsPackageEvaluate"}),
        ),
    )


def test_required_component_conformance_rejects_missing_execution_gradient_and_artifact() -> None:
    descriptor = METHOD.descriptor
    with pytest.raises(ValueError, match="typed batch"):
        validate_phase_execution(
            descriptor,
            ({"name": "fit", "routes": [], "parameter_groups": ["fixture"]},),
        )
    with pytest.raises(RuntimeError, match="omitted"):
        validate_objective_outputs(descriptor, "fit", {})
    with pytest.raises(RuntimeError, match="gradient coverage"):
        validate_gradient_coverage(
            descriptor,
            {
                "fixture": {
                    "finite_observed": True,
                    "nonzero_gradient_observed": False,
                    "parameter_update_observed": True,
                }
            },
        )
    with pytest.raises(RuntimeError, match="artifact coverage"):
        validate_artifact_coverage(
            descriptor,
            MethodArtifactInventory(frozenset(), frozenset()),
        )


def test_parameter_registry_rejects_orphan_trainable_parameters() -> None:
    model = METHOD.create_trainable({})
    METHOD.parameter_registry(model)
    model.register_parameter("orphan", torch.nn.Parameter(torch.ones(1)))
    with pytest.raises(ValueError, match="orphan"):
        METHOD.parameter_registry(model)


def test_artifact_inventory_uses_symbols_from_the_packaged_module_closure() -> None:
    runtime = METHOD.compile_program({})
    snapshot = SourceSnapshot(
        "ncls.layer-stack@1",
        1,
        "fixture@1",
        "f" * 64,
        b"fixture",
    )
    asset = METHOD.compile_asset(
        snapshot,
        {},
    )
    inventory = MethodArtifactInventory.from_payloads(
        runtime,
        asset,
        checkpoint_model_state=False,
    )
    assert "slang:fixture.slang" in inventory.runtime_artifacts
    assert "NclsPackageBackend" in inventory.slang_entry_points
    assert "NclsPackageEvaluate" not in inventory.slang_entry_points
