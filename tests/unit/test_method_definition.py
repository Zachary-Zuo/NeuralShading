from dataclasses import replace

import pytest

from ncls.core.source import SourceEditResult, SourceSnapshot
from ncls.learning.method import MethodReadinessPolicy
from ncls.learning.methods import method_plugins, public_method_keys
from tests.fixtures.method_definition import METHOD_DEFINITION


def _snapshot(family):
    return SourceSnapshot(family, 1, "fixture", "a" * 64, b"{}")


def test_product_registry_contains_canonical_product_methods_and_fixture_is_injected():
    assert public_method_keys() == ("metal", "nvidia")
    assert [item.descriptor.method_key for item in method_plugins()] == [
        "metal-budgeted-neural-material",
        "nvidia-neural-appearance",
    ]
    openpbr = _snapshot("openpbr.material@1.1.1")
    layer_stack = _snapshot("ncls.layer-stack@1")
    assert METHOD_DEFINITION.classify_edit(openpbr, SourceEditResult(openpbr, ("/inputs/base_weight",), ("material",))) == "runtime-patch"
    assert METHOD_DEFINITION.classify_edit(layer_stack, SourceEditResult(layer_stack, ("/interfaces/e0/roughness",), ("material",))) == "recompile"


def test_method_readiness_policy_is_owned_and_validated_by_the_descriptor():
    descriptor = METHOD_DEFINITION.descriptor
    assert "readiness_policies" not in descriptor.to_dict()
    with pytest.raises(ValueError, match="minimum_global_step"):
        MethodReadinessPolicy(("fixture",), ("fit",), minimum_global_step=0)
    with pytest.raises(ValueError, match="unsupported readiness policy"):
        replace(
            descriptor,
            readiness_policies={
                "formal": MethodReadinessPolicy(("fixture",), ("fit",))
            },
        )
    with pytest.raises(ValueError, match="unknown groups"):
        replace(
            descriptor,
            readiness_policies={
                "diagnostic-evaluator": MethodReadinessPolicy(
                    ("undeclared-group",), ("fit",)
                )
            },
        )
    with pytest.raises(ValueError, match="unknown phases"):
        replace(
            descriptor,
            readiness_policies={
                "diagnostic-evaluator": MethodReadinessPolicy(
                    ("fixture",), ("undeclared-phase",)
                )
            },
        )
