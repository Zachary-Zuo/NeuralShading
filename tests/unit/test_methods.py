from dataclasses import replace

import pytest

from ncls.core.source import SourceEditResult, SourceSnapshot
from ncls.learning.methods import registered_methods, method_keys
from tests.fixtures.method import METHOD


def _snapshot(family):
    return SourceSnapshot(family, 1, "fixture", "a" * 64, b"{}")


def test_product_registry_contains_canonical_product_methods_and_fixture_is_injected():
    assert method_keys() == ("metal", "nvidia")
    assert [item.descriptor.method_key for item in registered_methods()] == [
        "metal-budgeted-neural-material",
        "nvidia-neural-appearance",
    ]
    openpbr = _snapshot("openpbr.material@1.1.1")
    layer_stack = _snapshot("ncls.layer-stack@1")
    assert METHOD.classify_edit(openpbr, SourceEditResult(openpbr, ("/inputs/base_weight",), ("material",))) == "runtime-patch"
    assert METHOD.classify_edit(layer_stack, SourceEditResult(layer_stack, ("/interfaces/e0/roughness",), ("material",))) == "recompile"
