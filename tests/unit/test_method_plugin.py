from dataclasses import replace

import pytest

from ncls.data import DataRequirement
from ncls.learning.methods import get_method_plugin, method_plugins


def test_product_method_plugins_have_short_keys_and_complete_facets() -> None:
    plugins = method_plugins()
    assert tuple(item.key for item in plugins) == ("metal", "nvidia")
    for plugin in plugins:
        assert "@" not in plugin.key
        assert set(plugin.facet_identities) == {
            "model",
            "data",
            "objective",
            "lifecycle",
            "checkpoint",
            "deployment",
        }
        assert all(len(value) == 64 for value in plugin.facet_identities.values())
        assert {
            item.route_kind: item.fields for item in plugin.data.requirements()
        } == dict(plugin.descriptor.training_batch_requirements)


def test_method_plugin_rejects_data_requirement_drift() -> None:
    plugin = get_method_plugin("nvidia")

    class IncompleteDataFacet:
        implementation_sha256 = "a" * 64

        def requirements(self):
            return (DataRequirement("reference-evaluator", ("wo", "target_f")),)

        def create_source_adapter(self, snapshots, device):
            del snapshots, device
            raise AssertionError

    with pytest.raises(ValueError, match="requirements disagree"):
        replace(plugin, data=IncompleteDataFacet())


def test_unknown_public_method_plugin_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported method plugin"):
        get_method_plugin("metal@1")
