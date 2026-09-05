from dataclasses import replace

import pytest

from ncls.data import DataRequirement
from ncls.learning.methods import get_method, registered_methods


def test_registered_methods_directly_provide_their_data_requirements() -> None:
    plugins = registered_methods()
    assert tuple(item.key for item in plugins) == ("metal", "nvidia")
    for plugin in plugins:
        assert "@" not in plugin.key
        assert {
            item.route_kind: item.fields for item in plugin.requirements()
        } == dict(plugin.descriptor.training_batch_requirements)




def test_unknown_public_method_plugin_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported method"):
        get_method("metal@1")
