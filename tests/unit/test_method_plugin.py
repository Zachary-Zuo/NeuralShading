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


def test_evaluator_only_recipe_does_not_require_a_sampler_route() -> None:
    from ncls.paths import PROJECT_ROOT
    from ncls.learning.training.plan import TrainingPlanResolver

    plan = TrainingPlanResolver(PROJECT_ROOT).resolve(
        PROJECT_ROOT / "configs/training/runs/metal-spatial-probe-bronze-scratched.yaml")
    method = get_method("metal")
    config = plan.training.to_dict()
    assert {item.route_kind for item in method.requirements(config)} == {"reference-evaluator"}
    config["phases"][0]["routes"][0]["kind"] = "unknown-route"
    with pytest.raises(ValueError, match="not supported by the method"):
        method.requirements(config)
