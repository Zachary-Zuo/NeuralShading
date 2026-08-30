from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from ncls.learning.metal_runtime import (
    fake_quantize_fp16_ste,
    metal_runtime_parameter_names,
)
from ncls.learning.methods.metal_fused import METHOD_DEFINITION
from ncls.learning.models.metal_fused import (
    METAL_FUSED_REQUIRED_CONTEXT,
    MetalFusedNeuralMaterialModel,
)
from ncls.learning.training import TrainingConfig
from ncls.source_materials.mdl_metal import MdlMetalRegistry
from tools.learning.build_metal_training_configs import (
    LINUX_LONG_PATH,
    LINUX_SMOKE_PATH,
    REGISTRY_PATH,
    WINDOWS_SMOKE_PATH,
    build_linux_configs,
    semantic_training_fingerprint,
    validate_linux_config_pair,
)
from tools.learning.build_metal_linux_handoff import build_handoff_manifest


def test_fp16_runtime_fake_quantization_uses_deployed_values_and_master_gradient() -> None:
    value = torch.tensor([1.0003, -0.33337, 0.0], requires_grad=True)
    quantized = fake_quantize_fp16_ste(value)
    torch.testing.assert_close(
        quantized.detach(), value.detach().to(torch.float16).to(torch.float32)
    )
    quantized.sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value))


def test_qat_refine_phase_covers_every_exported_runtime_parameter_group() -> None:
    with torch.device("meta"):
        model = MetalFusedNeuralMaterialModel.from_context(
            METAL_FUSED_REQUIRED_CONTEXT
        )
    registry = METHOD_DEFINITION.parameter_registry(model)
    runtime_names = set(metal_runtime_parameter_names(model))
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    runtime_groups = {
        group
        for group, parameters in registry.items()
        if runtime_names.intersection(parameter_names[id(parameter)] for parameter in parameters)
    }
    config = TrainingConfig.load(WINDOWS_SMOKE_PATH)
    qat = config.phases[-1]
    assert qat.name == "qat-refine"
    assert runtime_groups.issubset(qat.parameter_groups)
    assert "optimized_state_teacher" not in qat.parameter_groups
    assert qat.precision == {"autocast": "fp32", "gradient_scaler": False}
    assert set(route.name for route in qat.routes) == {"asset", "evaluator", "sampler"}


def test_checked_in_linux_configs_are_full_cohort_canonical_pair() -> None:
    windows = TrainingConfig.load(WINDOWS_SMOKE_PATH).to_dict()
    registry = MdlMetalRegistry.load(REGISTRY_PATH)
    expected_smoke, expected_long = build_linux_configs(windows, registry)
    actual_smoke = TrainingConfig.load(LINUX_SMOKE_PATH).to_dict()
    actual_long = TrainingConfig.load(LINUX_LONG_PATH).to_dict()
    assert actual_smoke == expected_smoke
    assert actual_long == expected_long
    report = validate_linux_config_pair(actual_smoke, actual_long)
    assert report["source_count"] == 692
    assert report["smoke_steps"] == 16
    assert report["long_steps"] == 120_000
    assert report["phase_names"] == [
        "codec-warmup",
        "joint-appearance",
        "proposal-fit",
        "qat-refine",
    ]
    assert report["distributed"] is False
    assert report["visible_gpu_count"] == 1


def test_linux_semantic_fingerprint_rejects_method_data_or_precision_drift() -> None:
    smoke = TrainingConfig.load(LINUX_SMOKE_PATH).to_dict()
    baseline = semantic_training_fingerprint(smoke)
    for mutation in (
        lambda value: value["phases"][1]["loss_terms"].append("unregistered-loss"),
        lambda value: value["phases"][3]["precision"].update(autocast="bfloat16"),
        lambda value: value["source"]["materials"].pop(),
    ):
        changed = deepcopy(smoke)
        mutation(changed)
        assert semantic_training_fingerprint(changed) != baseline


def test_windows_smoke_uses_the_registry_generated_stratified_activation_set() -> None:
    config = TrainingConfig.load(WINDOWS_SMOKE_PATH)
    assert len(config.source["materials"]) == 3
    assert [phase.name for phase in config.phases] == [
        "codec-warmup",
        "joint-appearance",
        "proposal-fit",
        "qat-refine",
    ]
    asset_routes = [
        route
        for phase in config.phases
        for route in phase.routes
        if route.name == "asset"
    ]
    assert asset_routes
    assert all(route.options["asset_indices"] == [6, 50, 22] for route in asset_routes)
    assert Path(config.source["materials"][0]["locator"]["module_root"]).as_posix().endswith(
        "assets/source-materials/mdl-vmaterials2/2.4.0/Materials"
    )


def test_linux_handoff_is_single_gpu_recoverable_and_has_no_automatic_followup() -> None:
    manifest = build_handoff_manifest()
    assert manifest["producer"] == {
        "processes": 1,
        "visible_gpus": 1,
        "distributed": False,
        "persistent_training_batches": False,
        "response_target_device": "cuda:0",
    }
    assert manifest["linux_execution_status"] == "pending-on-target-host"
    assert manifest["automatic_followups"] == []
    assert "--stop-at-step 16" in manifest["commands"]["long_start_recoverable"]
    assert "--resume" in manifest["commands"]["long_resume"]
    assert "formal" not in " ".join(manifest["commands"].values())
