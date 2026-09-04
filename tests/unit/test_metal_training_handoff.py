from __future__ import annotations

from pathlib import Path

import torch

from ncls.learning.metal_runtime import (
    fake_quantize_fp16_ste,
    metal_runtime_parameter_names,
)
from ncls.learning.methods.metal_fused import METHOD_DEFINITION
from ncls.learning.models.metal_fused import (
    METAL_FUSED_REQUIRED_CONTEXT,
    MetalModel,
)
from ncls.learning.training import TrainingPlanResolver
from tools.learning.build_metal_linux_handoff import build_handoff_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _runtime(run: str):
    return TrainingPlanResolver(PROJECT_ROOT).resolve(
        f"configs/training/runs/{run}.yaml"
    ).to_runtime_config()


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
        model = MetalModel.from_context(
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
    config = _runtime("metal-windows-smoke")
    qat = config.phases[-1]
    assert qat.name == "qat-refine"
    assert runtime_groups.issubset(qat.parameter_groups)
    assert "optimized_state_teacher" not in qat.parameter_groups
    assert qat.precision == {"autocast": "fp32", "gradient_scaler": False}
    assert set(route.name for route in qat.routes) == {"asset", "evaluator", "sampler"}


def test_composed_linux_plans_are_full_cohort_canonical_pair() -> None:
    smoke = _runtime("metal-linux-smoke")
    long_run = _runtime("metal-linux-long")
    assert smoke.source == long_run.source
    assert smoke.model_context == long_run.model_context
    assert smoke.online_query == long_run.online_query
    assert len(smoke.source["materials"]) == 692
    assert smoke.total_steps == 16
    assert long_run.total_steps == 120_000
    assert [phase.name for phase in smoke.phases] == [
        "joint-coarse-to-fine",
        "qat-refine",
    ]
    assert [phase.name for phase in long_run.phases] == [
        "joint-coarse-to-fine",
        "qat-refine",
    ]
    assert smoke.run_class == "smoke"
    assert long_run.run_class == "formal"


def test_windows_smoke_uses_the_registry_generated_stratified_activation_set() -> None:
    config = _runtime("metal-windows-smoke")
    assert len(config.source["materials"]) == 3
    assert [phase.name for phase in config.phases] == [
        "joint-coarse-to-fine",
        "qat-refine",
    ]
    assert set(route.name for route in config.phases[0].routes) == {
        "asset",
        "evaluator",
        "sampler",
    }
    assert "proposal_sampler" in config.phases[0].parameter_groups
    assert config.phases[0].recipes["proposal_weight"]["start"] > 0.0
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
    assert "configs/learning" not in " ".join(manifest["commands"].values())
    assert "learn " not in " ".join(manifest["commands"].values())
    assert "formal" not in " ".join(manifest["commands"].values())
