from __future__ import annotations

from pathlib import Path

import torch

from ncls.learning.metal_runtime import (
    fake_quantize_fp16_ste,
)
from ncls.learning.methods.metal_budgeted import (
    METHOD_DEFINITION,
    metal_budgeted_runtime_parameter_names,
)
from ncls.learning.models.metal_budgeted import (
    METAL_BUDGETED_REQUIRED_CONTEXT,
    MetalBudgetedModel,
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


def test_qat_refine_phase_covers_every_budgeted_parameter_group() -> None:
    with torch.device("meta"):
        model = MetalBudgetedModel.from_context(
            METAL_BUDGETED_REQUIRED_CONTEXT
        )
    registry = METHOD_DEFINITION.parameter_registry(model)
    config = _runtime("metal-budgeted-hybrid-pilot")
    qat = config.phases[-1]
    assert qat.name == "deployment-qat-refine"
    assert set(registry) == set(qat.parameter_groups)
    assert qat.precision == {"autocast": "fp32", "gradient_scaler": False}
    assert set(route.name for route in qat.routes) == {"evaluator", "sampler"}
    assert (
        qat.recipes["runtime_quantization"]
        == "fp16-weights-state-rgba8-snorm-asset@1"
    )
    runtime = metal_budgeted_runtime_parameter_names(model)
    assert any(name.startswith("typed_compiler.") for name in runtime)
    assert any(name.startswith("prepared_model.") for name in runtime)
    assert any(name.startswith("evaluator.") for name in runtime)
    assert "asset.variant_scale_bias.weight" in runtime
    assert not any(name.startswith("asset.detail_encoder.") for name in runtime)


def test_composed_linux_pilots_are_matched_except_registered_profile_axis() -> None:
    hybrid = _runtime("metal-budgeted-hybrid-pilot")
    direct = _runtime("metal-budgeted-direct-pilot")
    assert hybrid.source == direct.source
    assert hybrid.online_query == direct.online_query
    assert len(hybrid.source["materials"]) == 1
    assert hybrid.total_steps == direct.total_steps == 2048
    assert [phase.name for phase in hybrid.phases] == [
        "joint-response-fit",
        "deployment-qat-refine",
    ]
    assert [phase.name for phase in direct.phases] == [
        "joint-response-fit",
        "deployment-qat-refine",
    ]
    assert hybrid.model_context["profile_id"] == "metal_budgeted_hybrid_v3"
    assert direct.model_context["profile_id"] == "metal_budgeted_direct_control_v3"
    assert hybrid.run_class == direct.run_class == "profile"
    for hybrid_phase, direct_phase in zip(hybrid.phases, direct.phases, strict=True):
        assert hybrid_phase.routes == direct_phase.routes
        assert hybrid_phase.parameter_groups == direct_phase.parameter_groups
        assert hybrid_phase.loss_terms == direct_phase.loss_terms
        assert hybrid_phase.optimizer == direct_phase.optimizer
        assert hybrid_phase.schedule == direct_phase.schedule
        assert hybrid_phase.precision == direct_phase.precision
        hybrid_recipes = dict(hybrid_phase.recipes)
        direct_recipes = dict(direct_phase.recipes)
        assert hybrid_recipes.pop("profile_id") != direct_recipes.pop("profile_id")
        assert hybrid_recipes == direct_recipes
        assert hybrid_recipes["asset_cook_mode"] == "encoder-only@1"


def test_tungsten_pilot_uses_frozen_spatial_and_direction_query_recipe() -> None:
    config = _runtime("metal-budgeted-hybrid-pilot")
    assert len(config.source["materials"]) == 1
    assert [phase.name for phase in config.phases] == [
        "joint-response-fit",
        "deployment-qat-refine",
    ]
    assert set(route.name for route in config.phases[0].routes) == {"evaluator", "sampler"}
    assert "proposal_sampler" in config.phases[0].parameter_groups
    assert config.phases[0].recipes["proposal_weight"]["start"] > 0.0
    evaluator = next(
        route for route in config.phases[0].routes if route.name == "evaluator"
    )
    assert evaluator.options["direction_proposal"] == "balanced-four-mode-probe@1"
    assert evaluator.options["footprint_recipe"] == "balanced-zero-one-four-texel@1"
    assert evaluator.options["paired_uv_recipe"] == "one-native-texel-axis-balanced@1"
    assert evaluator.options["validation_seed"] == 2026090402
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
    assert manifest["repository_state"] in {
        "clean-commit",
        "working-tree-snapshot-required",
    }
    assert "exact required working-tree changes" in manifest["transfer_precondition"]
    assert bool(manifest["required_worktree_changes"]) == (
        manifest["repository_state"] == "working-tree-snapshot-required"
    )
    for name in (
        "hybrid_resolved_plan_sha256",
        "direct_resolved_plan_sha256",
        "hybrid_training_config_sha256",
        "direct_training_config_sha256",
    ):
        assert len(manifest["config_pair"][name]) == 64
    assert "--stop-at-step 128" in manifest["commands"]["hybrid_start_recoverable"]
    assert "--resume" in manifest["commands"]["hybrid_resume"]
    assert "--stop-at-step 128" in manifest["commands"]["direct_start_recoverable"]
    assert "--batches 256" in manifest["commands"]["hybrid_step0_validation"]
    assert "configs/learning" not in " ".join(manifest["commands"].values())
    assert "learn " not in " ".join(manifest["commands"].values())
    assert "formal" not in " ".join(manifest["commands"].values())
