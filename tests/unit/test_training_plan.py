from pathlib import Path
import shutil

import pytest

from ncls.learning.training import (
    ResolvedTrainingPlan,
    TrainingPlanResolver,
)
from ncls.source_materials.mdl_metal import MdlMetalRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_resolved_training_plan_roundtrips_embedded_checkpoint_manifest() -> None:
    plan = TrainingPlanResolver(PROJECT_ROOT).resolve(
        "configs/training/runs/nvidia-layer-stack-smoke.yaml"
    )
    restored = ResolvedTrainingPlan.from_dict(plan.to_dict())
    assert restored.to_dict() == plan.to_dict()
    assert restored.sha256 == plan.sha256


@pytest.mark.parametrize(
    ("run_path", "public_method", "plan_sha256", "source_count", "total_steps"),
    (
        (
            "configs/training/runs/nvidia-layer-stack-smoke.yaml",
            "nvidia",
            "986b18b91c79083e3b4253581ae07e2f1beb69fe6534176cbb7560a0fc54f793",
            1,
            2,
        ),
            (
                "configs/training/runs/metal-budgeted-hybrid-pilot.yaml",
                "metal",
                "5d6fbd3e5b5b056c0dff0361a22642133e4722a2adb1b47c78bfaec36bb6f415",
            1,
            2048,
        ),
        (
            "configs/training/runs/nvidia-materialx-smoke.yaml",
            "nvidia",
            "f5122a0dc73fa78503f80ab85176d938a6c592bb91a26001a631ddeca00d360a",
            1,
            2,
        ),
        (
            "configs/training/runs/nvidia-materialx-formal.yaml",
            "nvidia",
            "b20eb5b9f20f703631021d3453e5f98f5e3575781dc80a8cc474a0512d111c96",
            1,
            300000,
        ),
        (
            "configs/training/runs/nvidia-mdl-effect-pigment-smoke.yaml",
            "nvidia",
            "2f668ac6d25343d8412cfb9ccd43c3f4af7ea6bf1ef36856d05d4fa27bb90f7e",
            1,
            2,
        ),
            (
                "configs/training/runs/metal-budgeted-direct-pilot.yaml",
                "metal",
                "aa7aaeba0535b4767efb5af573c6d2e5fc8902bf1c785e9113d779d10539a034",
            1,
            2048,
        ),
    ),
)
def test_resolved_training_plan_has_frozen_canonical_identity(
    run_path: str,
    public_method: str,
    plan_sha256: str,
    source_count: int,
    total_steps: int,
) -> None:
    resolver = TrainingPlanResolver(PROJECT_ROOT)
    first = resolver.resolve(run_path)
    second = resolver.resolve(run_path)

    assert first.sha256 == second.sha256 == plan_sha256
    assert first.selection.method == public_method
    assert "@" not in first.selection.method
    runtime = first.to_runtime_config()
    assert len(runtime.source["materials"]) == source_count
    assert runtime.total_steps == total_steps
    assert first.method_descriptor["public_key"] == public_method
    assert len(first.method_descriptor["descriptor_sha256"]) == 64
    assert [item.kind for item in first.inputs[:4]] == [
        "base",
        "method",
        "data",
        "recipe",
    ]


def test_only_devices_are_applied_as_cli_override() -> None:
    resolver = TrainingPlanResolver(PROJECT_ROOT)
    baseline = resolver.resolve(
        "configs/training/runs/nvidia-layer-stack-smoke.yaml"
    )
    overridden = resolver.resolve(
        "configs/training/runs/nvidia-layer-stack-smoke.yaml", devices=(2, 3)
    )

    assert baseline.execution.devices == (0,)
    assert overridden.execution.devices == (2, 3)
    assert overridden.overrides == {"execution.devices": (2, 3)}
    assert overridden.sha256 != baseline.sha256
    assert overridden.to_runtime_config().sha256 == baseline.to_runtime_config().sha256


def test_execution_settings_reject_duplicate_devices() -> None:
    resolver = TrainingPlanResolver(PROJECT_ROOT)
    with pytest.raises(ValueError, match="unique nonnegative GPU indices"):
        resolver.resolve(
            "configs/training/runs/nvidia-layer-stack-smoke.yaml", devices=(0, 0)
        )


def test_material_set_resolver_expands_full_metal_registry_without_inline_locators(
    tmp_path: Path,
) -> None:
    shutil.copytree(
        PROJECT_ROOT / "configs" / "training",
        tmp_path / "configs" / "training",
    )
    registry_relative = Path("references/mdl-vmaterials2-v1/metal-opaque-v1.json")
    registry_target = tmp_path / registry_relative
    registry_target.parent.mkdir(parents=True)
    shutil.copy2(PROJECT_ROOT / registry_relative, registry_target)
    run_path = tmp_path / "configs" / "training" / "runs" / "metal-full-fixture.yaml"
    run_path.write_text(
        """\
format_name: ncls.training-run
format_version: 1
compose:
  method: metal
  data: mdl-metal-budgeted-full
  recipe: metal-budgeted-hybrid-pilot
""",
        encoding="utf-8",
    )

    plan = TrainingPlanResolver(tmp_path).resolve(run_path)
    materials = plan.to_runtime_config().source["materials"]
    registry = MdlMetalRegistry.load(registry_target)
    assert len(materials) == len(registry.exports) == 692
    assert materials[0]["locator"]["export"] == registry.exports[0].exact_locator["export"]
    assert materials[-1]["locator"]["export"] == registry.exports[-1].exact_locator["export"]
    assert plan.inputs[-1].kind == "source-set"
    assert plan.inputs[-1].path == registry_relative.as_posix()


@pytest.mark.parametrize(
    ("run_name", "expected_export"),
    (
        (
            "metal-budgeted-hybrid-probe-brass-sheet.yaml",
            "Brass_Sheet_Yellow_Splotchy_Streaks",
        ),
        (
            "metal-budgeted-hybrid-probe-bronze-scratched.yaml",
            "Bronze_Scratched_Yellow_Heavy_Worn_Scratches",
        ),
        (
            "metal-budgeted-hybrid-probe-aluminum-anodized.yaml",
            "Aluminum_Anodized_Pink",
        ),
        (
            "metal-budgeted-hybrid-probe-steel-painted-cracked.yaml",
            "Steel_Painted_Light_Blue_Cracked_Matte",
        ),
    ),
)
def test_characteristic_metal_probe_freezes_one_native_source(
    run_name: str,
    expected_export: str,
) -> None:
    plan = TrainingPlanResolver(PROJECT_ROOT).resolve(
        PROJECT_ROOT / "configs/training/runs" / run_name,
        devices=(5, 6, 7, 8, 9),
    )
    config = plan.to_runtime_config()
    materials = config.source["materials"]

    assert len(materials) == 1
    assert expected_export in materials[0]["locator"]["export"]
    assert config.model_context["profile_id"] == "metal_budgeted_hybrid_v3"
    assert config.total_steps == 2048
    assert plan.execution.devices == (5, 6, 7, 8, 9)
