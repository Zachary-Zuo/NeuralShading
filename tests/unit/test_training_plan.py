from pathlib import Path
import shutil

import pytest

from ncls.learning.training import (
    ResolvedTrainingPlan,
    TrainingPlanResolver,
)


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
            "d618406461c9914cff96e257d3b4fbb333e54428476efacab7c3da9bdceeab72",
            1,
            2,
        ),
        (
            "configs/training/runs/metal-windows-smoke.yaml",
            "metal",
            "288226d90187c2dcdff6b0babf92c5cef5102a40c69b1e6652e5995ff5be71f9",
            3,
            16,
        ),
        (
            "configs/training/runs/nvidia-materialx-smoke.yaml",
            "nvidia",
            "0889992e5878d154aa96dfab05eece9016fab7e8ff1ad3d54a4e4581d86cea7b",
            1,
            2,
        ),
        (
            "configs/training/runs/nvidia-materialx-formal.yaml",
            "nvidia",
            "c47fc0105dd06aa636224ebb19762632c5116cdffc999aee69104cda41707e77",
            1,
            300000,
        ),
        (
            "configs/training/runs/nvidia-mdl-effect-pigment-smoke.yaml",
            "nvidia",
            "a4a603935349290688b12203e78a561da19b39c6c39e509d51d5d9dfcc4ee3c1",
            1,
            2,
        ),
        (
            "configs/training/runs/metal-linux-smoke.yaml",
            "metal",
            "28f754f2c5c128e32a11c110168f4a541099ccb4fbc01f61189da9a8f61b99da",
            692,
            16,
        ),
        (
            "configs/training/runs/metal-linux-long.yaml",
            "metal",
            "166f28c28f4ef58569ae459d0f90b694238585ce04c3b88d4b4e6ebe04a5ecd0",
            692,
            120000,
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
  data: mdl-metal-full
  recipe: metal-windows-smoke
""",
        encoding="utf-8",
    )

    plan = TrainingPlanResolver(tmp_path).resolve(run_path)
    expected = TrainingPlanResolver(PROJECT_ROOT).resolve(
        "configs/training/runs/metal-linux-smoke.yaml"
    )

    assert (
        plan.to_runtime_config().source["materials"]
        == expected.to_runtime_config().source["materials"]
    )
    assert len(plan.to_runtime_config().source["materials"]) == 692
    assert plan.inputs[-1].kind == "source-set"
    assert plan.inputs[-1].path == registry_relative.as_posix()
