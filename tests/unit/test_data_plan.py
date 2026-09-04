from pathlib import Path

import pytest

from ncls.data import DataExecutionPlan, DataRequirement, RankPartition
from ncls.learning.methods import get_method_plugin
from ncls.learning.training import TrainingPlanResolver


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("run", "method", "route_names"),
    (
        (
            "configs/training/runs/nvidia-layer-stack-smoke.yaml",
            "nvidia",
            ("evaluator", "sampler"),
        ),
        (
            "configs/training/runs/metal-budgeted-hybrid-pilot.yaml",
            "metal",
            ("evaluator", "sampler"),
        ),
    ),
)
def test_data_execution_plan_is_built_from_method_requirements_and_resolved_plan(
    run: str, method: str, route_names: tuple[str, ...]
) -> None:
    resolved = TrainingPlanResolver(PROJECT_ROOT).resolve(run)
    config = resolved.to_runtime_config()
    plan = DataExecutionPlan.build(
        data_key=resolved.selection.data,
        source_family_id=str(config.source["family_id"]),
        routes=[item.to_dict() for item in config.all_routes],
        requirements=get_method_plugin(method).data.requirements(),
        execution=resolved.execution.to_dict(),
        rank=1,
        world_size=2,
    )

    assert tuple(item.name for item in plan.routes) == route_names
    assert plan.partition == RankPartition(1, 2)
    assert plan.residency_budget_bytes == resolved.execution.residency_budget_mib * 1024 * 1024
    assert len(plan.identity) == 64


def test_data_execution_plan_rejects_missing_required_route() -> None:
    requirement = DataRequirement("reference-evaluator", ("wo", "wi", "target_f"))
    with pytest.raises(ValueError, match="omits required routes"):
        DataExecutionPlan.build(
            data_key="fixture",
            source_family_id="fixture.family",
            routes=[],
            requirements=(requirement,),
            execution={
                "num_workers": 0,
                "host_prefetch": 1,
                "ready_batches": 1,
                "reference_batch_steps": 1,
                "reference_inflight": 1,
                "transfer_streams": 0,
                "residency": {"budget_mib": 1},
            },
        )


def test_rank_partition_rejects_rank_outside_world() -> None:
    with pytest.raises(ValueError, match="outside the distributed world"):
        RankPartition(2, 2)


def test_checkpoint_data_identity_is_common_but_session_identity_is_rank_local() -> None:
    resolved = TrainingPlanResolver(PROJECT_ROOT).resolve(
        "configs/training/runs/nvidia-layer-stack-smoke.yaml"
    )
    config = resolved.to_runtime_config()
    arguments = {
        "data_key": resolved.selection.data,
        "source_family_id": str(config.source["family_id"]),
        "routes": [item.to_dict() for item in config.all_routes],
        "requirements": get_method_plugin("nvidia").data.requirements(),
        "execution": resolved.execution.to_dict(),
        "world_size": 2,
    }
    rank0 = DataExecutionPlan.build(**arguments, rank=0)
    rank1 = DataExecutionPlan.build(**arguments, rank=1)

    assert rank0.identity == rank1.identity
    assert rank0.session_identity != rank1.session_identity
