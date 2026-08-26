"""部署预算单元门（experiment_framework.md §0.1 软线）。

`deployment_candidate=True` 的 pipeline 必须从架构本身报出成本（`parameter_costs(None)`），
并满足全部软线；注册表里没有候选时，结构性断言仍然成立。
"""
from __future__ import annotations

from ncls.learning.pipelines import create_pipeline, pipeline_descriptors


SOFT_BUDGET = {
    "C_eval_macs": 2_000,
    "C_prepare_macs": 10_000,
    "state_bytes_per_pixel": 64,
    "B_asset": 512,
    "B_evaluate_weights": 32 * 1024,
}
P1_V1_PIPELINES = {
    "film-evaluator-s-v1",
    "film-evaluator-m-v1",
    "film-evaluator-l-v1",
    "analytic-residual-s-v1",
    "analytic-residual-m-v1",
    "analytic-residual-l-v1",
    "per-state-teacher-l-v1",
}


def test_every_pipeline_declares_deployment_candidacy() -> None:
    descriptors = pipeline_descriptors()
    assert descriptors
    for descriptor in descriptors:
        assert isinstance(descriptor.runtime["deployment_candidate"], bool)
        assert descriptor.deployment_candidate is descriptor.runtime["deployment_candidate"]
    assert {item.name for item in descriptors if item.name in P1_V1_PIPELINES} == P1_V1_PIPELINES
    assert not any(item.deployment_candidate for item in descriptors if item.name in P1_V1_PIPELINES)


def test_deployment_candidates_fit_soft_budget() -> None:
    candidates = [item for item in pipeline_descriptors() if item.deployment_candidate]
    assert {item.name for item in candidates} == {
        "lobe-residual-k2-v1",
        "core-frame-neural-v1",
    }
    for descriptor in candidates:
        costs = dict(create_pipeline(descriptor.name).parameter_costs(None))
        assert set(SOFT_BUDGET) <= set(costs), descriptor.name
        for name, limit in SOFT_BUDGET.items():
            assert 0 <= int(costs[name]) <= limit, f"{descriptor.name}: {name}={costs[name]} > {limit}"


def test_lobe_residual_costs_follow_plan_section_1_2() -> None:
    k2 = dict(create_pipeline("lobe-residual-k2-v1").parameter_costs(None))
    log32 = dict(create_pipeline("lobe-residual-k2-log32-v1").parameter_costs(None))
    k3 = dict(create_pipeline("lobe-residual-k3-log32-v1").parameter_costs(None))
    assert k2["C_prepare_macs"] == 23 * 64 + 64 * 64 + 64 * 26
    assert k2["state_bytes_per_pixel"] == 48
    assert log32["state_bytes_per_pixel"] == 64
    assert k3["state_bytes_per_pixel"] > 64
    assert log32["C_eval_macs"] - k2["C_eval_macs"] == 14 * 32 + 32 * 3
    assert k2["B_evaluate_weights"] == 0 and log32["B_evaluate_weights"] == 2 * (14 * 32 + 32 + 32 * 3 + 3)
    assert k2["B_asset"] == 64 + 64
    assert {"B_shared", "parameter_count", "analytic_core_state_bytes", "C_eval_excludes_analytic_core"} <= set(k2)
