from __future__ import annotations

from pathlib import Path

from ncls.core.identity import sha256_file
from ncls.learning.methods import get_method_plugin
from ncls.learning.training.checkpoint_v1 import load_training_checkpoint_v1
from ncls.learning.training.evaluation_snapshot import EvaluationSnapshot
from ncls.learning.training.plan import ResolvedTrainingPlan


def load_deployment_snapshot(path: Path | str) -> EvaluationSnapshot:
    """用当前 compiler 部署冻结权重，分别记录训练实现与部署实现。

    仅消费当前 checkpoint schema。训练恢复/评测仍走严格的
    load_evaluation_snapshot；部署不因 shader 改动要求重训。
    """
    path = Path(path)
    checkpoint = load_training_checkpoint_v1(path, map_location="cpu")
    plan = ResolvedTrainingPlan.from_manifest(checkpoint.plan_manifest)
    plugin = get_method_plugin(str(checkpoint.method["public_key"]))
    if checkpoint.method["implementation_key"] != plugin.descriptor.method_key:
        raise ValueError("deployment checkpoint method key does not match the compiler")
    config = plan.to_runtime_config().to_dict()
    model = plugin.model_factory.create(config["model_context"])
    template = plugin.checkpoint.encode(model)
    if set(template) != set(checkpoint.model_state):
        raise ValueError("deployment checkpoint tensor names drifted from the compiler")
    for name, expected in template.items():
        actual = checkpoint.model_state[name]
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(f"deployment checkpoint tensor shape/dtype drifted: {name}")
    plugin.checkpoint.restore(model, checkpoint.model_state)
    groups = frozenset(
        group for component in plugin.descriptor.components if component.required
        for group in component.parameter_groups
    )
    failed = sorted(group for group in groups if not all(
        checkpoint.gradient_coverage.get(group, {}).get(field, False)
        for field in ("finite_observed", "nonzero_gradient_observed", "parameter_update_observed")
    ))
    exact = checkpoint.method["implementation_sha256"] == plugin.descriptor.implementation_sha256
    reasons = []
    if checkpoint.phase_name != "complete":
        reasons.append("deployment requires a completed checkpoint")
    if failed:
        reasons.append(f"required gradient/update coverage is incomplete: {failed}")
    readiness = {
        "schema": "ncls.deployment-readiness@1", "ready": not reasons,
        "exact_method_identity": exact, "complete_training": checkpoint.phase_name == "complete",
        "required_groups": sorted(groups), "failed_groups": failed, "reasons": reasons,
        "training_method": dict(checkpoint.method),
        "runtime_implementation_sha256": plugin.descriptor.implementation_sha256,
        "runtime_validation": "gpu-parity-required",
    }
    return EvaluationSnapshot(
        plugin.key, plugin.descriptor.method_key, sha256_file(path),
        checkpoint.global_step, checkpoint.phase_name, config["source"],
        tuple(checkpoint.data_identity["source_snapshot_ids"]),
        {name: str(checkpoint.data_identity[name]) for name in (
            "data_execution_plan_identity", "reference_program_identity",
            "reference_execution_plan_identity", "native_asset_collection_identity", "query_stream_identity",
        )},
        {
            "model_state": dict(checkpoint.model_state), "training_config": config,
            "source_snapshot_ids": list(checkpoint.data_identity["source_snapshot_ids"]),
            "resolved_plan": plan.to_dict(), "training_method": dict(checkpoint.method),
        },
        {"diagnostic-evaluator": readiness}, False,
    )
