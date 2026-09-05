from dataclasses import replace
from pathlib import Path
import random

import pytest
import torch

from ncls.core.identity import sha256_json
from ncls.learning.methods import get_method_plugin
from ncls.learning.training import (
    TrainingCheckpointV1,
    TrainingPlanResolver,
    load_evaluation_snapshot,
    load_training_checkpoint_v1,
    save_training_checkpoint_v1,
)
from ncls.learning.training.checkpoint import TrainingCheckpoint, save_checkpoint
from ncls.learning.deployment_snapshot import load_deployment_snapshot


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_deployment_preserves_training_identity_when_runtime_code_changes(tmp_path: Path) -> None:
    plan, plugin, runner = _runner_checkpoint()
    checkpoint = TrainingCheckpointV1.from_runner_checkpoint(
        runner, plan=plan, plugin=plugin, data_execution_plan_identity="7" * 64)
    method = {**checkpoint.method, "implementation_sha256": "f" * 64}
    manifest = {**checkpoint.plan_manifest, "method_descriptor": method}
    checkpoint = replace(checkpoint, method=method, plan_manifest=manifest, plan_identity=sha256_json(manifest))
    path = tmp_path / "checkpoint.pt"
    original_hash = save_training_checkpoint_v1(path, checkpoint)
    with pytest.raises(ValueError, match="implementation drifted"):
        load_evaluation_snapshot(path)
    deployed = load_deployment_snapshot(path)
    assert deployed.checkpoint_sha256 == original_hash
    assert deployed.deployment_payload["training_method"] == method
    assert not deployed.readiness["diagnostic-evaluator"]["exact_method_identity"]
    with pytest.raises(ValueError, match="completed checkpoint"):
        deployed.require_ready("diagnostic-evaluator")
    assert load_training_checkpoint_v1(path).method == method


def test_deployment_rejects_incompatible_weight_shape(tmp_path: Path) -> None:
    plan, plugin, runner = _runner_checkpoint()
    checkpoint = TrainingCheckpointV1.from_runner_checkpoint(
        runner, plan=plan, plugin=plugin, data_execution_plan_identity="7" * 64)
    state = dict(checkpoint.model_state)
    name = next(name for name, tensor in state.items() if tensor.numel() > 1)
    state[name] = state[name].reshape(-1)[:1]
    path = tmp_path / "checkpoint.pt"
    save_training_checkpoint_v1(path, replace(checkpoint, model_state=state))
    with pytest.raises((ValueError, RuntimeError), match="shape|size|drifted"):
        load_deployment_snapshot(path)


def _runner_checkpoint():
    plan = TrainingPlanResolver(PROJECT_ROOT).resolve(
        "configs/training/runs/nvidia-layer-stack-smoke.yaml"
    )
    plugin = get_method_plugin("nvidia")
    config = plan.to_runtime_config()
    model = plugin.model_factory.create(config.model_context)
    model_state = plugin.checkpoint.encode(model)
    descriptor = plugin.descriptor
    components = {
        "schema": "ncls.method-components@1",
        "parameter_groups": {
            name: list(values) for name, values in descriptor.parameter_groups.items()
        },
        "components": [item.to_dict() for item in descriptor.components],
    }
    coverage = {
        name: {
            "finite_observed": False,
            "nonzero_gradient_observed": False,
            "parameter_update_observed": False,
            "last_audit_step": -1,
        }
        for name in descriptor.parameter_groups
    }
    checkpoint = TrainingCheckpoint(
        descriptor.method_key,
        descriptor.descriptor_sha256,
        descriptor.implementation_sha256,
        components,
        config.to_dict(),
        config.sha256,
        sha256_json([phase.to_dict() for phase in config.phases]),
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
        (descriptor.supported_sources[0].to_dict(),),
        ("5" * 64,),
        0,
        0,
        "bootstrap",
        0,
        {"policy": "tail_guard", "tail": []},
        model_state,
        {
            "phase_name": "bootstrap",
            "optimizer": {},
            "scheduler": {},
            "precision": {},
        },
        {"python": random.getstate(), "torch": torch.get_rng_state()},
        {"query_stream_identity": "4" * 64},
        coverage,
        {"rows": []},
    )
    return plan, plugin, checkpoint


def test_new_checkpoint_roundtrip_and_internal_runner_adapter(tmp_path: Path) -> None:
    plan, plugin, runner = _runner_checkpoint()
    probe_id = "6" * 64
    checkpoint = TrainingCheckpointV1.from_runner_checkpoint(
        runner,
        plan=plan,
        plugin=plugin,
        data_execution_plan_identity="7" * 64,
        hook_state={"tensorboard": {"last_step": 0}},
        visual_eval_probe_ids=(probe_id,),
    )
    assert checkpoint.format_version == 1
    assert checkpoint.method["public_key"] == "nvidia"
    assert checkpoint.plan_manifest["format_version"] == 1
    assert "training_config" not in checkpoint.to_payload()

    path = tmp_path / "checkpoint.pt"
    digest = save_training_checkpoint_v1(path, checkpoint)
    restored = load_training_checkpoint_v1(path)
    assert len(digest) == 64
    assert restored.plan_identity == plan.sha256
    assert restored.visual_eval_probe_ids == (probe_id,)
    resumed = restored.to_runner_checkpoint(plan=plan, plugin=plugin)
    assert resumed.global_step == runner.global_step
    assert resumed.training_config_sha256 == runner.training_config_sha256
    assert set(resumed.model_state) == set(runner.model_state)
    assert isinstance(resumed.rng_state["python"], tuple)
    evaluation = load_evaluation_snapshot(path)
    assert not evaluation.legacy_v4
    assert evaluation.public_method_key == "nvidia"
    assert evaluation.data_identity["query_stream_identity"] == "4" * 64
    assert evaluation.data_identity["data_execution_plan_identity"] == "7" * 64


def test_new_checkpoint_rejects_plan_tamper() -> None:
    plan, plugin, runner = _runner_checkpoint()
    checkpoint = TrainingCheckpointV1.from_runner_checkpoint(
        runner,
        plan=plan,
        plugin=plugin,
        data_execution_plan_identity="7" * 64,
    )
    with pytest.raises(ValueError, match="resolved plan hash mismatch"):
        replace(checkpoint, plan_identity="0" * 64)


def test_new_resume_loader_explicitly_rejects_v4(tmp_path: Path) -> None:
    _, _, runner = _runner_checkpoint()
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, runner)
    with pytest.raises(ValueError, match="read-only legacy input"):
        load_training_checkpoint_v1(path)
