from pathlib import Path

import pytest
import torch

from ncls.learning.methods import get_method
from ncls.learning.training.checkpoint import TrainingCheckpoint, load_checkpoint, save_checkpoint
from ncls.learning.training.plan import TrainingPlanResolver
from ncls.paths import PROJECT_ROOT


def test_current_state_roundtrip_preserves_optimizer_and_provenance(tmp_path):
    plan = TrainingPlanResolver(PROJECT_ROOT).resolve("configs/training/runs/nvidia-layer-stack-smoke.yaml")
    method = get_method(plan.selection.method)
    model = method.create_trainable(plan.training.model_context)
    state = TrainingCheckpoint(
        method.key, plan.training.to_dict(), method.export_training_state(model),
        global_step=2, phase_name="complete", phase_optimization_state={"optimizer": {"moment": torch.ones(3)}},
        resolved_plan=plan.to_dict(), provenance={"implementation": "an earlier source revision"},
    )
    path = tmp_path / "state.pt"
    save_checkpoint(path, state)
    restored = load_checkpoint(path)
    method.restore_training_state(model, restored.model_state)
    assert torch.equal(restored.phase_optimization_state["optimizer"]["moment"], torch.ones(3))
    assert restored.provenance == state.provenance
    assert list(tmp_path.iterdir()) == [path]


def test_model_restore_rejects_broadcastable_wrong_shape():
    plan = TrainingPlanResolver(PROJECT_ROOT).resolve("configs/training/runs/nvidia-layer-stack-smoke.yaml")
    method = get_method("nvidia")
    model = method.create_trainable(plan.training.model_context)
    state = dict(method.export_training_state(model))
    name = next(name for name in dict(model.named_parameters()) if name in state and state[name].ndim == 2)
    state[name] = state[name][:1, :1]
    with pytest.raises(ValueError, match="shape"):
        method.restore_training_state(model, state)
