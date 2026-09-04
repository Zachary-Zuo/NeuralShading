from pathlib import Path

import pytest

from ncls.core.identity import sha256_json
from ncls.learning.methods import get_method_plugin
from ncls.learning.training import LegacyCheckpointV4Importer, TrainingPlanResolver
from ncls.learning.training.checkpoint import TrainingCheckpoint, save_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _checkpoint() -> TrainingCheckpoint:
    plugin = get_method_plugin("nvidia")
    descriptor = plugin.descriptor
    config = TrainingPlanResolver(PROJECT_ROOT).resolve(
        "configs/training/runs/nvidia-layer-stack-smoke.yaml"
    ).to_runtime_config()
    model = plugin.model_factory.create(config.model_context)
    state = plugin.checkpoint.encode(model)
    phase_index, phase_step = config.locate_step(1)
    phase_name = config.phases[phase_index].name
    manifest = {
        "schema": "ncls.method-components@1",
        "parameter_groups": {
            name: list(values) for name, values in descriptor.parameter_groups.items()
        },
        "components": [item.to_dict() for item in descriptor.components],
    }
    return TrainingCheckpoint(
        descriptor.method_key,
        descriptor.descriptor_sha256,
        descriptor.implementation_sha256,
        manifest,
        config.to_dict(),
        config.sha256,
        sha256_json([phase.to_dict() for phase in config.phases]),
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
        tuple(item.to_dict() for item in descriptor.supported_sources),
        ("a" * 64,),
        1,
        phase_index,
        phase_name,
        phase_step,
        {},
        state,
        {
            "phase_name": phase_name,
            "optimizer": {},
            "scheduler": {},
            "precision": {},
        },
        {},
        {},
        {
            name: {
                "finite_observed": False,
                "nonzero_gradient_observed": False,
                "parameter_update_observed": False,
                "last_audit_step": -1,
            }
            for name in descriptor.parameter_groups
        },
        {},
    )


def test_legacy_checkpoint_import_is_read_only_evaluation_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, _checkpoint())
    snapshot = LegacyCheckpointV4Importer().load(path)
    assert snapshot.legacy_v4
    assert snapshot.public_method_key == "nvidia"
    assert snapshot.global_step == 1
    assert snapshot.require_ready("visual-diagnostic")["ready"]
    assert not hasattr(snapshot, "optimizer_state")
    assert not hasattr(snapshot, "to_runner_checkpoint")
    assert set(snapshot.deployment_payload) == {
        "model_state",
        "training_config",
        "source_snapshot_ids",
    }


def test_legacy_checkpoint_import_rejects_tampered_payload(tmp_path: Path) -> None:
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, _checkpoint())
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        LegacyCheckpointV4Importer().load(path)
