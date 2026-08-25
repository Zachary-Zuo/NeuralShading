from __future__ import annotations

from pathlib import Path

import pytest

from ncls.learning.training import TrainingConfig
from ncls.learning.training.selection import select_checkpoint


def test_training_config_v1_uses_readable_pipeline_and_capacity() -> None:
    config = TrainingConfig(
        pipeline="film-evaluator-s-v1",
        stage="P1",
        capacity="S",
        model={"width": 96, "latent_dimension": 24},
    )
    restored = TrainingConfig.from_json(config.to_json())
    assert restored == config
    assert restored.schema_name == "training-config"
    assert restored.schema_version == 1
    assert restored.pipeline == "film-evaluator-s-v1"
    assert len(restored.resolved_sha256) == 64


def test_training_config_has_fixed_checkpoint_selection_contract() -> None:
    config = TrainingConfig(
        pipeline="film-evaluator-m-v1",
        stage="P1",
        capacity="M",
        model={},
    )
    assert "selection_metric" not in config.to_dict()
    assert config.early_stopping_patience is None
    with pytest.raises(ValueError, match="S/M/L"):
        TrainingConfig(pipeline="film-v1", stage="P1", capacity="XL", model={})


def test_training_config_supports_bounded_adaptive_stopping() -> None:
    config = TrainingConfig(
        pipeline="film-evaluator-s-v1",
        stage="P1",
        capacity="S",
        model={},
        steps=25000,
        minimum_steps=4000,
        early_stopping_patience=8,
    )
    assert TrainingConfig.from_json(config.to_json()) == config
    with pytest.raises(ValueError, match="minimum_steps"):
        TrainingConfig(
            pipeline="film-evaluator-s-v1",
            stage="P1",
            capacity="S",
            model={},
            steps=100,
            minimum_steps=101,
        )


def test_training_config_rejects_removed_fields() -> None:
    with pytest.raises(ValueError, match="unsupported fields"):
        TrainingConfig.from_dict({
            "pipeline": "film-evaluator-s-v1",
            "stage": "P1",
            "capacity": "S",
            "model": {},
            "selection_metric": "relative-l1",
        })


def test_all_versioned_learning_configs_parse() -> None:
    paths = sorted(Path("configs/learning").rglob("*.json"))
    assert len(paths) >= 14
    for path in paths:
        config = TrainingConfig.load(path)
        assert config.stage == "P1"
        if path.stem.startswith("lobe-residual"):
            assert config.checkpoint_selection == "tail_guard"
            assert set(config.model) == {"state_count", "latent_dim", "lobe_count", "correction"}


def _validation_record(step: int, median: float, p95: float) -> dict:
    return {"step": step, "primary": {"directional_l1_by_state": {"median": median, "p95": p95}}}


def test_checkpoint_selection_defaults_to_legacy_and_keeps_its_hash() -> None:
    legacy = TrainingConfig(pipeline="film-evaluator-s-v1", stage="P1", capacity="S", model={})
    assert legacy.checkpoint_selection == "median_then_p95"
    assert "checkpoint_selection" in legacy.to_dict()
    without_field = {
        name: value for name, value in legacy.to_dict().items() if name != "checkpoint_selection"
    }
    assert TrainingConfig.from_dict(without_field).resolved_sha256 == legacy.resolved_sha256
    guarded = TrainingConfig.from_dict({**legacy.to_dict(), "checkpoint_selection": "tail_guard"})
    assert guarded.resolved_sha256 != legacy.resolved_sha256
    with pytest.raises(ValueError, match="checkpoint_selection"):
        TrainingConfig.from_dict({**legacy.to_dict(), "checkpoint_selection": "best_p95"})


def test_tail_guard_replays_p1_v1_m2s_history_to_step_7500() -> None:
    # P1 v1 M2-S（p1_audit.md §4.2）：best@4500 median 0.018 / p95 0.586；早停 7500 为 0.022 / 0.340。
    history = [
        _validation_record(4000, 0.021, 0.610),
        _validation_record(4500, 0.018, 0.586),
        _validation_record(5000, 0.020, 0.560),
        _validation_record(7500, 0.022, 0.340),
    ]
    assert select_checkpoint(history, "median_then_p95")["step"] == 4500
    assert select_checkpoint(history, "tail_guard")["step"] == 7500
    assert select_checkpoint(history[:2], "tail_guard")["step"] == 4500
    assert select_checkpoint([], "tail_guard") is None
    with pytest.raises(ValueError, match="checkpoint selection"):
        select_checkpoint(history, "best_p95")
