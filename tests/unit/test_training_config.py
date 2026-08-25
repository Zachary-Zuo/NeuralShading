from __future__ import annotations

from pathlib import Path

import pytest

from ncls.learning.training import TrainingConfig


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
    assert len(paths) >= 10
    for path in paths:
        config = TrainingConfig.load(path)
        assert config.stage == "P1"
