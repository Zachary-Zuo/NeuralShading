from __future__ import annotations

import pytest

from ncls.learning.training import TrainingConfig


def test_v4_training_config_keeps_immutable_constant_schedule_payload() -> None:
    config = TrainingConfig(schema_version=4)
    payload = config.to_dict()
    assert "learning_rate_schedule" not in payload
    assert "final_learning_rate_fraction" not in payload
    with pytest.raises(ValueError, match="v4 only supports"):
        TrainingConfig(schema_version=4, learning_rate_schedule="cosine")


def test_v5_training_config_versions_cosine_schedule() -> None:
    config = TrainingConfig(
        schema_version=5,
        learning_rate_schedule="cosine",
        final_learning_rate_fraction=0.05,
    )
    assert config.to_dict()["learning_rate_schedule"] == "cosine"
    assert config.to_dict()["final_learning_rate_fraction"] == 0.05
    with pytest.raises(ValueError, match="lie in"):
        TrainingConfig(schema_version=5, final_learning_rate_fraction=1.1)
