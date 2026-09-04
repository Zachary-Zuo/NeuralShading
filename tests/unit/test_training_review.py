from __future__ import annotations

from pathlib import Path

import pytest

from ncls.learning.training import load_metric_rows


def test_metric_loader_allows_empty_step_zero_input_when_explicit(tmp_path: Path) -> None:
    metrics = tmp_path / "checkpoint.metrics.jsonl"
    metrics.write_text("", encoding="utf-8")

    assert load_metric_rows(
        metrics,
        config_sha256="config",
        allow_empty=True,
    ) == []


def test_metric_loader_rejects_empty_training_input_by_default(tmp_path: Path) -> None:
    metrics = tmp_path / "checkpoint.metrics.jsonl"
    metrics.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="requires metric rows"):
        load_metric_rows(metrics, config_sha256="config")
