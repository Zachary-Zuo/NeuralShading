from pathlib import Path

import pytest

from ncls.learning.training.plan import TrainingPlanResolver


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_training_yaml_rejects_duplicate_and_unknown_run_keys(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    _write(
        duplicate,
        """\
format_name: ncls.training-run
format_version: 1
compose: {method: nvidia, data: fixture, recipe: fixture}
compose: {method: metal, data: fixture, recipe: fixture}
""",
    )
    resolver = TrainingPlanResolver(tmp_path)
    with pytest.raises(ValueError, match="duplicate YAML key 'compose'"):
        resolver.resolve(duplicate)

    unknown = tmp_path / "unknown.yaml"
    _write(
        unknown,
        """\
format_name: ncls.training-run
format_version: 1
compose: {method: nvidia, data: fixture, recipe: fixture}
surprise: true
""",
    )
    with pytest.raises(ValueError, match=r"unknown \['surprise'\]"):
        resolver.resolve(unknown)


def test_training_yaml_rejects_fragment_inheritance_cycle(tmp_path: Path) -> None:
    base = tmp_path / "configs" / "training" / "base"
    template = """\
format_name: ncls.training-fragment
format_version: 1
kind: base
key: {key}
extends: {parent}
compatible_methods: []
payload: {{}}
"""
    _write(base / "default.yaml", template.format(key="default", parent="alternate"))
    _write(base / "alternate.yaml", template.format(key="alternate", parent="default"))
    run = tmp_path / "run.yaml"
    _write(
        run,
        """\
format_name: ncls.training-run
format_version: 1
compose: {method: nvidia, data: fixture, recipe: fixture}
""",
    )

    with pytest.raises(
        ValueError,
        match="training fragment inheritance cycle: default -> alternate -> default",
    ):
        TrainingPlanResolver(tmp_path).resolve(run)


def test_training_yaml_public_keys_reject_version_suffix(tmp_path: Path) -> None:
    run = tmp_path / "run.yaml"
    _write(
        run,
        """\
format_name: ncls.training-run
format_version: 1
compose: {method: nvidia@1, data: fixture, recipe: fixture}
""",
    )
    with pytest.raises(ValueError, match="without a version suffix"):
        TrainingPlanResolver(tmp_path).resolve(run)
