from dataclasses import replace
from pathlib import Path
import shutil

import pytest
import yaml

from ncls.paths import PROJECT_ROOT
from ncls.learning.training.plan import TrainingPlanResolver, ResolvedTrainingPlan


def test_all_current_configs_resolve_and_roundtrip():
    resolver = TrainingPlanResolver(PROJECT_ROOT)
    for path in (PROJECT_ROOT / "configs/training/runs").glob("*.yaml"):
        plan = resolver.resolve(path)
        restored = ResolvedTrainingPlan.from_dict(plan.to_dict())
        assert restored.to_dict() == plan.to_dict()
        assert plan.training is plan.training


@pytest.mark.parametrize("spp", [1, 33, 128, 257])
def test_yaml_alone_controls_visual_spp_and_run_settings(tmp_path, spp):
    source = PROJECT_ROOT / "configs/training/runs/nvidia-layer-stack-smoke.yaml"
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    value["hooks"] = {"visual_eval": {"reference_spp": spp, "neural_mode": "path-tracing", "neural_spp": spp + 3}}
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    resolver = TrainingPlanResolver(PROJECT_ROOT)
    baseline = resolver.resolve(source)
    edited = resolver.resolve(path, devices=(2, 4))
    assert edited.hooks.visual_eval.reference_spp == spp
    assert edited.hooks.visual_eval.neural_spp == spp + 3
    assert edited.execution.devices == (2, 4)
    assert edited.training.resume_signature == baseline.training.resume_signature
    assert replace(edited.training, checkpoint_interval=7).resume_signature == baseline.training.resume_signature


def test_yaml_reports_typo_and_inheritance_cycle(tmp_path):
    shutil.copytree(PROJECT_ROOT / "configs/training", tmp_path / "configs/training")
    run = tmp_path / "configs/training/runs/nvidia-layer-stack-smoke.yaml"
    original = run.read_text(encoding="utf-8")
    run.write_text(original + "\nhookz: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        TrainingPlanResolver(tmp_path).resolve(run)
    run.write_text(original, encoding="utf-8")
    base = tmp_path / "configs/training/base/default.yaml"
    base.write_text("extends: default\n" + base.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="成环"):
        TrainingPlanResolver(tmp_path).resolve(run)
