from pathlib import Path

from ncls.core.identity import sha256_file
from ncls.learning.training import TrainingEvent, TrainingPlanResolver
from ncls.learning.training.hooks import VisualEvalHook
from ncls.visual_eval import VisualEvalSpool


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_visual_eval_hook_publishes_immutable_deterministic_snapshot(tmp_path: Path) -> None:
    plan = TrainingPlanResolver(PROJECT_ROOT).resolve(
        "configs/training/runs/nvidia-materialx-formal.yaml"
    )
    output = tmp_path / "checkpoint.pt"
    periodic = tmp_path / "checkpoint.step00005000.pt"
    periodic.write_bytes(b"checkpoint fixture")
    digest = sha256_file(periodic)
    periodic.with_suffix(".pt.sha256").write_text(digest + "\n", encoding="ascii")
    spool = VisualEvalSpool(tmp_path / "visual-eval", capacity=2)
    hook = VisualEvalHook(plan, output, tmp_path, spool, rank=0)
    event = TrainingEvent("checkpoint-committed", 5000, 0, 1, "finetune")

    hook.handle(event)
    hook.handle(event)
    claim = spool.claim_next("worker")
    assert claim is not None
    assert claim.request.global_step == 5000
    assert claim.request.reference_spp == 1024
    assert claim.request.neural_mode == "deferred"
    assert claim.request.neural_spp == 0
    assert claim.request.method_key == "nvidia"
    assert claim.request.snapshot.sha256 == digest
    assert len(hook.probe_ids) == 1

    restored = VisualEvalHook(plan, output, tmp_path, spool, rank=0)
    restored.load_state_dict(hook.state_dict())
    assert restored.probe_ids == hook.probe_ids
