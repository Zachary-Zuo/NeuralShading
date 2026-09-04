from dataclasses import replace
from pathlib import Path

from PIL import Image
import pytest

from ncls.core.identity import sha256_file
from ncls.learning.training import HookBinding, TrainingEventBus
from ncls.visual_eval import (
    DiagnosticSnapshot,
    VisualArtifact,
    VisualEvalRequest,
    VisualEvalResult,
    VisualEvalCollector,
    VisualEvalSpool,
)


def _request(cadence_index: int = 0) -> VisualEvalRequest:
    return VisualEvalRequest(
        run_identity="a" * 64,
        plan_identity="b" * 64,
        method_key="metal",
        global_step=5000 * (cadence_index + 1),
        cadence_index=cadence_index,
        visual_seed=20260904,
        snapshot=DiagnosticSnapshot(
            "snapshots/step.pt",
            "c" * 64,
            "diagnostic",
            "ncls.evaluation-snapshot",
            1,
        ),
        source={"locator": {"kind": "fixture"}, "parameter_state": {"roughness": 0.2}},
        camera={"eye": [0.0, 0.0, 3.0], "target": [0.0, 0.0, 0.0]},
        lighting={"environment": "studio"},
        renderer={
            "name": "ncls-viewer",
            "backend": "d3d12",
            "neural_mode": "deferred",
            "neural_spp": 0,
        },
    )


def _result(request: VisualEvalRequest, worker: str = "windows-4090") -> VisualEvalResult:
    artifacts = {
        name: VisualArtifact(
            f"captures/{request.request_id}/{name}.{extension}",
            digest * 64,
            color_space,
        )
        for name, extension, digest, color_space in (
            ("reference", "exr", "1", "linear-srgb"),
            ("neural", "exr", "2", "linear-srgb"),
            ("difference", "exr", "3", "linear-srgb"),
            ("display", "png", "4", "srgb"),
        )
    }
    return VisualEvalResult(
        request.request_id,
        request.probe_id,
        worker,
        artifacts,
        VisualArtifact(
            f"captures/{request.request_id}/capture.json", "5" * 64, "data"
        ),
    )


def test_visual_eval_request_identity_is_independent_from_training_rng() -> None:
    first = _request(0)
    repeated = _request(0)
    next_probe = _request(1)

    assert first.probe_id == repeated.probe_id
    assert first.request_id == repeated.request_id
    assert next_probe.probe_id != first.probe_id
    assert VisualEvalRequest.from_dict(first.to_dict()) == first
    assert first.neural_mode == "deferred"
    assert first.neural_spp == 0
    with pytest.raises(ValueError, match="exactly 1024"):
        replace(first, reference_spp=64)
    with pytest.raises(ValueError, match="deferred neural mode"):
        replace(first, renderer={**first.renderer, "neural_spp": 2048})
    with pytest.raises(ValueError, match="neural mode"):
        replace(first, renderer={**first.renderer, "neural_mode": "invalid"})


def test_visual_eval_spool_claim_complete_and_repeat_are_idempotent(
    tmp_path: Path,
) -> None:
    spool = VisualEvalSpool(tmp_path, capacity=2)
    request = _request()
    assert spool.publish(request).state == "pending"
    assert spool.publish(request).state == "pending"

    claim = spool.claim_next("windows-4090")
    assert claim is not None and claim.request == request
    assert spool.claim_next("another-worker") is None
    result = _result(request)
    assert spool.complete(result, worker_identity="windows-4090").state == "completed"
    assert spool.complete(result, worker_identity="windows-4090").state == "completed"
    assert [item.to_dict() for item in spool.completed_results()] == [result.to_dict()]


def test_visual_eval_spool_capacity_and_failure_are_explicit(tmp_path: Path) -> None:
    spool = VisualEvalSpool(tmp_path, capacity=1)
    first = _request(0)
    second = _request(1)
    assert spool.publish(first).state == "pending"
    skipped = spool.publish(second)
    assert skipped.state == "skipped-capacity"
    assert "capacity 1" in str(skipped.message)

    claim = spool.claim_next("worker")
    assert claim is not None
    failed = spool.fail(first.request_id, worker_identity="worker", message="capture failed")
    assert failed.state == "failed"
    assert spool.completed_results() is not None


def test_visual_eval_spool_rejects_non_owner_result(tmp_path: Path) -> None:
    spool = VisualEvalSpool(tmp_path, capacity=1)
    request = _request()
    spool.publish(request)
    spool.claim_next("owner")
    with pytest.raises(ValueError, match="does not own"):
        spool.complete(_result(request, "intruder"), worker_identity="intruder")


def test_visual_eval_collector_emits_verified_result_once(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    spool = VisualEvalSpool(tmp_path / "spool", capacity=1)
    request = _request()
    spool.publish(request)
    spool.claim_next("windows-4090")
    capture_root = artifact_root / "captures" / request.request_id
    capture_root.mkdir(parents=True)
    artifacts = {}
    for name in ("reference", "neural", "difference"):
        path = capture_root / f"{name}.exr"
        path.write_bytes(name.encode("ascii"))
        artifacts[name] = VisualArtifact(
            path.relative_to(artifact_root).as_posix(), sha256_file(path), "linear-srgb"
        )
    display = capture_root / "display.png"
    Image.new("RGB", (4, 2), color=(1, 2, 3)).save(display)
    artifacts["display"] = VisualArtifact(
        display.relative_to(artifact_root).as_posix(), sha256_file(display), "srgb"
    )
    manifest = capture_root / "capture.json"
    manifest.write_text("{}\n", encoding="utf-8")
    result = VisualEvalResult(
        request.request_id,
        request.probe_id,
        "windows-4090",
        artifacts,
        VisualArtifact(
            manifest.relative_to(artifact_root).as_posix(), sha256_file(manifest), "data"
        ),
    )
    spool.complete(result, worker_identity="windows-4090")

    class RecordingHook:
        def __init__(self) -> None:
            self.events = []

        def handle(self, event):
            self.events.append(event)

        def flush(self):
            pass

        def close(self):
            pass

    hook = RecordingHook()
    bus = TrainingEventBus((HookBinding("record", hook, "fatal", True),))
    collector = VisualEvalCollector(spool, artifact_root, rank=0, world_size=1)
    assert collector.collect(bus) == 1
    assert collector.collect(bus) == 0
    assert hook.events[0].kind == "visual-eval-completed"
    assert hook.events[0].global_step == request.global_step
    assert hook.events[0].artifacts["display"] == str(display.resolve())
