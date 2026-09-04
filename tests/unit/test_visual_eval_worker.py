import json
from pathlib import Path
from types import SimpleNamespace

from ncls.core.identity import sha256_file
from ncls.visual_eval import (
    DiagnosticSnapshot,
    ViewerCapture,
    VisualEvalRequest,
    VisualEvalSpool,
    VisualEvalWorker,
)


def _request(artifact_root: Path) -> VisualEvalRequest:
    snapshot = artifact_root / "snapshots" / "step.pt"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"snapshot")
    digest = sha256_file(snapshot)
    return VisualEvalRequest(
        "a" * 64,
        "b" * 64,
        "nvidia",
        5000,
        0,
        7,
        DiagnosticSnapshot(
            snapshot.relative_to(artifact_root).as_posix(),
            digest,
            "diagnostic",
            "ncls.training-checkpoint",
            1,
        ),
        {"family_id": "ncls.layer-stack@1", "material_index": 0, "locator": {"kind": "fixture"}},
        {"target": [0, 0, 0], "yaw": 0, "pitch": 0, "distance": 3},
        {"preset": "viewer-studio"},
        {
            "backend": "windows-d3d12-viewer",
            "resolution": [16, 8],
            "neural_mode": "deferred",
            "neural_spp": 0,
        },
    )


class _Executor:
    def __init__(self, *, spp: int = 1024) -> None:
        self.spp = spp

    def __call__(self, request, snapshot, output_root):
        del snapshot
        output_root.mkdir(parents=True)
        reference = output_root / "capture-slot-0.exr"
        neural = output_root / "capture-slot-1.exr"
        difference = output_root / "capture-difference.exr"
        display = output_root / "capture-display.png"
        for path in (reference, neural, difference, display):
            path.write_bytes(path.name.encode("ascii"))
        manifest = output_root / "capture.json"
        manifest.write_text(
            json.dumps(
                {
                    "format_name": "ncls.viewer-capture",
                    "format_version": 4,
                    "comparison_purpose": "training-diagnostic",
                    "reference_spp": self.spp,
                    "slots": [
                        {
                            "status": "ready",
                            "mode": "path-tracing",
                            "spp": self.spp,
                            "target_spp": self.spp,
                        },
                        {
                            "status": "ready",
                            "mode": request.neural_mode,
                            "spp": 0,
                            "target_spp": 0,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return ViewerCapture(reference, neural, difference, display, manifest)


def test_visual_eval_worker_completes_reference_1024_and_neural_deferred_result(
    tmp_path: Path, monkeypatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    spool = VisualEvalSpool(artifact_root / "spool", capacity=1)
    request = _request(artifact_root)
    spool.publish(request)
    snapshot = SimpleNamespace(
        public_method_key="nvidia", require_ready=lambda mode: {"ready": mode == "visual-diagnostic"}
    )
    monkeypatch.setattr(
        "ncls.visual_eval.worker.load_evaluation_snapshot", lambda path: snapshot
    )
    worker = VisualEvalWorker(
        spool, artifact_root, _Executor(), worker_identity="windows-test"
    )
    status = worker.run_once()
    assert status is not None and status.state == "completed"
    result = tuple(spool.completed_results())[0]
    assert set(result.artifacts) == {"reference", "neural", "difference", "display"}
    assert worker.run_once() is None


def test_visual_eval_worker_records_invalid_capture_as_failure(
    tmp_path: Path, monkeypatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    spool = VisualEvalSpool(artifact_root / "spool", capacity=1)
    request = _request(artifact_root)
    spool.publish(request)
    snapshot = SimpleNamespace(
        public_method_key="nvidia", require_ready=lambda mode: {"ready": True}
    )
    monkeypatch.setattr(
        "ncls.visual_eval.worker.load_evaluation_snapshot", lambda path: snapshot
    )
    worker = VisualEvalWorker(
        spool, artifact_root, _Executor(spp=64), worker_identity="windows-test"
    )
    status = worker.run_once()
    assert status is not None and status.state == "failed"
    assert "capture contract mismatch" in str(status.message)
