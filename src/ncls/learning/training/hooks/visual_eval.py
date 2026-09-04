from __future__ import annotations

import os
from pathlib import Path
import random
import math
import shutil
from typing import Any, Mapping

from ncls.core.identity import sha256_file, sha256_json
from ncls.visual_eval.contracts import DiagnosticSnapshot, VisualEvalRequest
from ncls.visual_eval.spool import VisualEvalSpool

from ..events import TrainingEvent
from ..plan import ResolvedTrainingPlan


class VisualEvalHook:
    """Publishes immutable rank-0 diagnostic snapshots to a durable file spool."""

    def __init__(
        self,
        plan: ResolvedTrainingPlan,
        checkpoint_path_pattern: Path | str,
        artifact_root: Path | str,
        spool: VisualEvalSpool,
        *,
        rank: int,
    ) -> None:
        if rank != 0:
            raise ValueError("VisualEvalHook may only be constructed on rank 0")
        if not plan.hooks.visual_eval.enabled:
            raise ValueError("VisualEvalHook requires an enabled visual eval plan")
        self.plan = plan
        self.checkpoint_path_pattern = Path(checkpoint_path_pattern).resolve()
        self.artifact_root = Path(artifact_root).resolve()
        self.spool = spool
        self._requested: dict[int, str] = {}
        self._run_identity = sha256_json(
            {"schema": "ncls.training-run@1", "plan_identity": plan.sha256}
        )

    @property
    def probe_ids(self) -> tuple[str, ...]:
        return tuple(self._requested[index] for index in sorted(self._requested))

    def _checkpoint_for_step(self, step: int) -> Path:
        path = self.checkpoint_path_pattern
        return path.with_name(f"{path.stem}.step{step:08d}{path.suffix}")

    def _snapshot(self, checkpoint: Path) -> DiagnosticSnapshot:
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"visual eval checkpoint was not committed before its event: {checkpoint}"
            )
        digest = sha256_file(checkpoint)
        source_sidecar = checkpoint.with_suffix(checkpoint.suffix + ".sha256")
        if not source_sidecar.is_file() or source_sidecar.read_text(
            encoding="ascii"
        ).strip() != digest:
            raise ValueError("visual eval checkpoint sidecar is missing or disagrees")
        target = self.spool.root / "snapshots" / f"{digest}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256_file(target) != digest:
                raise ValueError("immutable visual eval snapshot path was replaced")
        else:
            temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            shutil.copyfile(checkpoint, temporary)
            if sha256_file(temporary) != digest:
                temporary.unlink(missing_ok=True)
                raise ValueError("visual eval snapshot copy hash mismatch")
            os.replace(temporary, target)
        sidecar = target.with_suffix(target.suffix + ".sha256")
        if sidecar.exists():
            if sidecar.read_text(encoding="ascii").strip() != digest:
                raise ValueError("immutable visual eval snapshot sidecar was replaced")
        else:
            temporary_sidecar = sidecar.with_name(
                f".{sidecar.name}.{os.getpid()}.tmp"
            )
            temporary_sidecar.write_text(digest + "\n", encoding="ascii")
            os.replace(temporary_sidecar, sidecar)
        try:
            uri = target.relative_to(self.artifact_root).as_posix()
        except ValueError as error:
            raise ValueError("visual eval spool must reside below artifact_root") from error
        return DiagnosticSnapshot(
            uri,
            digest,
            "diagnostic",
            "ncls.training-checkpoint",
            1,
        )

    def _request(self, step: int, snapshot: DiagnosticSnapshot) -> VisualEvalRequest:
        settings = self.plan.hooks.visual_eval
        cadence_index = step // settings.interval_steps - 1
        training = dict(self.plan.training)
        source = training.get("source")
        if not isinstance(source, Mapping):
            raise ValueError("visual eval training plan has no source definition")
        materials = source.get("materials")
        if not isinstance(materials, (list, tuple)) or not materials:
            raise ValueError("visual eval training source has no materials")
        selector = sha256_json(
            {
                "run_identity": self._run_identity,
                "cadence_index": cadence_index,
                "visual_seed": settings.seed,
            }
        )
        material_index = int(selector[:16], 16) % len(materials)
        material = materials[material_index]
        if not isinstance(material, Mapping) or not isinstance(
            material.get("locator"), Mapping
        ):
            raise ValueError("visual eval selected source locator is invalid")
        generator = random.Random(int(selector[16:32], 16))
        camera = {
            "target": [-0.05485052, 1.04786098, -0.06448951],
            "yaw": generator.uniform(-math.pi, math.pi),
            "pitch": generator.uniform(-0.35, 0.55),
            "distance": generator.uniform(3.8, 4.8),
            "vertical_fov_degrees": 38.0,
        }
        return VisualEvalRequest(
            self._run_identity,
            self.plan.sha256,
            self.plan.selection.method,
            step,
            cadence_index,
            settings.seed,
            snapshot,
            {
                "family_id": str(source.get("family_id")),
                "material_index": material_index,
                "locator": dict(material["locator"]),
            },
            camera,
            {
                "preset": "viewer-studio",
                "environment_rotation": generator.uniform(0.0, 360.0),
                "environment_intensity": 1.0,
            },
            {
                "backend": "windows-d3d12-viewer",
                "resolution": [640, 360],
                "reference_samples_per_frame": 16,
                "neural_mode": settings.neural_mode,
                "neural_spp": settings.neural_spp,
            },
            settings.reference_spp,
        )

    def handle(self, event: TrainingEvent) -> None:
        if event.rank != 0 or event.kind != "checkpoint-committed":
            return
        interval = self.plan.hooks.visual_eval.interval_steps
        if event.global_step == 0 or event.global_step % interval:
            return
        cadence_index = event.global_step // interval - 1
        if cadence_index in self._requested:
            return
        request = self._request(
            event.global_step,
            self._snapshot(self._checkpoint_for_step(event.global_step)),
        )
        self.spool.publish(request)
        self._requested[cadence_index] = request.probe_id

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "format_name": "ncls.visual-eval-hook",
            "format_version": 1,
            "run_identity": self._run_identity,
            "requested": [
                {"cadence_index": index, "probe_id": probe_id}
                for index, probe_id in sorted(self._requested.items())
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        required = {"format_name", "format_version", "run_identity", "requested"}
        if set(state) != required:
            raise ValueError("visual eval hook state fields are invalid")
        if (
            state["format_name"] != "ncls.visual-eval-hook"
            or int(state["format_version"]) != 1
            or state["run_identity"] != self._run_identity
        ):
            raise ValueError("visual eval hook state identity mismatch")
        requested = {}
        for item in state["requested"]:
            index = int(item["cadence_index"])
            probe_id = str(item["probe_id"])
            if index < 0 or index in requested or len(probe_id) != 64:
                raise ValueError("visual eval hook state cursor is invalid")
            requested[index] = probe_id
        self._requested = requested

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


__all__ = ["VisualEvalHook"]
