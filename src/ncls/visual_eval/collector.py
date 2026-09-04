from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ncls.core.identity import sha256_file
from ncls.learning.training.events import TrainingEvent, TrainingEventBus

from .contracts import VisualArtifact
from .spool import VisualEvalSpool


class VisualEvalCollector:
    def __init__(
        self,
        spool: VisualEvalSpool,
        artifact_root: Path | str,
        *,
        rank: int,
        world_size: int,
    ) -> None:
        if rank != 0:
            raise ValueError("visual eval collector may only run on rank 0")
        if world_size < 1:
            raise ValueError("visual eval collector world size must be positive")
        self.spool = spool
        self.artifact_root = Path(artifact_root).resolve()
        self.rank = rank
        self.world_size = world_size
        self._collected: set[str] = set()

    def _artifact_path(self, artifact: VisualArtifact) -> Path:
        path = (self.artifact_root / Path(*artifact.uri.split("/"))).resolve()
        try:
            path.relative_to(self.artifact_root)
        except ValueError as error:
            raise ValueError("visual eval artifact resolves outside artifact root") from error
        if not path.is_file() or sha256_file(path) != artifact.sha256:
            raise ValueError(f"visual eval artifact {artifact.uri!r} is missing or tampered")
        return path

    def collect(self, event_bus: TrainingEventBus) -> int:
        count = 0
        for result in self.spool.completed_results():
            if result.request_id in self._collected:
                continue
            request = self.spool.load_request(result.request_id)
            artifacts = {
                name: str(self._artifact_path(artifact))
                for name, artifact in result.artifacts.items()
            }
            self._artifact_path(result.capture_manifest)
            event_bus.emit(
                TrainingEvent(
                    "visual-eval-completed",
                    request.global_step,
                    self.rank,
                    self.world_size,
                    artifacts=artifacts,
                    details={
                        "request_id": result.request_id,
                        "probe_id": result.probe_id,
                        "worker_identity": result.worker_identity,
                        "reference_spp": request.reference_spp,
                        "neural_mode": request.neural_mode,
                        "neural_spp": request.neural_spp,
                    },
                )
            )
            self._collected.add(result.request_id)
            count += 1
        return count

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "format_name": "ncls.visual-eval-collector",
            "format_version": 1,
            "collected_request_ids": sorted(self._collected),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        required = {"format_name", "format_version", "collected_request_ids"}
        if set(state) != required:
            raise ValueError("visual eval collector state fields are invalid")
        if (
            state["format_name"] != "ncls.visual-eval-collector"
            or int(state["format_version"]) != 1
        ):
            raise ValueError("unsupported visual eval collector state format")
        values = tuple(str(item) for item in state["collected_request_ids"])
        if len(set(values)) != len(values):
            raise ValueError("visual eval collector state repeats request identities")
        self._collected = set(values)


__all__ = ["VisualEvalCollector"]
