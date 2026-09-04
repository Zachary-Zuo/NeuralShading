from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Iterator

from .contracts import VisualEvalRequest, VisualEvalResult, VisualEvalStatus


@dataclass(frozen=True)
class ClaimedVisualEval:
    request: VisualEvalRequest
    worker_identity: str


class VisualEvalSpool:
    def __init__(self, root: Path | str, *, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("visual eval spool capacity must be positive")
        self.root = Path(root).resolve()
        self.capacity = int(capacity)
        self._pending = self.root / "pending"
        self._claimed = self.root / "claimed"
        self._completed = self.root / "completed"
        self._failed = self.root / "failed"
        self._results = self.root / "results"
        self._statuses = self.root / "status"
        for path in (
            self._pending,
            self._claimed,
            self._completed,
            self._failed,
            self._results,
            self._statuses,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _read(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError(f"visual eval spool document {path} must be an object")
        return value

    def _status_path(self, request_id: str) -> Path:
        return self._statuses / f"{request_id}.json"

    @staticmethod
    def _write_atomic(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _write_status(self, status: VisualEvalStatus) -> VisualEvalStatus:
        self._write_atomic(self._status_path(status.request_id), status.to_dict())
        return status

    def status(self, request_id: str) -> VisualEvalStatus | None:
        path = self._status_path(request_id)
        if not path.exists():
            return None
        return VisualEvalStatus.from_dict(self._read(path))

    def publish(self, request: VisualEvalRequest) -> VisualEvalStatus:
        existing = self.status(request.request_id)
        if existing is not None:
            return existing
        active_count = sum(1 for _ in self._pending.glob("*.json")) + sum(
            1 for _ in self._claimed.glob("*.json")
        )
        if active_count >= self.capacity:
            return self._write_status(
                VisualEvalStatus(
                    request.request_id,
                    "skipped-capacity",
                    message=f"visual eval spool capacity {self.capacity} is exhausted",
                )
            )
        self._write_atomic(
            self._pending / f"{request.request_id}.json", request.to_dict()
        )
        return self._write_status(VisualEvalStatus(request.request_id, "pending"))

    def claim_next(self, worker_identity: str) -> ClaimedVisualEval | None:
        if not worker_identity:
            raise ValueError("visual eval worker identity is required")
        for source in sorted(self._pending.glob("*.json")):
            pending_status = self.status(source.stem)
            if pending_status is None or pending_status.state != "pending":
                continue
            target = self._claimed / source.name
            try:
                os.replace(source, target)
            except FileNotFoundError:
                continue
            request = VisualEvalRequest.from_dict(self._read(target))
            self._write_status(
                VisualEvalStatus(request.request_id, "claimed", worker_identity)
            )
            return ClaimedVisualEval(request, worker_identity)
        return None

    def _claimed_request(self, request_id: str) -> VisualEvalRequest:
        path = self._claimed / f"{request_id}.json"
        if not path.exists():
            raise ValueError("visual eval request is not claimed")
        return VisualEvalRequest.from_dict(self._read(path))

    def load_request(self, request_id: str) -> VisualEvalRequest:
        for directory in (
            self._pending,
            self._claimed,
            self._completed,
            self._failed,
        ):
            path = directory / f"{request_id}.json"
            if path.exists():
                return VisualEvalRequest.from_dict(self._read(path))
        raise ValueError(f"visual eval request {request_id!r} does not exist")

    def complete(self, result: VisualEvalResult, *, worker_identity: str) -> VisualEvalStatus:
        existing = self.status(result.request_id)
        if existing is not None and existing.state == "completed":
            stored = VisualEvalResult.from_dict(
                self._read(self._results / f"{result.request_id}.json")
            )
            if stored.to_dict() != result.to_dict():
                raise ValueError("completed visual eval result cannot be replaced")
            return existing
        request = self._claimed_request(result.request_id)
        status = self.status(result.request_id)
        if (
            status is None
            or status.state != "claimed"
            or status.worker_identity != worker_identity
        ):
            raise ValueError("visual eval result worker does not own the claim")
        if result.worker_identity != worker_identity or result.probe_id != request.probe_id:
            raise ValueError("visual eval result identity disagrees with the claimed request")
        self._write_atomic(
            self._results / f"{result.request_id}.json", result.to_dict()
        )
        os.replace(
            self._claimed / f"{result.request_id}.json",
            self._completed / f"{result.request_id}.json",
        )
        return self._write_status(
            VisualEvalStatus(result.request_id, "completed", worker_identity)
        )

    def fail(
        self, request_id: str, *, worker_identity: str, message: str
    ) -> VisualEvalStatus:
        self._claimed_request(request_id)
        status = self.status(request_id)
        if (
            status is None
            or status.state != "claimed"
            or status.worker_identity != worker_identity
        ):
            raise ValueError("visual eval worker does not own the claim")
        if not message:
            raise ValueError("visual eval failure message is required")
        os.replace(
            self._claimed / f"{request_id}.json",
            self._failed / f"{request_id}.json",
        )
        return self._write_status(
            VisualEvalStatus(request_id, "failed", worker_identity, message)
        )

    def completed_results(self) -> Iterator[VisualEvalResult]:
        for path in sorted(self._results.glob("*.json")):
            status = self.status(path.stem)
            if status is not None and status.state == "completed":
                yield VisualEvalResult.from_dict(self._read(path))
