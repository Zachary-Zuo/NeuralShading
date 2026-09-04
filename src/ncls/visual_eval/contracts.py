from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Any, Literal, Mapping, cast

from ncls.core.identity import require_sha256, safe_relative_uri, sha256_json


VisualEvalState = Literal[
    "pending",
    "claimed",
    "completed",
    "failed",
    "skipped-capacity",
    "expired",
]
SnapshotReadiness = Literal["diagnostic", "formal"]
_PUBLIC_KEY = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_STATES = {
    "pending",
    "claimed",
    "completed",
    "failed",
    "skipped-capacity",
    "expired",
}


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a nonempty object")
    if any(not isinstance(key, str) or not key for key in value):
        raise ValueError(f"{name} keys must be nonempty strings")
    return value


def _exact(name: str, value: Mapping[str, Any], fields: set[str]) -> None:
    if set(value) != fields:
        raise ValueError(f"{name} fields must be exactly {sorted(fields)}")


def derive_probe_id(run_identity: str, cadence_index: int, visual_seed: int) -> str:
    require_sha256("visual eval run identity", run_identity)
    if cadence_index < 0 or visual_seed < 0:
        raise ValueError("visual eval cadence index and seed must be nonnegative")
    return sha256_json(
        {
            "run_identity": run_identity,
            "cadence_index": cadence_index,
            "visual_seed": visual_seed,
        }
    )


@dataclass(frozen=True)
class DiagnosticSnapshot:
    uri: str
    sha256: str
    readiness: SnapshotReadiness
    format_name: str
    format_version: int

    def __post_init__(self) -> None:
        safe_relative_uri(self.uri)
        require_sha256("visual eval snapshot", self.sha256)
        if self.readiness not in {"diagnostic", "formal"}:
            raise ValueError("visual eval snapshot readiness is invalid")
        if not self.format_name or self.format_version < 1:
            raise ValueError("visual eval snapshot format identity is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "sha256": self.sha256,
            "readiness": self.readiness,
            "format_name": self.format_name,
            "format_version": self.format_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DiagnosticSnapshot":
        fields = {"uri", "sha256", "readiness", "format_name", "format_version"}
        _exact("diagnostic snapshot", value, fields)
        return cls(
            str(value["uri"]),
            str(value["sha256"]),
            cast(SnapshotReadiness, str(value["readiness"])),
            str(value["format_name"]),
            int(value["format_version"]),
        )


@dataclass(frozen=True)
class VisualEvalRequest:
    run_identity: str
    plan_identity: str
    method_key: str
    global_step: int
    cadence_index: int
    visual_seed: int
    snapshot: DiagnosticSnapshot
    source: Mapping[str, Any]
    camera: Mapping[str, Any]
    lighting: Mapping[str, Any]
    renderer: Mapping[str, Any]
    reference_spp: int = 1024
    format_name: str = "ncls.visual-eval-request"
    format_version: int = 1

    def __post_init__(self) -> None:
        if self.format_name != "ncls.visual-eval-request" or self.format_version != 1:
            raise ValueError("unsupported visual eval request format")
        require_sha256("visual eval run identity", self.run_identity)
        require_sha256("visual eval plan identity", self.plan_identity)
        if not _PUBLIC_KEY.fullmatch(self.method_key) or "@" in self.method_key:
            raise ValueError("visual eval method key must be a short public key")
        if min(self.global_step, self.cadence_index, self.visual_seed) < 0:
            raise ValueError("visual eval step, cadence and seed must be nonnegative")
        if self.reference_spp != 1024:
            raise ValueError("visual eval request v1 requires exactly 1024 reference spp")
        for name in ("source", "camera", "lighting", "renderer"):
            object.__setattr__(
                self,
                name,
                MappingProxyType(dict(_mapping(f"visual eval {name}", getattr(self, name)))),
            )
        neural_mode = str(self.renderer.get("neural_mode", "path-tracing"))
        neural_spp = int(self.renderer.get("neural_spp", 16))
        if neural_mode not in {"deferred", "path-tracing"}:
            raise ValueError("visual eval neural mode must be deferred or path-tracing")
        if neural_mode == "deferred" and neural_spp != 0:
            raise ValueError("visual eval deferred neural mode requires neural_spp=0")
        if neural_mode == "path-tracing" and not 1 <= neural_spp <= self.reference_spp:
            raise ValueError("visual eval path-tracing neural spp must be within [1, reference_spp]")

    @property
    def neural_mode(self) -> str:
        return str(self.renderer.get("neural_mode", "path-tracing"))

    @property
    def neural_spp(self) -> int:
        return int(self.renderer.get("neural_spp", 16))

    @property
    def probe_id(self) -> str:
        return derive_probe_id(self.run_identity, self.cadence_index, self.visual_seed)

    @property
    def request_id(self) -> str:
        return sha256_json(self._content_dict())

    def _content_dict(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name,
            "format_version": self.format_version,
            "run_identity": self.run_identity,
            "plan_identity": self.plan_identity,
            "method_key": self.method_key,
            "global_step": self.global_step,
            "cadence_index": self.cadence_index,
            "visual_seed": self.visual_seed,
            "probe_id": self.probe_id,
            "snapshot": self.snapshot.to_dict(),
            "source": dict(self.source),
            "camera": dict(self.camera),
            "lighting": dict(self.lighting),
            "renderer": dict(self.renderer),
            "reference_spp": self.reference_spp,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "request_id": self.request_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VisualEvalRequest":
        fields = {
            "format_name",
            "format_version",
            "run_identity",
            "plan_identity",
            "method_key",
            "global_step",
            "cadence_index",
            "visual_seed",
            "probe_id",
            "snapshot",
            "source",
            "camera",
            "lighting",
            "renderer",
            "reference_spp",
            "request_id",
        }
        _exact("visual eval request", value, fields)
        request = cls(
            str(value["run_identity"]),
            str(value["plan_identity"]),
            str(value["method_key"]),
            int(value["global_step"]),
            int(value["cadence_index"]),
            int(value["visual_seed"]),
            DiagnosticSnapshot.from_dict(
                _mapping("visual eval snapshot", value["snapshot"])
            ),
            _mapping("visual eval source", value["source"]),
            _mapping("visual eval camera", value["camera"]),
            _mapping("visual eval lighting", value["lighting"]),
            _mapping("visual eval renderer", value["renderer"]),
            int(value["reference_spp"]),
            str(value["format_name"]),
            int(value["format_version"]),
        )
        if value["probe_id"] != request.probe_id or value["request_id"] != request.request_id:
            raise ValueError("visual eval request identity mismatch")
        return request


@dataclass(frozen=True)
class VisualArtifact:
    uri: str
    sha256: str
    color_space: str

    def __post_init__(self) -> None:
        safe_relative_uri(self.uri)
        require_sha256("visual eval artifact", self.sha256)
        if not self.color_space:
            raise ValueError("visual eval artifact color space is required")

    def to_dict(self) -> dict[str, str]:
        return {
            "uri": self.uri,
            "sha256": self.sha256,
            "color_space": self.color_space,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VisualArtifact":
        _exact("visual artifact", value, {"uri", "sha256", "color_space"})
        return cls(str(value["uri"]), str(value["sha256"]), str(value["color_space"]))


@dataclass(frozen=True)
class VisualEvalResult:
    request_id: str
    probe_id: str
    worker_identity: str
    artifacts: Mapping[str, VisualArtifact]
    capture_manifest: VisualArtifact
    format_name: str = "ncls.visual-eval-result"
    format_version: int = 1

    def __post_init__(self) -> None:
        if self.format_name != "ncls.visual-eval-result" or self.format_version != 1:
            raise ValueError("unsupported visual eval result format")
        require_sha256("visual eval result request", self.request_id)
        require_sha256("visual eval result probe", self.probe_id)
        if not self.worker_identity:
            raise ValueError("visual eval worker identity is required")
        artifacts = dict(self.artifacts)
        required = {"reference", "neural", "difference", "display"}
        if set(artifacts) != required or any(
            not isinstance(item, VisualArtifact) for item in artifacts.values()
        ):
            raise ValueError(f"visual eval result artifacts must be exactly {sorted(required)}")
        object.__setattr__(self, "artifacts", MappingProxyType(artifacts))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name,
            "format_version": self.format_version,
            "request_id": self.request_id,
            "probe_id": self.probe_id,
            "worker_identity": self.worker_identity,
            "artifacts": {
                name: artifact.to_dict() for name, artifact in self.artifacts.items()
            },
            "capture_manifest": self.capture_manifest.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VisualEvalResult":
        fields = {
            "format_name",
            "format_version",
            "request_id",
            "probe_id",
            "worker_identity",
            "artifacts",
            "capture_manifest",
        }
        _exact("visual eval result", value, fields)
        artifacts = _mapping("visual eval result artifacts", value["artifacts"])
        return cls(
            str(value["request_id"]),
            str(value["probe_id"]),
            str(value["worker_identity"]),
            {
                name: VisualArtifact.from_dict(
                    _mapping(f"visual eval artifact {name}", artifact)
                )
                for name, artifact in artifacts.items()
            },
            VisualArtifact.from_dict(
                _mapping("visual eval capture manifest", value["capture_manifest"])
            ),
            str(value["format_name"]),
            int(value["format_version"]),
        )


@dataclass(frozen=True)
class VisualEvalStatus:
    request_id: str
    state: VisualEvalState
    worker_identity: str | None = None
    message: str | None = None
    format_name: str = "ncls.visual-eval-status"
    format_version: int = 1

    def __post_init__(self) -> None:
        require_sha256("visual eval status request", self.request_id)
        if self.state not in _STATES:
            raise ValueError("visual eval status state is invalid")
        if self.format_name != "ncls.visual-eval-status" or self.format_version != 1:
            raise ValueError("unsupported visual eval status format")
        if self.state == "claimed" and not self.worker_identity:
            raise ValueError("claimed visual eval status requires a worker identity")
        if self.state in {"failed", "skipped-capacity", "expired"} and not self.message:
            raise ValueError("terminal visual eval failure status requires a message")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name,
            "format_version": self.format_version,
            "request_id": self.request_id,
            "state": self.state,
            "worker_identity": self.worker_identity,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VisualEvalStatus":
        fields = {
            "format_name",
            "format_version",
            "request_id",
            "state",
            "worker_identity",
            "message",
        }
        _exact("visual eval status", value, fields)
        return cls(
            str(value["request_id"]),
            cast(VisualEvalState, str(value["state"])),
            None if value["worker_identity"] is None else str(value["worker_identity"]),
            None if value["message"] is None else str(value["message"]),
            str(value["format_name"]),
            int(value["format_version"]),
        )
