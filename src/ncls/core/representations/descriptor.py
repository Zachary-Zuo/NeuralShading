from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RepresentationDescriptor:
    representation_id: str
    representation_version: int
    display_name: str
    parameter_count: int
    state_bytes: int
    bounded: bool
    status: str

    def __post_init__(self) -> None:
        if not self.representation_id or self.representation_version < 1 or not self.display_name:
            raise ValueError("representation identity must be valid")
        if self.parameter_count < 0 or self.state_bytes < 0:
            raise ValueError("representation sizes must be nonnegative")
        if self.status not in {"research-baseline", "candidate", "retired"}:
            raise ValueError("unsupported representation status")
