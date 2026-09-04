from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol


TrainingEventKind = Literal[
    "run-started",
    "phase-started",
    "step-completed",
    "validation-completed",
    "checkpoint-committed",
    "visual-eval-requested",
    "visual-eval-completed",
    "run-completed",
    "run-failed",
]
HookFailurePolicy = Literal["fatal", "diagnostic"]
_EVENT_KINDS = {
    "run-started",
    "phase-started",
    "step-completed",
    "validation-completed",
    "checkpoint-committed",
    "visual-eval-requested",
    "visual-eval-completed",
    "run-completed",
    "run-failed",
}


@dataclass(frozen=True)
class TrainingEvent:
    kind: TrainingEventKind
    global_step: int
    rank: int
    world_size: int
    phase_name: str | None = None
    scalars: Mapping[str, float] = MappingProxyType({})
    artifacts: Mapping[str, str] = MappingProxyType({})
    details: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if self.kind not in _EVENT_KINDS:
            raise ValueError(f"unsupported training event kind {self.kind!r}")
        if self.global_step < 0 or self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("training event step/rank is invalid")
        scalars = {str(name): float(value) for name, value in self.scalars.items()}
        if any(not name or not math.isfinite(value) for name, value in scalars.items()):
            raise ValueError("training event scalars must be named and finite")
        artifacts = {str(name): str(value) for name, value in self.artifacts.items()}
        if any(not name or not value for name, value in artifacts.items()):
            raise ValueError("training event artifacts must be named and nonempty")
        object.__setattr__(self, "phase_name", None if self.phase_name is None else str(self.phase_name))
        object.__setattr__(self, "scalars", MappingProxyType(scalars))
        object.__setattr__(self, "artifacts", MappingProxyType(artifacts))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


class TrainingHook(Protocol):
    def handle(self, event: TrainingEvent) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class HookBinding:
    name: str
    hook: TrainingHook
    failure_policy: HookFailurePolicy
    rank_zero_only: bool

    def __post_init__(self) -> None:
        if not self.name or self.failure_policy not in {"fatal", "diagnostic"}:
            raise ValueError("hook binding identity or failure policy is invalid")


@dataclass(frozen=True)
class HookFailure:
    hook_name: str
    operation: str
    event_kind: str | None
    message: str


class TrainingEventBus:
    def __init__(self, bindings: tuple[HookBinding, ...]) -> None:
        if len({item.name for item in bindings}) != len(bindings):
            raise ValueError("training hook names must be unique")
        self._bindings = tuple(bindings)
        self._failures: list[HookFailure] = []
        self._closed = False

    @property
    def failures(self) -> tuple[HookFailure, ...]:
        return tuple(self._failures)

    def _invoke(
        self,
        binding: HookBinding,
        operation: str,
        callback: Any,
        event_kind: str | None,
    ) -> None:
        try:
            callback()
        except Exception as error:
            failure = HookFailure(binding.name, operation, event_kind, str(error))
            self._failures.append(failure)
            if binding.failure_policy == "fatal":
                raise RuntimeError(
                    f"fatal training hook {binding.name!r} failed during {operation}"
                ) from error

    def emit(self, event: TrainingEvent) -> None:
        if self._closed:
            raise RuntimeError("training event bus is closed")
        for binding in self._bindings:
            if binding.rank_zero_only and event.rank != 0:
                continue
            self._invoke(
                binding,
                "handle",
                lambda binding=binding: binding.hook.handle(event),
                event.kind,
            )

    def flush(self) -> None:
        if self._closed:
            raise RuntimeError("training event bus is closed")
        for binding in self._bindings:
            self._invoke(binding, "flush", binding.hook.flush, None)

    def close(self) -> None:
        if self._closed:
            return
        fatal_error: Exception | None = None
        for binding in reversed(self._bindings):
            try:
                self._invoke(binding, "close", binding.hook.close, None)
            except Exception as error:
                if fatal_error is None:
                    fatal_error = error
        self._closed = True
        if fatal_error is not None:
            raise fatal_error


__all__ = [
    "HookBinding",
    "HookFailure",
    "HookFailurePolicy",
    "TrainingEvent",
    "TrainingEventBus",
    "TrainingEventKind",
    "TrainingHook",
]
