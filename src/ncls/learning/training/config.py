from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from ncls.core.identity import sha256_json


@dataclass(frozen=True)
class TrainingPhase:
    name: str
    steps: int
    learning_rate: float

    def __post_init__(self) -> None:
        if self.name not in {"evaluator", "joint", "sampler"}:
            raise ValueError("training phase name must be evaluator, joint or sampler")
        if self.steps < 1 or self.learning_rate <= 0.0:
            raise ValueError("training phase steps and learning_rate must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "steps": self.steps, "learning_rate": self.learning_rate}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingPhase":
        if set(value) != {"name", "steps", "learning_rate"}:
            raise ValueError("training phase fields must be exactly name, steps and learning_rate")
        return cls(str(value["name"]), int(value["steps"]), float(value["learning_rate"]))


@dataclass(frozen=True)
class TrainingConfig:
    method_key: str
    batch_source: Mapping[str, Any]
    model_context: Mapping[str, Any]
    phases: tuple[TrainingPhase, ...]
    batch_size: int
    seed: int
    device: str
    optimizer: Mapping[str, Any]
    validation: Mapping[str, Any]
    checkpoint_selection: str
    format_name: str = "ncls.training-config"
    format_version: int = 1

    def __post_init__(self) -> None:
        if self.format_name != "ncls.training-config" or self.format_version != 1:
            raise ValueError("unsupported training config format")
        if not self.method_key or self.batch_size < 1 or self.seed < 0 or not self.device:
            raise ValueError("training config identity, batch size, seed and device are invalid")
        source = dict(self.batch_source)
        if set(source) != {"kind", "options"} or source["kind"] not in {"offline", "live"}:
            raise ValueError("batch_source fields must be kind/offline|live and options")
        if not isinstance(source["options"], Mapping):
            raise ValueError("batch_source options must be an object")
        phases = tuple(self.phases)
        if not phases or len({phase.name for phase in phases}) != len(phases):
            raise ValueError("training phases must be nonempty with unique names")
        optimizer = dict(self.optimizer)
        if set(optimizer) != {"kind", "weight_decay"} or optimizer["kind"] != "adamw":
            raise ValueError("optimizer must be exactly AdamW with weight_decay")
        if float(optimizer["weight_decay"]) < 0.0:
            raise ValueError("optimizer weight_decay must be nonnegative")
        validation = dict(self.validation)
        if set(validation) != {"interval", "batches"} or min(int(value) for value in validation.values()) < 1:
            raise ValueError("validation interval and batches must be positive")
        if self.checkpoint_selection != "tail_guard":
            raise ValueError("new training configs require tail_guard checkpoint selection")
        object.__setattr__(self, "batch_source", {"kind": source["kind"], "options": dict(source["options"])})
        object.__setattr__(self, "model_context", dict(self.model_context))
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "optimizer", optimizer)
        object.__setattr__(self, "validation", validation)

    @property
    def total_steps(self) -> int:
        return sum(phase.steps for phase in self.phases)

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name,
            "format_version": self.format_version,
            "method_key": self.method_key,
            "batch_source": {"kind": self.batch_source["kind"], "options": dict(self.batch_source["options"])},
            "model_context": dict(self.model_context),
            "phases": [phase.to_dict() for phase in self.phases],
            "batch_size": self.batch_size,
            "seed": self.seed,
            "device": self.device,
            "optimizer": dict(self.optimizer),
            "validation": dict(self.validation),
            "checkpoint_selection": self.checkpoint_selection,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingConfig":
        required = {
            "format_name", "format_version", "method_key", "batch_source", "model_context",
            "phases", "batch_size", "seed", "device", "optimizer", "validation",
            "checkpoint_selection",
        }
        if set(value) != required:
            raise ValueError(f"training config fields must be exactly {sorted(required)}")
        return cls(
            str(value["method_key"]), value["batch_source"], value["model_context"],
            tuple(TrainingPhase.from_dict(item) for item in value["phases"]),
            int(value["batch_size"]), int(value["seed"]), str(value["device"]),
            value["optimizer"], value["validation"], str(value["checkpoint_selection"]),
            str(value["format_name"]), int(value["format_version"]),
        )

    @classmethod
    def load(cls, path: Path | str) -> "TrainingConfig":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("training config root must be an object")
        return cls.from_dict(value)
