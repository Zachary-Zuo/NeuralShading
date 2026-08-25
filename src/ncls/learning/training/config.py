from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .selection import CHECKPOINT_SELECTIONS


@dataclass(frozen=True)
class TrainingConfig:
    pipeline: str
    stage: str
    capacity: str
    model: Mapping[str, Any]
    dataset_selection: Mapping[str, Any] = field(default_factory=dict)
    steps: int = 25000
    batch_size: int = 64
    learning_rate: float = 3e-4
    learning_rate_schedule: str = "cosine"
    final_learning_rate_fraction: float = 0.05
    weight_decay: float = 1e-5
    gradient_clip: float = 5.0
    validation_interval: int = 250
    checkpoint_interval: int = 250
    minimum_steps: int = 0
    early_stopping_patience: int | None = None
    seed: int = 20260824
    device: str | None = None
    deterministic: bool = True
    initialization_checkpoint: str | None = None
    checkpoint_selection: str = "median_then_p95"
    schema_name: str = "training-config"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_name != "training-config" or self.schema_version != 1:
            raise ValueError("unsupported training config")
        if not self.pipeline or not self.stage or self.capacity not in {"S", "M", "L"}:
            raise ValueError("training config requires pipeline, stage and S/M/L capacity")
        if self.learning_rate_schedule not in {"constant", "cosine"}:
            raise ValueError("unsupported learning rate schedule")
        if not 0.0 <= self.final_learning_rate_fraction <= 1.0:
            raise ValueError("final learning-rate fraction must lie in [0, 1]")
        if not isinstance(self.model, Mapping) or not isinstance(self.dataset_selection, Mapping):
            raise ValueError("model and dataset_selection must be objects")
        allowed_selection = {"state_ids", "asset_ids", "family_ids"}
        if set(self.dataset_selection) - allowed_selection:
            raise ValueError("dataset_selection contains unsupported fields")
        for name, values in self.dataset_selection.items():
            if not isinstance(values, (list, tuple)) or not values or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(f"dataset_selection.{name} must be a nonempty string array")
        positive = (
            self.steps,
            self.batch_size,
            self.learning_rate,
            self.validation_interval,
            self.checkpoint_interval,
        )
        if min(positive) <= 0 or self.weight_decay < 0.0 or self.gradient_clip <= 0.0:
            raise ValueError("training config contains invalid numeric values")
        if self.minimum_steps < 0 or self.minimum_steps > self.steps:
            raise ValueError("minimum_steps must lie in [0, steps]")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive when enabled")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.checkpoint_selection not in CHECKPOINT_SELECTIONS:
            raise ValueError("unsupported checkpoint_selection strategy")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["model"] = dict(self.model)
        value["dataset_selection"] = {
            name: list(values) for name, values in self.dataset_selection.items()
        }
        return value

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=2) + "\n"

    @property
    def resolved_sha256(self) -> str:
        """旧默认 `median_then_p95` 不进哈希，P1 v1 checkpoint 的 `training_config_sha256` 保持可复核。"""

        value = self.to_dict()
        if value["checkpoint_selection"] == "median_then_p95":
            del value["checkpoint_selection"]
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingConfig":
        fields = set(cls.__dataclass_fields__)
        unknown = set(value) - fields
        if unknown:
            raise ValueError(f"training config contains unsupported fields: {sorted(unknown)}")
        required = {"pipeline", "stage", "capacity", "model"}
        missing = required - set(value)
        if missing:
            raise ValueError(f"training config is missing required fields: {sorted(missing)}")
        return cls(**{name: value[name] for name in fields if name in value})

    @classmethod
    def from_json(cls, text: str) -> "TrainingConfig":
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("training config root must be an object")
        return cls.from_dict(value)

    @classmethod
    def load(cls, path: Path | str) -> "TrainingConfig":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
