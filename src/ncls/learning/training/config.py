from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ncls.learning.models.legacy_ltc_k2_p1 import ARCHITECTURE_ID


@dataclass(frozen=True)
class TrainingConfig:
    architecture_id: str = ARCHITECTURE_ID
    representation_id: str = "legacy-ltc-k2@1"
    width: int = 64
    steps: int = 10000
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 5.0
    validation_interval: int = 250
    checkpoint_interval: int = 250
    max_validation_query_groups: int = 4096
    seed: int = 20260822
    device: str | None = None
    deterministic: bool = True
    schema_name: str = "ncls.training-config"
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_name != "ncls.training-config" or self.schema_version != 2:
            raise ValueError("unsupported training config schema")
        if self.architecture_id != ARCHITECTURE_ID or self.representation_id != "legacy-ltc-k2@1":
            raise ValueError("only the explicitly named legacy-ltc-k2 P1 baseline is currently registered")
        positive = (
            self.width,
            self.steps,
            self.batch_size,
            self.learning_rate,
            self.validation_interval,
            self.checkpoint_interval,
            self.max_validation_query_groups,
        )
        if min(positive) <= 0 or self.weight_decay < 0.0 or self.gradient_clip <= 0.0 or self.seed < 0:
            raise ValueError("training config contains invalid numeric values")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=2) + "\n"

    @property
    def resolved_sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrainingConfig:
        return cls(**{name: value[name] for name in cls.__dataclass_fields__ if name in value})

    @classmethod
    def from_json(cls, text: str) -> TrainingConfig:
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("training config root must be an object")
        return cls.from_dict(value)

    @classmethod
    def load(cls, path: Path | str) -> TrainingConfig:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
