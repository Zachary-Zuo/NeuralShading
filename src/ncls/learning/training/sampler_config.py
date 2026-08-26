from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SAMPLER_NAMES = ("nvidia-diffuse-ggx9", "ltc-k2")


@dataclass(frozen=True)
class SamplerTrainingConfig:
    evaluator_pipeline: str
    evaluator_checkpoint: str
    sampler: str
    steps: int = 10_000
    batch_size: int = 16
    learning_rate: float = 3e-4
    final_learning_rate_fraction: float = 0.05
    weight_decay: float = 1e-5
    gradient_clip: float = 5.0
    validation_interval: int = 250
    checkpoint_interval: int = 250
    seed: int = 20260824
    device: str = "cuda"
    deterministic: bool = True
    schema_name: str = "unified-sampler-training-config"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_name != "unified-sampler-training-config" or self.schema_version != 1:
            raise ValueError("unsupported sampler training config")
        if not self.evaluator_pipeline or not self.evaluator_checkpoint:
            raise ValueError("sampler training requires an evaluator identity")
        if self.sampler not in SAMPLER_NAMES:
            raise ValueError("unsupported unified sampler")
        positive = (
            self.steps,
            self.batch_size,
            self.learning_rate,
            self.gradient_clip,
            self.validation_interval,
            self.checkpoint_interval,
        )
        if min(positive) <= 0 or self.weight_decay < 0.0:
            raise ValueError("sampler config contains invalid numeric values")
        if not 0.0 <= self.final_learning_rate_fraction <= 1.0 or self.seed < 0:
            raise ValueError("sampler config schedule or seed is invalid")

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
    def from_dict(cls, value: Mapping[str, Any]) -> "SamplerTrainingConfig":
        fields = set(cls.__dataclass_fields__)
        unknown = set(value) - fields
        if unknown:
            raise ValueError(f"sampler config contains unsupported fields: {sorted(unknown)}")
        required = {"evaluator_pipeline", "evaluator_checkpoint", "sampler"}
        missing = required - set(value)
        if missing:
            raise ValueError(f"sampler config is missing required fields: {sorted(missing)}")
        return cls(**{name: value[name] for name in fields if name in value})

    @classmethod
    def load(cls, path: Path | str) -> "SamplerTrainingConfig":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("sampler config root must be an object")
        return cls.from_dict(value)
