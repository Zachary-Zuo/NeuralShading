from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ncls.learning.pipelines.legacy_ltc_k2 import PIPELINE_ID


@dataclass(frozen=True)
class TrainingConfig:
    pipeline_id: str = PIPELINE_ID
    research_stage: str = "deployment-regression"
    model_parameters: Mapping[str, Any] = field(default_factory=lambda: {"width": 64})
    dataset_selection: Mapping[str, Any] = field(default_factory=dict)
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
    selection_metric: str = "relative_l1.median"
    schema_name: str = "ncls.training-config"
    schema_version: int = 4

    def __post_init__(self) -> None:
        if self.schema_name != "ncls.training-config" or self.schema_version != 4:
            raise ValueError("unsupported training config schema")
        if "@" not in self.pipeline_id or not self.research_stage or self.selection_metric.count(".") != 1:
            raise ValueError("training config requires a versioned pipeline, research stage and selection metric")
        if not isinstance(self.model_parameters, Mapping):
            raise ValueError("model_parameters must be an object")
        if not isinstance(self.dataset_selection, Mapping):
            raise ValueError("dataset_selection must be an object")
        allowed_selection = {"state_ids", "asset_ids", "family_ids"}
        unknown_selection = set(self.dataset_selection) - allowed_selection
        if unknown_selection:
            raise ValueError(f"unsupported dataset selection fields: {sorted(unknown_selection)}")
        for name, values in self.dataset_selection.items():
            if (
                not isinstance(values, (list, tuple))
                or not values
                or any(not isinstance(value, str) or not value for value in values)
            ):
                raise ValueError(f"dataset_selection.{name} must be a nonempty string array")
        positive = (
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
        value = asdict(self)
        value["model_parameters"] = dict(self.model_parameters)
        value["dataset_selection"] = {
            name: list(values) for name, values in self.dataset_selection.items()
        }
        return value

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
