from __future__ import annotations

from dataclasses import dataclass, field, fields
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from ncls.core.identity import sha256_file, sha256_json


@dataclass
class TrainingCheckpoint:
    """唯一训练状态；来源记录不参与加载门禁。"""

    method_key: str
    training_config: Mapping[str, Any]
    model_state: Mapping[str, torch.Tensor]
    global_step: int = 0
    phase_index: int = 0
    phase_name: str = "initialization"
    phase_step: int = 0
    phase_optimization_state: Mapping[str, Any] = field(default_factory=dict)
    rng_state: Mapping[str, Any] = field(default_factory=dict)
    query_stream_state: Mapping[str, Any] = field(default_factory=dict)
    source_contracts: tuple[Mapping[str, Any], ...] = ()
    source_snapshot_ids: tuple[str, ...] = ()
    reference_program_identity: str = ""
    reference_execution_plan_identity: str = ""
    native_asset_collection_identity: str = ""
    query_stream_identity: str = ""
    gradient_coverage: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    validation_state: Mapping[str, Any] = field(default_factory=dict)
    selection_evidence: Mapping[str, Any] = field(default_factory=dict)
    resolved_plan: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def training_config_sha256(self) -> str:
        return sha256_json(self.training_config)

    @property
    def source(self) -> Mapping[str, Any]:
        return self.training_config["source"]

    @property
    def model_payload(self) -> dict[str, Any]:
        return {
            "training_config": dict(self.training_config),
            "model_state": dict(self.model_state),
            "source_snapshot_ids": list(self.source_snapshot_ids),
        }

    def to_payload(self) -> dict[str, Any]:
        return {"format": "ncls.checkpoint", **{item.name: getattr(self, item.name) for item in fields(self)}}


def save_checkpoint(path: Path | str, checkpoint: TrainingCheckpoint) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(checkpoint.to_payload(), temporary)
    os.replace(temporary, target)
    return sha256_file(target)


def load_checkpoint(
    path: Path | str, *, map_location: str | torch.device = "cpu",
) -> TrainingCheckpoint:
    value = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(value, dict) or value.pop("format", None) != "ncls.checkpoint":
        raise ValueError("文件不是当前 ncls checkpoint")
    return TrainingCheckpoint(**value)
