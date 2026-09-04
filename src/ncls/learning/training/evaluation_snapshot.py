from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, cast

import torch

from ncls.core.identity import require_sha256, sha256_file, sha256_json
from ncls.learning.methods import get_method_plugin, method_plugins

from .checkpoint import TrainingCheckpoint
from .checkpoint_v1 import load_training_checkpoint_v1
from .plan import ResolvedTrainingPlan
from .readiness import CheckpointReadinessMode, assess_checkpoint_readiness


@dataclass(frozen=True)
class EvaluationSnapshot:
    public_method_key: str
    implementation_key: str
    checkpoint_sha256: str
    global_step: int
    phase_name: str
    source: Mapping[str, Any]
    source_snapshot_ids: tuple[str, ...]
    data_identity: Mapping[str, str]
    deployment_payload: Mapping[str, Any]
    readiness: Mapping[str, Mapping[str, Any]]
    legacy_v4: bool

    def __post_init__(self) -> None:
        require_sha256("evaluation snapshot checkpoint", self.checkpoint_sha256)
        if not self.public_method_key or not self.implementation_key:
            raise ValueError("evaluation snapshot method identity is required")
        if self.global_step < 0 or not self.phase_name:
            raise ValueError("evaluation snapshot cursor is invalid")
        if not self.source_snapshot_ids:
            raise ValueError("evaluation snapshot source identities are required")
        for value in self.source_snapshot_ids:
            require_sha256("evaluation snapshot source", value)
        object.__setattr__(self, "source", MappingProxyType(dict(self.source)))
        identities = {str(name): str(value) for name, value in self.data_identity.items()}
        required_identities = {
            "data_execution_plan_identity",
            "reference_program_identity",
            "reference_execution_plan_identity",
            "native_asset_collection_identity",
            "query_stream_identity",
        }
        if set(identities) != required_identities:
            raise ValueError("evaluation snapshot data identity fields are invalid")
        for name, value in identities.items():
            require_sha256(f"evaluation snapshot {name}", value)
        object.__setattr__(self, "data_identity", MappingProxyType(identities))
        object.__setattr__(
            self, "deployment_payload", MappingProxyType(dict(self.deployment_payload))
        )
        object.__setattr__(
            self,
            "readiness",
            MappingProxyType(
                {
                    str(name): MappingProxyType(dict(value))
                    for name, value in self.readiness.items()
                }
            ),
        )

    def require_ready(self, mode: CheckpointReadinessMode) -> Mapping[str, Any]:
        try:
            value = self.readiness[mode]
        except KeyError as error:
            raise ValueError(f"evaluation snapshot has no readiness mode {mode!r}") from error
        if not bool(value.get("ready", False)):
            reasons = "; ".join(str(item) for item in value.get("reasons", ()))
            raise ValueError(f"checkpoint is not ready for {mode}: {reasons}")
        return value


def _snapshot_from_runner(
    checkpoint: TrainingCheckpoint,
    *,
    public_method_key: str,
    checkpoint_sha256: str,
    legacy_v4: bool,
    data_execution_plan_identity: str | None = None,
    resolved_plan: ResolvedTrainingPlan | None = None,
) -> EvaluationSnapshot:
    plugin = get_method_plugin(public_method_key)
    checkpoint.validate_method(plugin.descriptor)
    training_config = dict(checkpoint.training_config)
    source = training_config.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("checkpoint training config has no source definition")
    readiness = {
        mode: assess_checkpoint_readiness(
            checkpoint,
            plugin.descriptor,
            mode=cast(CheckpointReadinessMode, mode),
        ).to_dict()
        for mode in ("formal", "diagnostic-evaluator", "visual-diagnostic")
    }
    return EvaluationSnapshot(
        public_method_key,
        checkpoint.method_key,
        checkpoint_sha256,
        checkpoint.global_step,
        checkpoint.phase_name,
        dict(source),
        tuple(checkpoint.source_snapshot_ids),
        {
            "data_execution_plan_identity": (
                data_execution_plan_identity
                if data_execution_plan_identity is not None
                else sha256_json(
                    {"schema": "ncls.legacy-data-execution-plan-unavailable@1"}
                )
            ),
            "reference_program_identity": checkpoint.reference_program_identity,
            "reference_execution_plan_identity": checkpoint.reference_execution_plan_identity,
            "native_asset_collection_identity": checkpoint.native_asset_collection_identity,
            "query_stream_identity": checkpoint.query_stream_identity,
        },
        {
            "model_state": dict(checkpoint.model_state),
            "training_config": training_config,
            "source_snapshot_ids": list(checkpoint.source_snapshot_ids),
            **(
                {}
                if resolved_plan is None
                else {"resolved_plan": resolved_plan.to_dict()}
            ),
        },
        readiness,
        legacy_v4,
    )


def _public_key_for_checkpoint(checkpoint: TrainingCheckpoint) -> str:
    matches = tuple(
        plugin.key
        for plugin in method_plugins()
        if plugin.descriptor.method_key == checkpoint.method_key
    )
    if len(matches) != 1:
        raise ValueError("legacy checkpoint method is not one current explicit plugin")
    return matches[0]


def load_evaluation_snapshot(
    path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
) -> EvaluationSnapshot:
    target = Path(path)
    sidecar = target.with_suffix(target.suffix + ".sha256")
    if not sidecar.is_file():
        raise ValueError("evaluation checkpoint SHA-256 sidecar is missing")
    expected = sidecar.read_text(encoding="ascii").strip()
    require_sha256("evaluation checkpoint sidecar", expected)
    if sha256_file(target) != expected:
        raise ValueError("evaluation checkpoint file hash mismatch")
    raw = torch.load(target, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping):
        raise ValueError("evaluation checkpoint payload root must be an object")
    version = int(raw.get("format_version", -1))
    if raw.get("format_name") != "ncls.training-checkpoint":
        raise ValueError("unsupported evaluation checkpoint format")
    if version == 4:
        from .legacy_checkpoint import LegacyCheckpointV4Importer

        return LegacyCheckpointV4Importer().load(
            target, map_location=map_location
        )
    if version != 1:
        raise ValueError(f"unsupported evaluation checkpoint version {version}")
    checkpoint = load_training_checkpoint_v1(target, map_location=map_location)
    plan = ResolvedTrainingPlan.from_dict(checkpoint.plan_manifest)
    if plan.sha256 != checkpoint.plan_identity:
        raise ValueError("evaluation checkpoint resolved plan identity mismatch")
    plugin = get_method_plugin(str(checkpoint.method["public_key"]))
    runner = checkpoint.to_runner_checkpoint(plan=plan, plugin=plugin)
    return _snapshot_from_runner(
        runner,
        public_method_key=plugin.key,
        checkpoint_sha256=expected,
        legacy_v4=False,
        data_execution_plan_identity=str(
            checkpoint.data_identity["data_execution_plan_identity"]
        ),
        resolved_plan=plan,
    )


__all__ = ["EvaluationSnapshot", "load_evaluation_snapshot"]
