from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any, Mapping

import torch

from ncls.core.identity import require_sha256, sha256_file, sha256_json
from ncls.learning.methods.contracts import MethodPlugin

from .checkpoint import TrainingCheckpoint as RunnerCheckpointV4
from .plan import ResolvedTrainingPlan


_PUBLIC_KEY = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_DATA_IDENTITY_FIELDS = {
    "data_execution_plan_identity",
    "reference_program_identity",
    "reference_execution_plan_identity",
    "native_asset_collection_identity",
    "query_stream_identity",
    "source_contracts",
    "source_snapshot_ids",
}


def _mutable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(name): _mutable(item) for name, item in value.items()}
    if isinstance(value, list):
        return [_mutable(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_mutable(item) for item in value)
    return value


@dataclass(frozen=True)
class TrainingCheckpointV1:
    method: Mapping[str, Any]
    plan_manifest: Mapping[str, Any]
    plan_identity: str
    data_identity: Mapping[str, Any]
    global_step: int
    phase_index: int
    phase_name: str
    phase_step: int
    model_state: Mapping[str, torch.Tensor]
    optimization_state: Mapping[str, Any]
    rank_rng_state: Mapping[str, Any]
    rank_data_state: Mapping[str, Any]
    gradient_coverage: Mapping[str, Mapping[str, Any]]
    validation_state: Mapping[str, Any]
    selection_evidence: Mapping[str, Any]
    hook_state: Mapping[str, Any]
    visual_eval_probe_ids: tuple[str, ...]
    format_name: str = "ncls.training-checkpoint"
    format_version: int = 1

    def __post_init__(self) -> None:
        if self.format_name != "ncls.training-checkpoint" or self.format_version != 1:
            raise ValueError("unsupported new training checkpoint format")
        method = _mutable(self.method)
        required_method = {
            "public_key",
            "implementation_key",
            "descriptor_sha256",
            "implementation_sha256",
            "facets",
        }
        if set(method) != required_method:
            raise ValueError("training checkpoint method fields are invalid")
        if not _PUBLIC_KEY.fullmatch(str(method["public_key"])):
            raise ValueError("training checkpoint public method key is invalid")
        require_sha256("checkpoint method descriptor", str(method["descriptor_sha256"]))
        require_sha256(
            "checkpoint method implementation", str(method["implementation_sha256"])
        )
        facets = method["facets"]
        if not isinstance(facets, Mapping) or not facets:
            raise ValueError("training checkpoint facet identities are required")
        for name, value in facets.items():
            require_sha256(f"checkpoint method facet {name}", str(value))

        plan_manifest = _mutable(self.plan_manifest)
        if (
            plan_manifest.get("format_name") != "ncls.training-plan"
            or int(plan_manifest.get("format_version", -1)) != 1
        ):
            raise ValueError("training checkpoint resolved plan format is invalid")
        require_sha256("checkpoint plan identity", self.plan_identity)
        if sha256_json(plan_manifest) != self.plan_identity:
            raise ValueError("training checkpoint resolved plan hash mismatch")
        if plan_manifest.get("method_descriptor") != method:
            raise ValueError("training checkpoint method disagrees with resolved plan")

        data_identity = _mutable(self.data_identity)
        if set(data_identity) != _DATA_IDENTITY_FIELDS:
            raise ValueError("training checkpoint data identity fields are invalid")
        for name in (
            "data_execution_plan_identity",
            "reference_program_identity",
            "reference_execution_plan_identity",
            "native_asset_collection_identity",
            "query_stream_identity",
        ):
            require_sha256(f"checkpoint {name}", str(data_identity[name]))
        source_contracts = tuple(dict(item) for item in data_identity["source_contracts"])
        source_snapshot_ids = tuple(str(item) for item in data_identity["source_snapshot_ids"])
        if not source_contracts or not source_snapshot_ids:
            raise ValueError("training checkpoint source identities are required")
        for value in source_snapshot_ids:
            require_sha256("checkpoint source snapshot", value)
        data_identity["source_contracts"] = source_contracts
        data_identity["source_snapshot_ids"] = source_snapshot_ids

        training = plan_manifest.get("training")
        if not isinstance(training, Mapping):
            raise ValueError("training checkpoint plan has no training definition")
        phases = training.get("phases")
        if not isinstance(phases, (list, tuple)) or not phases:
            raise ValueError("training checkpoint plan has no phase graph")
        phase_steps = tuple(int(item["steps"]) for item in phases)
        total_steps = sum(phase_steps)
        if not 0 <= self.global_step <= total_steps:
            raise ValueError("training checkpoint global step is outside its plan")
        cursor = self.global_step
        expected_index = len(phases)
        expected_step = 0
        expected_name = "complete"
        for index, (phase, steps) in enumerate(zip(phases, phase_steps, strict=True)):
            if cursor < steps:
                expected_index = index
                expected_step = cursor
                expected_name = str(phase["name"])
                break
            cursor -= steps
        if (
            self.phase_index != expected_index
            or self.phase_step != expected_step
            or self.phase_name != expected_name
        ):
            raise ValueError("training checkpoint phase cursor disagrees with global step")

        model_state = dict(self.model_state)
        if not model_state or any(
            not isinstance(value, torch.Tensor) for value in model_state.values()
        ):
            raise ValueError("training checkpoint model state must be a nonempty tensor mapping")
        if any(
            not bool(torch.isfinite(value).all())
            for value in model_state.values()
            if value.is_floating_point()
        ):
            raise ValueError("training checkpoint model state contains non-finite tensors")
        optimization_state = dict(self.optimization_state)
        if self.phase_name == "complete" and optimization_state:
            raise ValueError("complete training checkpoint cannot retain optimization state")
        probe_ids = tuple(str(item) for item in self.visual_eval_probe_ids)
        if len(set(probe_ids)) != len(probe_ids):
            raise ValueError("training checkpoint repeats a visual eval probe identity")
        for value in probe_ids:
            require_sha256("checkpoint visual eval probe", value)

        object.__setattr__(self, "method", method)
        object.__setattr__(self, "plan_manifest", plan_manifest)
        object.__setattr__(self, "data_identity", data_identity)
        object.__setattr__(self, "model_state", model_state)
        object.__setattr__(self, "optimization_state", optimization_state)
        object.__setattr__(self, "rank_rng_state", dict(self.rank_rng_state))
        object.__setattr__(self, "rank_data_state", dict(self.rank_data_state))
        object.__setattr__(
            self,
            "gradient_coverage",
            {str(name): dict(value) for name, value in self.gradient_coverage.items()},
        )
        object.__setattr__(self, "validation_state", dict(self.validation_state))
        object.__setattr__(self, "selection_evidence", dict(self.selection_evidence))
        object.__setattr__(self, "hook_state", dict(self.hook_state))
        object.__setattr__(self, "visual_eval_probe_ids", probe_ids)

    @classmethod
    def from_runner_checkpoint(
        cls,
        checkpoint: RunnerCheckpointV4,
        *,
        plan: ResolvedTrainingPlan,
        plugin: MethodPlugin,
        data_execution_plan_identity: str,
        hook_state: Mapping[str, Any] | None = None,
        visual_eval_probe_ids: tuple[str, ...] = (),
    ) -> "TrainingCheckpointV1":
        if checkpoint.training_config_sha256 != plan.to_runtime_config().sha256:
            raise ValueError("runner checkpoint disagrees with resolved training plan")
        if checkpoint.method_key != plugin.descriptor.method_key:
            raise ValueError("runner checkpoint disagrees with method plugin")
        return cls(
            dict(plan.method_descriptor),
            plan.to_dict(),
            plan.sha256,
            {
                "data_execution_plan_identity": data_execution_plan_identity,
                "reference_program_identity": checkpoint.reference_program_identity,
                "reference_execution_plan_identity": checkpoint.reference_execution_plan_identity,
                "native_asset_collection_identity": checkpoint.native_asset_collection_identity,
                "query_stream_identity": checkpoint.query_stream_identity,
                "source_contracts": tuple(checkpoint.source_contracts),
                "source_snapshot_ids": tuple(checkpoint.source_snapshot_ids),
            },
            checkpoint.global_step,
            checkpoint.phase_index,
            checkpoint.phase_name,
            checkpoint.phase_step,
            checkpoint.model_state,
            checkpoint.phase_optimization_state,
            checkpoint.rng_state,
            checkpoint.query_stream_state,
            checkpoint.gradient_coverage,
            checkpoint.validation_state,
            checkpoint.selection_evidence,
            {} if hook_state is None else hook_state,
            visual_eval_probe_ids,
        )

    def to_runner_checkpoint(
        self, *, plan: ResolvedTrainingPlan, plugin: MethodPlugin
    ) -> RunnerCheckpointV4:
        if self.plan_identity != plan.sha256 or self.plan_manifest != plan.to_dict():
            raise ValueError("resume checkpoint resolved plan identity mismatch")
        if dict(self.method) != dict(plan.method_descriptor):
            raise ValueError("resume checkpoint method identity mismatch")
        if plugin.key != self.method["public_key"]:
            raise ValueError("resume checkpoint method plugin key mismatch")
        descriptor = plugin.descriptor
        if (
            descriptor.descriptor_sha256 != self.method["descriptor_sha256"]
            or descriptor.implementation_sha256 != self.method["implementation_sha256"]
            or dict(plugin.facet_identities) != self.method["facets"]
        ):
            raise ValueError("resume checkpoint method implementation drifted")
        config = plan.to_runtime_config()
        component_manifest = {
            "schema": "ncls.method-components@1",
            "parameter_groups": {
                name: list(values) for name, values in descriptor.parameter_groups.items()
            },
            "components": [item.to_dict() for item in descriptor.components],
        }
        checkpoint = RunnerCheckpointV4(
            descriptor.method_key,
            descriptor.descriptor_sha256,
            descriptor.implementation_sha256,
            component_manifest,
            config.to_dict(),
            config.sha256,
            sha256_json([phase.to_dict() for phase in config.phases]),
            str(self.data_identity["reference_program_identity"]),
            str(self.data_identity["reference_execution_plan_identity"]),
            str(self.data_identity["native_asset_collection_identity"]),
            str(self.data_identity["query_stream_identity"]),
            tuple(self.data_identity["source_contracts"]),
            tuple(self.data_identity["source_snapshot_ids"]),
            self.global_step,
            self.phase_index,
            self.phase_name,
            self.phase_step,
            self.selection_evidence,
            self.model_state,
            self.optimization_state,
            self.rank_rng_state,
            self.rank_data_state,
            self.gradient_coverage,
            self.validation_state,
        )
        checkpoint.validate_method(descriptor)
        return checkpoint

    def to_payload(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name,
            "format_version": self.format_version,
            "method": _mutable(self.method),
            "plan_manifest": _mutable(self.plan_manifest),
            "plan_identity": self.plan_identity,
            "data_identity": _mutable(self.data_identity),
            "global_step": self.global_step,
            "phase_index": self.phase_index,
            "phase_name": self.phase_name,
            "phase_step": self.phase_step,
            "model_state": dict(self.model_state),
            "optimization_state": _mutable(self.optimization_state),
            "rank_rng_state": _mutable(self.rank_rng_state),
            "rank_data_state": _mutable(self.rank_data_state),
            "gradient_coverage": {
                name: dict(value) for name, value in self.gradient_coverage.items()
            },
            "validation_state": _mutable(self.validation_state),
            "selection_evidence": _mutable(self.selection_evidence),
            "hook_state": _mutable(self.hook_state),
            "visual_eval_probe_ids": list(self.visual_eval_probe_ids),
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "TrainingCheckpointV1":
        fields = {
            "format_name",
            "format_version",
            "method",
            "plan_manifest",
            "plan_identity",
            "data_identity",
            "global_step",
            "phase_index",
            "phase_name",
            "phase_step",
            "model_state",
            "optimization_state",
            "rank_rng_state",
            "rank_data_state",
            "gradient_coverage",
            "validation_state",
            "selection_evidence",
            "hook_state",
            "visual_eval_probe_ids",
        }
        if set(value) != fields:
            raise ValueError(f"training checkpoint fields must be exactly {sorted(fields)}")
        return cls(
            value["method"],
            value["plan_manifest"],
            str(value["plan_identity"]),
            value["data_identity"],
            int(value["global_step"]),
            int(value["phase_index"]),
            str(value["phase_name"]),
            int(value["phase_step"]),
            value["model_state"],
            value["optimization_state"],
            value["rank_rng_state"],
            value["rank_data_state"],
            value["gradient_coverage"],
            value["validation_state"],
            value["selection_evidence"],
            value["hook_state"],
            tuple(str(item) for item in value["visual_eval_probe_ids"]),
            str(value["format_name"]),
            int(value["format_version"]),
        )


def save_training_checkpoint_v1(
    path: Path | str, checkpoint: TrainingCheckpointV1
) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(checkpoint.to_payload(), temporary)
    os.replace(temporary, target)
    digest = sha256_file(target)
    sidecar = target.with_suffix(target.suffix + ".sha256")
    temporary_sidecar = sidecar.with_name(sidecar.name + ".tmp")
    temporary_sidecar.write_text(digest + "\n", encoding="ascii")
    os.replace(temporary_sidecar, sidecar)
    return digest


def load_training_checkpoint_v1(
    path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
) -> TrainingCheckpointV1:
    target = Path(path)
    sidecar = target.with_suffix(target.suffix + ".sha256")
    if not sidecar.is_file():
        raise ValueError("training checkpoint SHA-256 sidecar is missing")
    expected = sidecar.read_text(encoding="ascii").strip()
    require_sha256("training checkpoint sidecar", expected)
    if sha256_file(target) != expected:
        raise ValueError("training checkpoint file hash mismatch")
    value = torch.load(target, map_location=map_location, weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError("training checkpoint payload root must be an object")
    if value.get("format_name") == "ncls.training-checkpoint" and int(
        value.get("format_version", -1)
    ) == 4:
        raise ValueError(
            "TrainingCheckpoint v4 is read-only legacy input and cannot resume training"
        )
    return TrainingCheckpointV1.from_payload(value)


__all__ = [
    "TrainingCheckpointV1",
    "load_training_checkpoint_v1",
    "save_training_checkpoint_v1",
]
